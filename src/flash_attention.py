import torch
import triton
import triton.language as tl

FP32_BYTESIZE = 4 # TODO future: accomodate other types than float32.
DTYPE = torch.float32

DOT_PRECISION: tl.constexpr = "ieee"

ORDER: tl.constexpr = (0, 1) # Hardcode for now. Maybe extend to other dim orders later.

def cdiv(a, b):
    return (a + b - 1) // b

def flash_attention_forward(
        Q,
        K,
        V,
        N, # Number of rows of Q,K,V
        d, # Number of columns of Q,K,V
        M, # SRAM size
        dev
        ):

    assert Q.dim_order() == ORDER and K.dim_order() == ORDER and V.dim_order() == ORDER
    shape = torch.Size([N, d])
    assert Q.shape == shape and K.shape == shape and V.shape == shape
    assert Q.dtype == DTYPE and K.dtype == DTYPE and V.dtype == DTYPE

    d_pow = triton.next_power_of_2(d)

    if d_pow != d:
        # Apply padding
        pad = (0, d_pow - d)
        Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
        K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
        V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 1 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d_pow * 4 # Assuming FP32
    block_size = cdiv(M, rows_bytesize)
    B_c = min(block_size, N) # TODO verify whether this min is okay, it's not in the algorithm.
    B_r = min(block_size, d_pow) # Should make sense though.
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)

    # L: 2 (Initialize output and statistics)
    O = torch.empty(N, d_pow, dtype=DTYPE, device=dev)
    L = torch.empty(N, 1, dtype=DTYPE, device=dev)

    forward_kernel[(T_r, )](
            Q,
            O,
            L,
            K,
            V,
            T_c,
            T_r,
            N,
            d_pow,
            B_c,
            B_r
            )

    # TODO store l, m? For now, just return
    return O[:, 0:d], L


@triton.jit
def forward_kernel(
        Q_ptr,
        O_ptr,
        L_ptr,
        K_ptr,
        V_ptr,
        T_c: tl.constexpr,
        T_r: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ):

    tl.static_print(f"JIT-compiling flash attention v1 forward kernel for:")
    tl.static_print(f"B_c={B_c}, B_r={B_r}, T_r={T_r}, T_c={T_c}, N={N}, d={d}")
    # This performs one iteration of the outer loop.
    # Note that the loops are swapped compared to Algorithm 1 in the
    # Flash Attention V1 paper, so this is run for one i,
    # 1 <= i <= T_r, with the loop over j inside.

    i = tl.program_id(axis=0) # TODO Batched. Currently non-batched only.

    # Initialize all block pointers
    Q_i_ptr = tl.make_block_ptr(
            Q_ptr,
            (N, d),
            (d, 1),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    O_i_ptr = tl.make_block_ptr(
            O_ptr,
            (N, d),
            (d, 1),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    L_i_ptr = tl.make_block_ptr(
            L_ptr,
            (N, 1),
            (1, 1),
            (i * B_r, 0),
            (B_r, 1),
            ORDER)
    K_j_ptr = tl.make_block_ptr(
            K_ptr,
            (N, d),
            (d, 1),
            (0, 0),
            (B_c, d),
            ORDER)
    V_j_ptr = tl.make_block_ptr(
            V_ptr,
            (N, d),
            (d, 1),
            (0, 0),
            (B_c, d),
            ORDER)

    Q_i = tl.load(Q_i_ptr)
    # The other values only need to be stored (at the end),
    # so no need to load them. Instead, init. in SRAM directly.
    O_i= tl.zeros_like(Q_i)
    m_i= tl.full((B_r, 1), float('-inf'), tl.float32)
    l_i= tl.zeros_like(m_i)

    for j in range(T_c):
        K_j = tl.load(K_j_ptr)
        V_j = tl.load(V_j_ptr)

        # Compute Q_i K_j^T (line 9)
        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision=DOT_PRECISION)

        # Verify S_ij is of shape (B_r, B_c) at compile time.
        # TODO this did not work with a tuple, hence the two checks.
        # Check why S_ij.shape == (B_r, B_c) didn't work.
        tl.static_assert(S_ij.shape[0] == B_r and S_ij.shape[1] == B_c)

        # Compute m~_ij, P~_ij, l~_ij (line 10)
        ms_ij = tl.max(S_ij, axis=1, keep_dims=True)
        Ps_ij = tl.exp(S_ij - ms_ij)
        ls_ij = tl.sum(Ps_ij, axis=1, keep_dims=True)

        # Compute m_i_new, l_i_new (line 11)
        m_i_new = tl.maximum(m_i, ms_ij)
        l_i_new = tl.exp(m_i - m_i_new) * l_i \
                + tl.exp(ms_ij - m_i_new) * ls_ij
        #m_i_new = tl.maximum(m_i, tl.reshape(mw_ij, (B_r, )))
        # prev_coeff = tl.exp(m_i - m_i_new)
        # curr_coeff = tl.exp(ms_ij - m_i_new)
        # l_i_new = prev_coeff * l_i + curr_coeff * lw_ij

        # Calculate new O_i (line 12)
        O_i = (l_i * tl.exp(m_i - m_i_new) * O_i
               + tl.exp(ms_ij - m_i_new) * tl.dot(Ps_ij, V_j, input_precision=DOT_PRECISION)) \
                       / l_i_new

        # Overwrite old l_i, m_i (line 13)
        l_i = l_i_new
        m_i = m_i_new

        K_j_ptr = tl.advance(K_j_ptr, (B_c, 0))
        V_j_ptr = tl.advance(V_j_ptr, (B_c, 0))

    # This loop/kernel is done (looped over all j for this i),
    # store the results and exit
    # Writes to O are not masked (I don't think that's really possible here),
    # whatever function called this should ensure to only read the appropriate sub-tensor
    L_i = m_i + tl.log(l_i)
    tl.store(O_i_ptr, O_i)
    tl.store(L_i_ptr, L_i) # TODO Check if you can get away with not masking L_i
    return



def flash_attention_backward(
        Q,
        K,
        V,
        O,
        dO,
        L,
        M,
        dev
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert Q.dim_order() == ORDER and K.dim_order() == ORDER and V.dim_order() == ORDER
    assert Q.shape ==  K.shape  and V.shape == K.shape
    assert Q.dtype == DTYPE and K.dtype == DTYPE and V.dtype == DTYPE

    N, d = Q.shape[-2:]

    d_pow = triton.next_power_of_2(d)

    if d_pow != d:
        # Apply padding
        pad = (0, d_pow - d)
        Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
        K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
        V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 1 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d_pow * 4 # Assuming FP32
    block_size = cdiv(M, rows_bytesize)
    B_c = min(block_size, N) # TODO verify whether this min is okay, it's not in the algorithm.
    B_r = min(block_size, d_pow) # Should make sense though.
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    # int64 would make more sense here,
    # because atomic ops pointers only work on aligned memory,
    # but I need to compare with constexprs, and then the compiler
    # complains because it can't compare int64 with int32,
    # and I don't know how to specify that a literal should be an
    # int64 and not int32. I don't even know if you can do this easily.
    # So, instead we double the length, and only index every 2nd element.
    _lock_dQ = torch.zeros(2 * T_r, dtype=torch.int32, device=dev)
    _written_dQ = torch.zeros(2 * T_r, dtype=torch.int32, device=dev)

    backward_kernel[(T_c, )](
            Q, K, V, O,
            dQ, dK, dV, dO,
            L,
            _lock_dQ, _written_dQ,
            T_c,
            T_r,
            N,
            d,
            B_c,
            B_r
            )

    return dQ, dK, dV


@triton.jit
def backward_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr,
        dQ_ptr, dK_ptr, dV_ptr, dO_ptr,
        L_ptr,
        lock_dQ, written_dQ,
        T_c: tl.constexpr,
        T_r: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ) -> None:

    # Loop over j < T_c first (parallel over different kernel launches),
    # and have the inner loop over i < T_r within each kernel
    j = tl.program_id(axis=0)
    # Create pointers and load the *_j blocks (line 6)
    K_j_ptr = tl.make_block_ptr(
            K_ptr,
            (N, d),
            (d, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    V_j_ptr = tl.make_block_ptr(
            V_ptr,
            (N, d),
            (d, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    dK_j_ptr = tl.make_block_ptr(
            dK_ptr,
            (N, d),
            (d, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    dV_j_ptr = tl.make_block_ptr(
            dV_ptr,
            (N, d),
            (d, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    K_j = tl.load(K_j_ptr)
    V_j = tl.load(V_j_ptr)
    dK_j = tl.zeros_like(K_j)
    dV_j = tl.zeros_like(V_j)

    for i in range(T_r):
        # def. *_i ptrs
        Q_i_ptr = tl.make_block_ptr(
                Q_ptr,
                (N, d),
                (d, 1),
                (i * B_r, 0),
                (B_r, d),
                ORDER)
        O_i_ptr = tl.make_block_ptr(
                O_ptr,
                (N, d),
                (d, 1),
                (i * B_r, 0),
                (B_r, d),
                ORDER)
        dO_i_ptr = tl.make_block_ptr(
                dO_ptr,
                (N, d),
                (d, 1),
                (i * B_r, 0),
                (B_r, d),
                ORDER)
        dQ_i_ptr = tl.make_block_ptr(
                dQ_ptr,
                (N, d),
                (d, 1),
                (i * B_r, 0),
                (B_r, d),
                ORDER)
        L_i_ptr = tl.make_block_ptr(
                L_ptr,
                (N, 1),
                (1, 1),
                (i * B_r, 0),
                (B_r, 1),
                ORDER)
        lock_dQ_i = lock_dQ + i
        written_dQ_i = written_dQ + i

        # Load *_i blocks
        Q_i = tl.load(Q_i_ptr)
        O_i = tl.load(O_i_ptr)
        dO_i = tl.load(dO_i_ptr)
        L_i = tl.load(L_i_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision=DOT_PRECISION)

        P_ij = tl.exp(S_ij - L_i)

        dV_j = dV_j + tl.dot(tl.trans(P_ij), dO_i, input_precision=DOT_PRECISION)

        dP_ij = tl.dot(dO_i, tl.trans(V_j), input_precision=DOT_PRECISION)

        D_i = tl.sum(dO_i * O_i, axis=1, keep_dims=True)

        dS_ij = P_ij * (dP_ij - D_i)

        dK_j = dK_j + tl.dot(tl.trans(dS_ij), Q_i, input_precision=DOT_PRECISION)

        # Writes to dQ_i need to be guarded, since multiple threads can try to do this at the
        # same time, leading to race conditions.
        # The below code guardes against this using a locking strategy to ensure
        # only one thread can update dQ_i at a time.
        while tl.atomic_cas(lock_dQ_i, 0, 1) == 1:
            # Wait for the lock to be released, and acquire it if it's released.
            tl.device_print("TODO: delete me")
            # TODO
            # For some reason, if this device print is not here, then this kernel will hang.
            # This probably has something to do with an empty while loop?
            # Figure out why or if I'm doing something wrong, or put some dummy noop here.
            # In the tutorials, this while: pass is used just like here. So it should (?) work.
            # Maybe check if the tutorial code yields the same problem.
            pass
        # ============== dQ_i (and written_dQ_i) Mem access is protected by a lock in this snippet.
        count = tl.load(written_dQ_i)
        if count == 0:
        # if tl.atomic_xchg(written_dQ_i, 1) == 1:
            # dQ_i was not written before, so it is uninitialized
            dQ_i = tl.zeros_like(Q_i)
            tl.atomic_xchg(written_dQ_i, 1)
        else:
            # dQ_i was written to before
            dQ_i = tl.load(dQ_i_ptr)
        dQ_i = dQ_i + tl.dot(dS_ij, K_j, input_precision=DOT_PRECISION)
        tl.store(dQ_i_ptr, dQ_i)
        # ==============
        # Release the lock again
        tl.atomic_xchg(lock_dQ_i, 0)

    tl.store(dK_j_ptr, dK_j)
    tl.store(dV_j_ptr, dV_j)

    return
