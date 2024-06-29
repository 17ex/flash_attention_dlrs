import torch
import triton
import triton.language as tl
from autotune_configs import get_fwd_autotune_config

FP32_BYTESIZE = 4 # TODO future: accomodate other types than float32.
DTYPE = torch.float32
DOT_PRECISION: tl.constexpr = "ieee"
ORDER: tl.constexpr = (0, 1)

def flash_attention_forward(
        Q,
        K,
        V,
        dev
        ) -> tuple[torch.Tensor, torch.Tensor]:

    # Takes tensors of shape (B, H, N, d), where
    # B: Batch size
    # H: Number of attention heads
    # N: Context size
    # d: Token dimension

    assert Q.dim() == 4
    assert Q.shape == K.shape and K.shape == V.shape
    assert Q.dtype == DTYPE and K.dtype == DTYPE and V.dtype == DTYPE

    B, H, N, d = Q.shape

    # Support for non-power-of-2 d
    d_pow = triton.next_power_of_2(d)

    if d_pow != d:
        # Apply padding
        pad = (0, d_pow - d)
        Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
        K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
        V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

    # Initialize output and statistics
    O = torch.empty(B, H, N, d_pow, dtype=DTYPE, device=dev)
    L = torch.empty(B, H, N, 1, dtype=DTYPE, device=dev)

    QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
    KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
    VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
    OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
    LB_stride, LH_stride, _, _ = L.stride()

    grid = lambda META: (B, H, triton.cdiv(N, META['B_r']))

    forward_kernel[grid](
            Q,
            K,
            V,
            O,
            L,
            QB_stride, QH_stride, QN_stride, Qd_stride,
            KB_stride, KH_stride, KN_stride, Kd_stride,
            VB_stride, VH_stride, VN_stride, Vd_stride,
            OB_stride, OH_stride, ON_stride, Od_stride,
            LB_stride, LH_stride,
            B, H, N, d_pow
            )

    return O[:, :, :, 0:d], L


@triton.autotune(
        configs=get_fwd_autotune_config(),
        key=['B', 'H', 'N', 'd'],
)
@triton.jit
def forward_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        O_ptr,
        L_ptr,
        QB_stride, QH_stride, QN_stride, Qd_stride,
        KB_stride, KH_stride, KN_stride, Kd_stride,
        VB_stride, VH_stride, VN_stride, Vd_stride,
        OB_stride, OH_stride, ON_stride, Od_stride,
        LB_stride, LH_stride,
        B, H, N, d: tl.constexpr, # B, H are here to re-tune the kernel when they change
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ):
    # This performs one iteration of the outer loop of FA-2
    # (more or less, the math is more like in FA-1 as of now)

    # Determine which on batch/head number and i of the outer loop
    # this kernel should operate on
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)

    T_c = tl.cdiv(N, B_c)

    # Initialize all block pointers
    Q_i_ptr = tl.make_block_ptr(
            Q_ptr + b * QB_stride + h * QH_stride,
            (N, d),
            (QN_stride, Qd_stride),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    O_i_ptr = tl.make_block_ptr(
            O_ptr + b * OB_stride + h * OH_stride,
            (N, d),
            (ON_stride, Od_stride),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    L_i_ptr = tl.make_block_ptr(
            L_ptr + b * LB_stride + h * LH_stride,
            (N, 1),
            (1, 1),
            (i * B_r, 0),
            (B_r, 1),
            ORDER)
    K_j_ptr = tl.make_block_ptr(
            K_ptr + b * KB_stride + h * KH_stride,
            (N, d),
            (KN_stride, Kd_stride),
            (0, 0),
            (B_c, d),
            ORDER)
    V_j_ptr = tl.make_block_ptr(
            V_ptr + b * VB_stride + h * VH_stride,
            (N, d),
            (VN_stride, Vd_stride),
            (0, 0),
            (B_c, d),
            ORDER)

    Q_i = tl.load(Q_i_ptr)
    # The other values only need to be stored (at the end),
    # so no need to load them. Instead, init. in SRAM directly.
    O_i= tl.zeros_like(Q_i)
    m_i= tl.full((B_r, 1), float('-inf'), tl.float32)
    l_i= tl.zeros_like(m_i)

    for _ in range(T_c):
        K_j = tl.load(K_j_ptr)
        V_j = tl.load(V_j_ptr)

        # Compute Q_i K_j^T
        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision=DOT_PRECISION)

        # Compute m_ij, P_ij, l_ij
        m_ij = tl.max(S_ij, axis=1, keep_dims=True)
        P_ij = tl.exp(S_ij - m_ij)
        l_ij = tl.sum(P_ij, axis=1, keep_dims=True)

        # Compute m_i_new, l_i_new
        m_i_new = tl.maximum(m_i, m_ij)
        l_i_new = tl.exp(m_i - m_i_new) * l_i \
                + tl.exp(m_ij - m_i_new) * l_ij

        # Calculate new O_i
        O_i = (l_i * tl.exp(m_i - m_i_new) * O_i
               + tl.exp(m_ij - m_i_new) * tl.dot(P_ij, V_j, input_precision=DOT_PRECISION)) \
                       / l_i_new

        # Overwrite old l_i, m_i
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
    tl.store(L_i_ptr, L_i)
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

    assert Q.dim() == 4
    assert Q.shape ==  K.shape  and V.shape == K.shape
    assert Q.dtype == DTYPE and K.dtype == DTYPE and V.dtype == DTYPE

    B, H, N, d = Q.shape

    d_pow = triton.next_power_of_2(d)

    if d_pow != d:
        # Apply padding
        pad = (0, d_pow - d)
        Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
        K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
        V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

    # Determine block sizes
    rows_bytesize = FP32_BYTESIZE * d_pow * 4 # Assuming FP32
    block_size = triton.cdiv(M, rows_bytesize)
    B_c = min(block_size, N)
    B_r = min(block_size, d_pow)
    T_r = triton.cdiv(N, B_r)
    T_c = triton.cdiv(N, B_c)
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
    _lock_dQ = torch.zeros(B, H, 2 * T_r, dtype=torch.int32, device=dev)
    _written_dQ = torch.zeros(B, H, 2 * T_r, dtype=torch.int32, device=dev)

    QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
    KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
    VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
    OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
    dQB_stride, dQH_stride, dQN_stride, dQd_stride = dQ.stride()
    dKB_stride, dKH_stride, dKN_stride, dKd_stride = dK.stride()
    dVB_stride, dVH_stride, dVN_stride, dVd_stride = dV.stride()
    dOB_stride, dOH_stride, dON_stride, dOd_stride = dO.stride()
    LB_stride, LH_stride, _, _ = L.stride()
    _lock_dQ_B_stride, _lock_dQ_H_stride, _ = _lock_dQ.stride()
    _written_dQ_B_stride, _written_dQ_H_stride, _ = _written_dQ.stride()

    backward_kernel[(B, H, T_c)](
            Q, K, V, O,
            dQ, dK, dV, dO,
            L,
            _lock_dQ, _written_dQ,
            QB_stride, QH_stride, QN_stride, Qd_stride,
            KB_stride, KH_stride, KN_stride, Kd_stride,
            VB_stride, VH_stride, VN_stride, Vd_stride,
            OB_stride, OH_stride, ON_stride, Od_stride,
            dQB_stride, dQH_stride, dQN_stride, dQd_stride,
            dKB_stride, dKH_stride, dKN_stride, dKd_stride,
            dVB_stride, dVH_stride, dVN_stride, dVd_stride,
            dOB_stride, dOH_stride, dON_stride, dOd_stride,
            LB_stride, LH_stride,
            _lock_dQ_B_stride, _lock_dQ_H_stride,
            _written_dQ_B_stride, _written_dQ_H_stride,
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
        QB_stride, QH_stride, QN_stride, Qd_stride,
        KB_stride, KH_stride, KN_stride, Kd_stride,
        VB_stride, VH_stride, VN_stride, Vd_stride,
        OB_stride, OH_stride, ON_stride, Od_stride,
        dQB_stride, dQH_stride, dQN_stride, dQd_stride,
        dKB_stride, dKH_stride, dKN_stride, dKd_stride,
        dVB_stride, dVH_stride, dVN_stride, dVd_stride,
        dOB_stride, dOH_stride, dON_stride, dOd_stride,
        LB_stride, LH_stride,
        lock_dQ_B_stride, lock_dQ_H_stride,
        written_dQ_B_stride, written_dQ_H_stride,
        T_r: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ) -> None:

    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    j = tl.program_id(axis=2)

    # Create pointers and load the *_j blocks (line 6)
    K_j_ptr = tl.make_block_ptr(
            K_ptr + b * KB_stride + h * KH_stride,
            (N, d),
            (KN_stride, Kd_stride),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    V_j_ptr = tl.make_block_ptr(
            V_ptr + b * VB_stride + h * VH_stride,
            (N, d),
            (VN_stride, Vd_stride),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    dK_j_ptr = tl.make_block_ptr(
            dK_ptr + b * dKB_stride + h * dKH_stride,
            (N, d),
            (dKN_stride, dKd_stride),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    dV_j_ptr = tl.make_block_ptr(
            dV_ptr + b * dVB_stride + h * dVH_stride,
            (N, d),
            (dVN_stride, dVd_stride),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    Q_i_ptr = tl.make_block_ptr(
            Q_ptr + b * QB_stride + h * QH_stride,
            (N, d),
            (QN_stride, Qd_stride),
            (0, 0),
            (B_r, d),
            ORDER)
    O_i_ptr = tl.make_block_ptr(
            O_ptr + b * OB_stride + h * OH_stride,
            (N, d),
            (ON_stride, Od_stride),
            (0, 0),
            (B_r, d),
            ORDER)
    dO_i_ptr = tl.make_block_ptr(
            dO_ptr + b * dOB_stride + h * dOH_stride,
            (N, d),
            (dON_stride, dOd_stride),
            (0, 0),
            (B_r, d),
            ORDER)
    dQ_i_ptr = tl.make_block_ptr(
            dQ_ptr + b * dQB_stride + h * dQH_stride,
            (N, d),
            (dQN_stride, dQd_stride),
            (0, 0),
            (B_r, d),
            ORDER)
    L_i_ptr = tl.make_block_ptr(
            L_ptr + b * LB_stride + h * LH_stride,
            (N, 1),
            (1, 1),
            (0, 0),
            (B_r, 1),
            ORDER)

    K_j = tl.load(K_j_ptr)
    V_j = tl.load(V_j_ptr)
    dK_j = tl.zeros_like(K_j)
    dV_j = tl.zeros_like(V_j)

    lock_dQ_i = lock_dQ + b * lock_dQ_B_stride + h * lock_dQ_H_stride
    written_dQ_i = written_dQ + b * written_dQ_B_stride + h * written_dQ_H_stride

    for _ in range(T_r):

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

        # Advance the *_i_ptrs
        Q_i_ptr = tl.advance(Q_i_ptr, (B_r, 0))
        O_i_ptr = tl.advance(O_i_ptr, (B_r, 0))
        dO_i_ptr = tl.advance(dO_i_ptr, (B_r, 0))
        dQ_i_ptr = tl.advance(dQ_i_ptr, (B_r, 0))
        L_i_ptr = tl.advance(L_i_ptr, (B_r, 0))
        lock_dQ_i += 1
        written_dQ_i += 1


    tl.store(dK_j_ptr, dK_j)
    tl.store(dV_j_ptr, dV_j)
    return
