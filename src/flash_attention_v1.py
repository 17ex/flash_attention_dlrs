import torch
import triton
import triton.language as tl

FP32_BYTESIZE = 4 # TODO future: accomodate other types than float32.
DTYPE = torch.float32

ORDER=(0, 1) # Hardcode for now. Maybe extend to other dim orders later.

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

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 1 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d * 4 # Assuming FP32
    block_size = cdiv(M, rows_bytesize)
    B_c = min(block_size, N) # TODO verify whether this min is okay, it's not in the algorithm.
    B_r = min(block_size, d) # Should make sense though.
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)

    # L: 2 (Initialize output and statistics)
    O = torch.zeros_like(Q, device=dev)
    l = torch.zeros(N, 1, dtype=torch.float32, device=dev)
    m = torch.full((N, 1), float('-inf'), dtype=torch.float32, device=dev)

    forward_kernel[(T_r, )](
            Q,
            O,
            l,
            m,
            K,
            V,
            T_c,
            N,
            d,
            B_c,
            B_r
            )

    # TODO store l, m? For now, just return
    return O, l, m


@triton.jit
def forward_kernel(
        Q_ptr,
        O_ptr,
        l_ptr,
        m_ptr,
        K_ptr,
        V_ptr,
        T_c: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ):
    # This performs one iteration of the outer loop.
    # Note that the loops are swapped compared to Algorithm 1 in the
    # Flash Attention V1 paper, so this is run for one i,
    # 1 <= i <= T_r, with the loop over j inside.

    i = tl.program_id(axis=0) # TODO Batched. Currently non-batched only.

    # Create pointers to all *_i blocks (line 8)
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
    l_i_ptr = tl.make_block_ptr(
            l_ptr,
            (N, 1),
            (1, 1),
            (i * B_r, 0),
            (B_r, 1),
            ORDER)
    m_i_ptr = tl.make_block_ptr(
            m_ptr,
            (N, 1),
            (1, 1),
            (i * B_r, 0),
            (B_r, 1),
            ORDER)

    Q_i = tl.load(Q_i_ptr)
    # The other values only need to be stored (at the end),
    # so no need to load them. Instead, init. in SRAM directly.
    O_i= tl.zeros_like(Q_i)
    m_i= tl.full((B_r, 1), float('-inf'), tl.float32)
    l_i= tl.zeros_like(m_i)

    for j in range(T_c):

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
        K_j = tl.load(K_j_ptr)
        V_j = tl.load(V_j_ptr)

        # Compute Q_i K_j^T (line 9)
        S_ij = tl.dot(Q_i, tl.trans(K_j))

        # Verify S_ij is of shape (B_r, B_c) at compile time.
        # TODO this did not work with a tuple, hence the two checks.
        # Check why S_ij.shape == (B_r, B_c) didn't work.
        tl.static_assert(S_ij.shape[0] == B_r and S_ij.shape[1] == B_c)

        # Compute m~_ij, P~_ij, l~_ij (line 10)
        # TODO doc says there is a keep_dims=True argument which would make some of this
        # dumb adding/removing the last dimension unnecessary, but it doesn't exist
        # in my Triton (2.3.0). Check which version has this
        # TODO better do reshaping/expand dims with eg. ms_ij[:, None] ?
        ms_ij = tl.expand_dims(tl.max(S_ij, axis=1), axis=1)
        Ps_ij = tl.exp(S_ij # TODO see if this can be refactored.
                       - tl.broadcast_to(ms_ij, (B_r, B_c)))
        ls_ij = tl.expand_dims(tl.sum(Ps_ij, axis=1), axis=1)

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
               + tl.exp(ms_ij - m_i_new) * tl.dot(Ps_ij, V_j)) \
                       / l_i_new

        # Overwrite old l_i, m_i (line 13)
        l_i = l_i_new
        m_i = m_i_new

    # This loop/kernel is done (looped over all j for this i),
    # store the results and exit
    tl.store(O_i_ptr, O_i)
    tl.store(l_i_ptr, l_i)
    tl.store(m_i_ptr, m_i)
    return
