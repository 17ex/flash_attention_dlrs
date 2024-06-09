import numpy as np
import torch
import triton
import triton.language as tl

FP32_BYTESIZE = 4 # TODO future: accomodate other types than float32.

def cdiv(a, b):
    return (a + b - 1) // b

def flash_attention_forward(
        Q,
        K,
        V,
        N, # Number of rows of Q,K,V
        d, # Number of columns of Q,K,V
        M: tl.constexpr # SRAM size
        ):

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 1 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d # Assuming FP32
    B_c = cdiv(M, rows_bytesize)
    B_r = min(B_c, d)
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)

    # L: 2 (Initialize output and statistics)
    O = torch.zeros_like(Q)
    l = torch.zeros(N)
    m = torch.full((N, ), -np.inf, dtype=torch.float32)

    # Check if below can be used like this:
    Q_stride = Q.stride()
    Q_order = Q.dim_order()
    K_stride = K.stride()
    K_order = K.dim_order() # TODO apparently K.T.dim_order() is just swapped -> swap instead of transpose op?
    V_stride = V.stride()
    V_order = V.dim_order()
    O_stride = O.stride()
    O_order = O.dim_order()
    m_stride = m.stride()
    m_order = m.dim_order()
    l_stride = l.stride()
    l_order = l.dim_order()

    forward_kernel[(T_c, )](
            Q, Q_stride, Q_order,
            O, O_stride, O_order,
            l, l_stride, l_order,
            m, m_stride, m_order,
            K, K_stride, K_order,
            V, V_stride, V_order,
            T_r,
            N,
            d,
            B_c,
            B_r,
            (B_c, d),
            (B_r, d),
            (B_r, )
            )

    # TODO store l, m? For now, just return
    return O, l, m


@triton.jit
def forward_kernel(
        Q_ptr, Q_stride, Q_order,
        O_ptr, O_stride, O_order,
        l_ptr, l_stride, l_order, # TODO l, m params can probably be omitted/set constant
        m_ptr, m_stride, m_order,
        K_ptr, K_stride, K_order,
        V_ptr, V_stride, V_order,
        T_r,
        N,
        d,
        B_c,
        B_r,
        KV_block_shape,
        OQ_block_shape,
        statistics_vec_shape
        ):

    j = tl.program_id(axis=0)

    K_j_ptr = tl.make_block_ptr(
            K_ptr,
            (N, d),
            K_stride,
            (j * B_c, 0),
            KV_block_shape,
            K_order)
    V_j_ptr = tl.make_block_ptr(
            V_ptr,
            (N, d),
            V_stride,
            (j * B_c, 0),
            KV_block_shape,
            V_order)

    forward_outer(
            Q_ptr, Q_stride, Q_order,
            O_ptr, O_stride, O_order,
            l_ptr, l_stride, l_order, # | TODO l, m params can probably be omitted/set constant
            m_ptr, m_stride, m_order, # |
            K_j_ptr,
            V_j_ptr,
            T_r,
            N,
            d,
            B_r,
            OQ_block_shape,
            statistics_vec_shape
            )


@triton.jit
def forward_outer(
        Q_ptr, Q_stride, Q_order,
        O_ptr, O_stride, O_order,
        l_ptr, l_stride, l_order, # | TODO l, m params can probably be omitted/set constant
        m_ptr, m_stride, m_order, # |
        K_j_ptr,
        V_j_ptr,
        T_r,
        N,
        d,
        B_r,
        OQ_block_shape,
        statistics_vec_shape
        ):

    # L: 6
    K_j = tl.load(K_j_ptr) # | TODO padding_option or boundary-check
    V_j = tl.load(V_j_ptr) # |

    # TODO staging (num_stages)
    # Look into what staging was again, forgot what I meant by this
    # L: 7
    for i in range(T_r):

        # make *_i_ptrs for the blocks that are loaded in forward_inner
        # TODO
        # maybe use advance? -> only call make_block_ptr once, then advance
        Q_i_ptr = tl.make_block_ptr(
                Q_ptr,
                (N, d),
                Q_stride,
                (i * B_r, 0),
                OQ_block_shape,
                Q_order)
        O_i_ptr = tl.make_block_ptr(
                O_ptr,
                (N, d),
                O_stride,
                (i * B_r, 0),
                OQ_block_shape,
                O_order)
        l_i_ptr = tl.make_block_ptr(
                l_ptr,
                (N, ),
                l_stride,
                (i * B_r, ),
                statistics_vec_shape,
                l_order)
        m_i_ptr = tl.make_block_ptr(
                m_ptr,
                (N, ),
                m_stride,
                (i * B_r, ),
                statistics_vec_shape,
                m_order)

        forward_inner(
                Q_i_ptr,
                O_i_ptr,
                l_i_ptr,
                m_i_ptr,
                tl.trans(K_j),
                V_j
                )



@triton.jit
def forward_inner(
        Q_i_ptr,
        O_i_ptr,
        l_i_ptr,
        m_i_ptr,
        K_jT,
        V_j
        ):
    # This function is for the inner loop (Lines 8-13 in Algorithm 1, Flash attention paper)
    # It takes as arguments all of the correctly set block pointers

    # L: 8 (Load to SRAM)
    Q_i = tl.load(Q_i_ptr) # | TODO padding_option or boundary-check
    O_i = tl.load(O_i_ptr) # |
    l_i = tl.load(l_i_ptr) # |
    m_i = tl.load(l_i_ptr) # |

    # L: 9 (Q_i * K_j^T)
    S_ij = tl.dot(Q_i, K_jT)

    # L: 10 (~ Softmax for this block)
    mw_ij = tl.max(S_ij, axis=1)
    Pw_ij = tl.exp(S_ij - mw_ij)
    lw_ij = tl.sum(Pw_ij, axis=1)

    # L: 11 (~ Calculate new softmax combining above with previous data)
    m_i_new = tl.maximum(m_i, mw_ij)
    prev_coeff = tl.exp(m_i - m_i_new)
    curr_coeff = tl.exp(mw_ij - m_i_new)
    l_i_new = prev_coeff * l_i + curr_coeff * lw_ij
    # TODO Determine if one should use fma here
    # Probably compiler does optimize anyways ?

    # L: 12 (Calculate new O_i)
    l_i_new_inv = 1 / l_i_new
    # TODO find out if below works (automatic broadcasting?)
    O_i_new = l_i_new_inv * (l_i * prev_coeff * O_i + curr_coeff * tl.dot(Pw_ij, V_j))

    # L: 12-13 (Write to HBM)
    tl.store(O_i_ptr, O_i_new)
    tl.store(l_i_ptr, l_i_new)
    tl.store(m_i_ptr, m_i_new)
    return None



# Backward pass
# =============
# Implementing Algorithm 4 of the Flash Attention v1 paper (Appendix B.4)


def flash_attention_backward(
        Q,
        K,
        V,
        O,
        dO,
        l,
        m,
        N, # Number of rows of Q,K,V
        d, # Number of columns of Q,K,V
        M: tl.constexpr # SRAM size
        ):

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 2-3 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d # Assuming FP32
    B_c = cdiv(M, rows_bytesize)
    B_r = min(B_c, d)
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)

    # L: 5
    dQ = torch.zeros(N, d)
    dK = torch.zeros(N, d)
    dV = torch.zeros(N, d)

    Q_stride = Q.stride()
    Q_order = Q.dim_order()
    K_stride = K.stride()
    K_order = K.dim_order()
    V_stride = V.stride()
    V_order = V.dim_order()
    O_stride = O.stride()
    O_order = O.dim_order()
    m_stride = m.stride()
    m_order = m.dim_order()
    l_stride = l.stride()
    l_order = l.dim_order()
    dQ_stride = dQ.stride()
    dQ_order = dQ.dim_order()
    dK_stride = dK.stride()
    dK_order = dK.dim_order()
    dV_stride = dV.stride()
    dV_order = dV.dim_order()

    backward_kernel[(T_c, )](
        Q, Q_stride, Q_order,
        O, O_stride, O_order,
        l, l_stride, l_order, # TODO l, m params can probably be omitted/set constant
        m, m_stride, m_order,
        K, K_stride, K_order,
        V, V_stride, V_order,
        dQ, dQ_stride, dQ_order,
        dK, dK_stride, dK_order,
        dV, dV_stride, dV_order,
        T_r,
        N,
        d,
        B_c,
        B_r,
        (B_c, d),
        (B_r, d),
        (B_r, )
        )

    # L: 26
    return dQ, dK, dV


@triton.jit
def backward_kernel(
        Q_ptr, Q_stride, Q_order,
        O_ptr, O_stride, O_order,
        l_ptr, l_stride, l_order, # TODO l, m params can probably be omitted/set constant
        m_ptr, m_stride, m_order,
        K_ptr, K_stride, K_order,
        V_ptr, V_stride, V_order,
        dQ_ptr, dQ_stride, dQ_order,
        dK_ptr, dK_stride, dK_order,
        dV_ptr, dV_stride, dV_order,
        T_r,
        N,
        d,
        B_c,
        B_r,
        KV_block_shape,
        OQ_block_shape,
        lm_vec_shape
        ):
    return None
