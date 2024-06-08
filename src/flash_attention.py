import numpy as np
import torch
import triton
import triton.language as tl

# TODO this is garbage.
@triton.jit
def flash_attention_forward(
        Q,
        K,
        V,
        N, # Number of rows of Q,K,V
        d, # Number of columns of Q,K,V
        M: tl.constexpr # SRAM size
        ):

    # Determine block sizes
    rows_bytesize = 4 * d
    B_c = tl.cdiv(M, rows_bytesize)
    B_r = tl.min(B_c, d)
    T_r = tl.cdiv(N, B_r)
    T_c = tl.cdiv(N, B_c)

    # Initialize values
    O = tl.zeros_like(Q)
    l = tl.zeros(N)
    m = tl.full((N, ), -np.inf, tl.float32)


@triton.jit
def forward_kernel(
        Q_ptr, Q_shape, Q_stride, Q_order,
        O_ptr, O_shape, O_stride, O_order,
        l_ptr, l_shape, l_stride, l_order, # TODO l, m params can probably be omitted/set constant
        m_ptr, m_shape, m_stride, m_order,
        K_ptr,
        V_ptr,
        T_r,
        N,
        d,
        B_c,
        B_r,
        ):
    # TODO determine program id, vars, etc.

    # TODO block pointer / load K/V_j
    K_j_ptr = tl.make_block_ptr(
            K_ptr,
            (N, d),
            (1, 1), # TODO deal with strides later on
            (), # TODO offset. This is the hard one
            (B_c, d),
            () # What is order?


            # TODO different order -> don't need tl.trans ?
            )

    # TODO l, m need to be stored for the backward pass



@triton.jit
def forward_outer(
        Q_ptr, Q_shape, Q_stride, Q_order,
        O_ptr, O_shape, O_stride, O_order,
        l_ptr, l_shape, l_stride, l_order, # | TODO l, m params can probably be omitted/set constant
        m_ptr, m_shape, m_stride, m_order, # |
        K_j_ptr,
        V_j_ptr,
        T_r,
        N,
        d,
        B_c,
        B_r,
        j
        ):

    # L: 6
    K_j = tl.load(K_j_ptr) # | TODO padding_option or boundary-check
    V_j = tl.load(V_j_ptr) # |

    # TODO staging (num_stages)
    # L: 7
    for i in range(T_r):

        # make *_i_ptrs for the blocks that are loaded in forward_inner
        Q_i_ptr = tl.make_block_ptr(
                Q_ptr,
                Q_shape,
                Q_stride,
                (i * B_r, 0),
                (B_r, d),
                Q_order)
        O_i_ptr = tl.make_block_ptr(
                O_ptr,
                O_shape,
                O_stride,
                (i * B_r, 0),
                (B_r, d),
                O_order)
        l_i_ptr = tl.make_block_ptr(
                l_ptr,
                l_shape,
                l_stride,
                (i * B_r, ),
                (B_r, ),
                l_order)
        m_i_ptr = tl.make_block_ptr(
                m_ptr,
                m_shape,
                m_stride,
                (i * B_r, ),
                (B_r, ),
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
    Q_i = tl.load(Q_i_ptr) # TODO padding_option or boundary-check
    O_i = tl.load(O_i_ptr)
    l_i = tl.load(l_i_ptr)
    m_i = tl.load(l_i_ptr)

    # L: 9
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


