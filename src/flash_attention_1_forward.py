import numpy as np
import torch
import triton
import triton.language as tl

FP32_BYTESIZE = 4 # TODO future: accomodate other types than float32.

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

    # TODO
    # When writing a module, these can be changed into module properties
    # L: 1 (Determine block sizes)
    rows_bytesize = FP32_BYTESIZE * d # Assuming FP32
    B_c = min(cdiv(M, rows_bytesize), N) # TODO verify whether min here is okay. Should make sense though.
    B_r = min(B_c, d)
    T_r = cdiv(N, B_r)
    T_c = cdiv(N, B_c)

    # L: 2 (Initialize output and statistics)
    O = torch.zeros_like(Q, device=dev)
    l = torch.zeros(N, 1, device=dev)
    m = torch.full((N, 1), -np.inf, dtype=torch.float32, device=dev)

    print(f"B_c={B_c}, B_r={B_r}, T_r={T_r}, T_c={T_c}, N={N}, d={d}")

    forward_kernel[(T_c, )](
            Q,
            O,
            l,
            m,
            K,
            V,
            T_r,
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
        T_r: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_c: tl.constexpr,
        B_r: tl.constexpr
        ):

    j = tl.program_id(axis=0)

    K_j_ptr = tl.make_block_ptr(
            K_ptr,
            (N, d),
            (N, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)
    V_j_ptr = tl.make_block_ptr(
            V_ptr,
            (N, d),
            (N, 1),
            (j * B_c, 0),
            (B_c, d),
            ORDER)

    forward_outer(
            Q_ptr,
            O_ptr,
            l_ptr,
            m_ptr,
            K_j_ptr,
            V_j_ptr,
            T_r,
            N,
            d,
            B_r,
            B_c
            )


@triton.jit
def forward_outer(
        Q_ptr,
        O_ptr,
        l_ptr,
        m_ptr,
        K_j_ptr,
        V_j_ptr,
        T_r: tl.constexpr,
        N: tl.constexpr,
        d: tl.constexpr,
        B_r: tl.constexpr,
        B_c: tl.constexpr
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
                (N, 1),
                (i * B_r, 0),
                (B_r, d),
                ORDER)
        O_i_ptr = tl.make_block_ptr(
                O_ptr,
                (N, d),
                (N, 1),
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

        forward_inner(
                Q_i_ptr,
                O_i_ptr,
                l_i_ptr,
                m_i_ptr,
                tl.trans(K_j),
                V_j,
                B_r,
                B_c
                )



@triton.jit
def forward_inner(
        Q_i_ptr,
        O_i_ptr,
        l_i_ptr,
        m_i_ptr,
        K_jT,
        V_j,
        B_r: tl.constexpr,
        B_c: tl.constexpr
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
    mw_ij = tl.expand_dims(mw_ij, 1)
    Pw_ij = tl.exp(S_ij - tl.broadcast_to(mw_ij, (B_r, B_c)))
    lw_ij = tl.expand_dims(tl.sum(Pw_ij, axis=1), 1)

    # L: 11 (~ Calculate new softmax combining above with previous data)
    m_i_new = tl.maximum(m_i, mw_ij)
    #m_i_new = tl.maximum(m_i, tl.reshape(mw_ij, (B_r, )))
    prev_coeff = tl.exp(m_i - m_i_new)
    curr_coeff = tl.exp(mw_ij - m_i_new)
    l_i_new = prev_coeff * l_i + curr_coeff * lw_ij
    # TODO Determine if one should use fma here
    # Probably compiler does optimize anyways ?

    # L: 12 (Calculate new O_i)
    O_i_new = ((
            l_i * prev_coeff * O_i +
            curr_coeff * tl.dot(Pw_ij, V_j))
        / l_i_new)

    # L: 12-13 (Write to HBM)
    tl.store(O_i_ptr, O_i_new)
    tl.store(l_i_ptr, l_i_new)
    tl.store(m_i_ptr, m_i_new)
    return
