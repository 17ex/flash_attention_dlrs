import triton
import triton.language as tl
import autotune_configs

DOT_PRECISION: tl.constexpr = "ieee"
ORDER: tl.constexpr = (0, 1)

@triton.autotune(
        configs=autotune_configs.get_autotune_config(),
        key=['B', 'H', 'N', 'd'],
        prune_configs_by={"early_config_prune": autotune_configs.fwd_conf_prune}
)
@triton.jit
def fwd_kernel(
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
        B, H, N: tl.constexpr, d: tl.constexpr, # B, H are here to re-tune the kernel when they change
        dtype: tl.constexpr,
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
    O_i= tl.zeros((B_r, d), dtype=tl.float32)
    m_i= tl.full((B_r, 1), float('-inf'), dtype=tl.float32)
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
               + tl.exp(m_ij - m_i_new) * tl.dot(P_ij.to(V_ptr.type.element_ty), V_j, input_precision=DOT_PRECISION)) \
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
    tl.store(O_i_ptr, O_i.to(O_ptr.type.element_ty)) # Why is this not in the documentation? :(
    tl.store(L_i_ptr, L_i.to(L_ptr.type.element_ty))
    return




@triton.autotune(
        configs=autotune_configs.get_autotune_config(),
        key=['B', 'H', 'N', 'd'],
        prune_configs_by={"early_config_prune": autotune_configs.bwd_D_conf_prune}
)
@triton.jit
def bwd_D_kernel(
    O_ptr, dO_ptr, D_ptr,
    OB_stride, OH_stride, ON_stride, Od_stride,
    dOB_stride, dOH_stride, dON_stride, dOd_stride,
    DB_stride, DH_stride,
    B, H, N: tl.constexpr, d: tl.constexpr, # B, H are here to re-tune the kernel when they change
    dtype: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr # Unused, present because configs specify it.
        ) -> None:
    # This kernel simply computes rowsum(dO * O), which is it's own seperate kernel,
    # computed before the rest, in FA-2.

    # TODO why does it say rowsum but R^d in the FA-2 paper??
    # Is that an error or am I misunderstanding something?
    # -> R^n makes more sense to me, so I'll use that.

    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)

    O_i_ptr = tl.make_block_ptr(
            O_ptr + b * OB_stride + h * OH_stride,
            (N, d),
            (ON_stride, Od_stride),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    dO_i_ptr = tl.make_block_ptr(
            dO_ptr + b * dOB_stride + h * dOH_stride,
            (N, d),
            (dON_stride, dOd_stride),
            (i * B_r, 0),
            (B_r, d),
            ORDER)
    D_i_ptr = tl.make_block_ptr(
            D_ptr + b * DB_stride + h * DH_stride,
            (N, 1),
            (1, 1),
            (i * B_r, 0),
            (B_r, 1),
            ORDER)

    O_i = tl.load(O_i_ptr)
    dO_i = tl.load(dO_i_ptr)
    D_i = tl.sum(O_i * dO_i, axis=1, keep_dims=True)
    tl.store(D_i_ptr, D_i.to(D_ptr.type.element_ty))


@triton.autotune(
        configs=autotune_configs.get_autotune_config(),
        key=['B', 'H', 'N', 'd'],
        prune_configs_by={"early_config_prune": autotune_configs.bwd_conf_prune}
)
@triton.jit
def bwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        dQ_ptr, dK_ptr, dV_ptr, dO_ptr,
        L_ptr, D_ptr,
        lock_dQ, written_dQ,
        QB_stride, QH_stride, QN_stride, Qd_stride,
        KB_stride, KH_stride, KN_stride, Kd_stride,
        VB_stride, VH_stride, VN_stride, Vd_stride,
        dQB_stride, dQH_stride, dQN_stride, dQd_stride,
        dKB_stride, dKH_stride, dKN_stride, dKd_stride,
        dVB_stride, dVH_stride, dVN_stride, dVd_stride,
        dOB_stride, dOH_stride, dON_stride, dOd_stride,
        LB_stride, LH_stride, DB_stride, DH_stride,
        lock_dQ_B_stride, lock_dQ_H_stride,
        written_dQ_B_stride, written_dQ_H_stride,
        B, H, N, d: tl.constexpr, # B, H are here to re-tune the kernel when they change
        dtype: tl.constexpr,
        B_c: tl.constexpr, # TODO when N is constexpr, it for whatever reason hangs upon execution
        B_r: tl.constexpr
        ) -> None:

    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    j = tl.program_id(axis=2)

    T_r = tl.cdiv(N, B_r)

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
    D_i_ptr = tl.make_block_ptr(
            D_ptr + b * DB_stride + h * DH_stride,
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
        Q_i = tl.load(Q_i_ptr)   # O_i doesn't need to be loaded,
        dO_i = tl.load(dO_i_ptr) # even though the paper says to load it
        L_i = tl.load(L_i_ptr)
        D_i = tl.load(D_i_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j), input_precision=DOT_PRECISION)

        P_ij = tl.exp(S_ij - L_i)

        dV_j = dV_j + tl.dot(tl.trans(P_ij).to(V_ptr.type.element_ty),dO_i, input_precision=DOT_PRECISION, out_dtype=dV_ptr.type.element_ty)

        dP_ij = tl.dot(dO_i, tl.trans(V_j), input_precision=DOT_PRECISION)

        dS_ij = P_ij * (dP_ij - D_i.to(tl.float32))

        dK_j = dK_j + tl.dot(tl.trans(dS_ij).to(dK_ptr.type.element_ty), Q_i, input_precision=DOT_PRECISION, out_dtype=dK_ptr.type.element_ty)

        # Writes to dQ_i need to be guarded, since multiple threads can try to do this at the
        # same time, leading to race conditions.
        # The below code guardes against this using a locking strategy to ensure
        # only one thread can update dQ_i at a time.
        # TODO
        # Important: This is non-deterministic. Add a seperate, deterministic bwd pass kernel,
        # eg. by not really using a lock, but something where every thread writes their j into the
        # communication data structure, so the order is always the same
        # TODO
        # Very Important: For some reason, the first time the bwd kernel is run,
        # the non-deterministic dQ that is returned is completely wrong.
        # After that, no matter how often it's run, the non-deterministic
        # dQ is always within very small tolerances.
        # Why is that the case? This needs to be fixed
        while tl.atomic_cas(lock_dQ_i, 0, 1) == 1:
            pass
        # ============== dQ_i (and written_dQ_i) Mem access is protected by a lock in this snippet.
        count = tl.load(written_dQ_i)
        if count == 0:
            # dQ_i was not written before, so it is uninitialized
            dQ_i = tl.zeros_like(Q_i)
            tl.atomic_xchg(written_dQ_i, 1)
        else:
            # dQ_i was written to before
            dQ_i = tl.load(dQ_i_ptr)
        dQ_i = dQ_i + tl.dot(dS_ij.to(K_ptr.type.element_ty), K_j, input_precision=DOT_PRECISION, out_dtype=dQ_ptr.type.element_ty)
        tl.store(dQ_i_ptr, dQ_i)
        # ==============
        # Release the lock again
        tl.atomic_xchg(lock_dQ_i, 0)

        # Advance the *_i_ptrs
        Q_i_ptr = tl.advance(Q_i_ptr, (B_r, 0))
        dO_i_ptr = tl.advance(dO_i_ptr, (B_r, 0))
        dQ_i_ptr = tl.advance(dQ_i_ptr, (B_r, 0))
        L_i_ptr = tl.advance(L_i_ptr, (B_r, 0))
        D_i_ptr = tl.advance(D_i_ptr, (B_r, 0))
        lock_dQ_i += 1
        written_dQ_i += 1


    tl.store(dK_j_ptr, dK_j.to(K_j_ptr.type.element_ty))
    tl.store(dV_j_ptr, dV_j)
    return
