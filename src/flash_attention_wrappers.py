import torch
import triton
from flash_attention_kernels import fwd_kernel, bwd_D_kernel, bwd_kernel, bwd_deterministic_kernel
from flash_attention_torch import convert_triton_dtype


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
    assert Q.dtype == K.dtype and K.dtype == V.dtype

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
    O = torch.empty(B, H, N, d_pow, dtype=Q.dtype, device=dev)
    L = torch.empty(B, H, N, 1, dtype=Q.dtype, device=dev)

    QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
    KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
    VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
    OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
    LB_stride, LH_stride, _, _ = L.stride()

    fwd_kernel_grid = lambda META: (B, H, triton.cdiv(N, META['B_r']))

    fwd_kernel[fwd_kernel_grid](
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
            B, H, N, d_pow,
            convert_triton_dtype(Q.dtype)
            )

    return O[:, :, :, 0:d], L


def flash_attention_backward(
        Q,
        K,
        V,
        O,
        dO,
        L,
        dev,
        deterministic=False
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    assert Q.dim() == 4
    assert Q.shape == K.shape and V.shape == K.shape
    assert Q.dtype == K.dtype and K.dtype == V.dtype

    B, H, N, d = Q.shape

    d_pow = triton.next_power_of_2(d)

    if d_pow != d:
        # Apply padding
        pad = (0, d_pow - d)
        Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
        K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
        V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

    # Allocate output tensors
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    D = torch.empty_like(L)

    QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
    KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
    VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
    OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
    dQB_stride, dQH_stride, dQN_stride, dQd_stride = dQ.stride()
    dKB_stride, dKH_stride, dKN_stride, dKd_stride = dK.stride()
    dVB_stride, dVH_stride, dVN_stride, dVd_stride = dV.stride()
    dOB_stride, dOH_stride, dON_stride, dOd_stride = dO.stride()
    LB_stride, LH_stride, _, _ = L.stride()
    DB_stride, DH_stride, _, _ = D.stride()

    # Precompute D
    bwd_D_grid = lambda META: (B, H, triton.cdiv(N, META['B_r']))
    bwd_D_kernel[bwd_D_grid](
            O, dO, D,
            OB_stride, OH_stride, ON_stride, Od_stride,
            dOB_stride, dOH_stride, dON_stride, dOd_stride,
            DB_stride, DH_stride,
            B, H, N, d,
            convert_triton_dtype(Q.dtype)
            )

    bwd_kernel_grid = lambda META: (B, H, triton.cdiv(N, META['B_c']))

    if deterministic:
        written_dQ = torch.empty(B, H, dtype=torch.int32, device=dev)
        written_dQ_B_stride, written_dQ_H_stride = written_dQ.stride()

        bwd_deterministic_kernel[bwd_kernel_grid](
                Q, K, V,
                dQ, dK, dV, dO,
                L, D,
                written_dQ,
                QB_stride, QH_stride, QN_stride, Qd_stride,
                KB_stride, KH_stride, KN_stride, Kd_stride,
                VB_stride, VH_stride, VN_stride, Vd_stride,
                dQB_stride, dQH_stride, dQN_stride, dQd_stride,
                dKB_stride, dKH_stride, dKN_stride, dKd_stride,
                dVB_stride, dVH_stride, dVN_stride, dVd_stride,
                dOB_stride, dOH_stride, dON_stride, dOd_stride,
                LB_stride, LH_stride, DB_stride, DH_stride,
                written_dQ_B_stride, written_dQ_H_stride,
                B, H, N, d,
                convert_triton_dtype(Q.dtype)
                )
    else:
        # int64 would make more sense here,
        # because atomic ops pointers only work on aligned memory,
        # but I need to compare with constexprs, and then the compiler
        # complains because it can't compare int64 with int32,
        # and I don't know how to specify that a literal should be an
        # int64 and not int32. I don't even know if you can do this easily.
        # So, instead we double the length, and only index every 2nd element.
        comm_ptr_len = N // 8 # Because 16 is smallest possible val of B_r, and we need double length
        lock_dQ = torch.zeros(B, H, comm_ptr_len, dtype=torch.int32, device=dev)
        written_dQ = torch.zeros(B, H, comm_ptr_len, dtype=torch.int32, device=dev)
        written_dQ_B_stride, written_dQ_H_stride, _ = written_dQ.stride()
        lock_dQ_B_stride, lock_dQ_H_stride, _ = lock_dQ.stride()

        bwd_kernel[bwd_kernel_grid](
                Q, K, V,
                dQ, dK, dV, dO,
                L, D,
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
                B, H, N, d,
                convert_triton_dtype(Q.dtype)
                )

    return dQ, dK, dV
