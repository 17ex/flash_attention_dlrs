import torch
import triton
import flash_attention_kernels

MIN_TENSOR_SIZE=16

def convert_triton_dtype(torch_dtype):
    match torch_dtype:
        case torch.float64:
            return triton.language.float64
        case torch.float32:
            return triton.language.float32
        case torch.float16:
            return triton.language.float16
        case torch.float8_e5m2:
            return triton.language.float8e5
        case _:
            raise TypeError(f"dtype {torch_dtype} not supported.")


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        dev = Q.device
        if dev.type != 'cuda' or dev != K.device or dev != V.device:
            raise NotImplementedError("Q, K, V must be on the same CUDA device")

        if Q.dim() != 4 or Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError("Q, K, V must all be of shape (B, H, N, d)")

        if Q.dtype != K.dtype or K.dtype != V.dtype:
            raise ValueError(f"Q, K, V must have same dtype")

        B, H, N, d = Q.shape
        dtype = convert_triton_dtype(Q.dtype)

        # Support for non-power-of-2 d or d < 16
        d_proper = max(triton.next_power_of_2(d), MIN_TENSOR_SIZE)

        if d_proper != d:
            padded = True
            pad = (0, d_proper - d)
            Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
            K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
            V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)
        else:
            padded = False

        # Initialize output and statistics
        O = torch.empty_like(Q, requires_grad=(Q.requires_grad or K.requires_grad or V.requires_grad))
        L = torch.empty(B, H, N, 1, dtype=Q.dtype, device=dev, requires_grad=False)

        QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
        KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
        VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
        OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
        LB_stride, LH_stride, _, _ = L.stride()

        fwd_kernel_grid = lambda META: (triton.cdiv(N, META['B_r']), H, B)

        flash_attention_kernels.fwd_kernel[fwd_kernel_grid](
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
                B, H, N, d_proper,
                dtype
                )

        # Store stuff for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.padded = padded
        ctx.d_used = d_proper

        if padded:
            return O[:, :, :, 0:d]
        else:
            return O

    @staticmethod
    def backward(ctx, grad_outputs, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q, K, V, O, L = ctx.saved_tensors
        B, H, N, d = Q.shape
        dO = grad_outputs
        if Q.dtype != dO.dtype:
            raise ValueError(f"dO must have same dtype as inputs")

        dtype = convert_triton_dtype(Q.dtype)

        if ctx.padded:
            pad = (0, ctx.d_used - d)
            Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
            K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
            V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

        # Allocate output tensors
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        D = torch.empty_like(L)
        comm_ptr_len = N // 8 # Divide by 16 (smallest possible val of B_r), and we need double length
        lock_dQ = torch.zeros(B, H, comm_ptr_len, dtype=torch.int32, device=Q.device)
        written_dQ = torch.zeros(B, H, comm_ptr_len, dtype=torch.int32, device=Q.device)

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
        lock_dQ_B_stride, lock_dQ_H_stride, _ = lock_dQ.stride()
        written_dQ_B_stride, written_dQ_H_stride, _ = written_dQ.stride()

        # Precompute D
        bwd_D_grid = lambda META: (triton.cdiv(N, META['B_r']), H, B)
        flash_attention_kernels.bwd_D_kernel[bwd_D_grid](
                O, dO, D,
                OB_stride, OH_stride, ON_stride, Od_stride,
                dOB_stride, dOH_stride, dON_stride, dOd_stride,
                DB_stride, DH_stride,
                B, H, N, d,
                dtype
                )

        # Launch indeterministic bwd kernel
        bwd_kernel_grid = lambda META: (triton.cdiv(N, META['B_c']), H, B)
        flash_attention_kernels.bwd_kernel[bwd_kernel_grid](
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
                dtype
                )
        if ctx.padded:
            return dQ[:, :, :, 0:d], dK[:, :, :, 0:d], dV[:, :, :, 0:d]
        else:
            return dQ, dK, dV


class FlashAttentionDeterministic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        dev = Q.device
        if dev.type != 'cuda' or dev != K.device or dev != V.device:
            raise NotImplementedError("Q, K, V must be on the same CUDA device")

        if Q.dim() != 4 or Q.shape != K.shape or Q.shape != V.shape:
            raise ValueError("Q, K, V must all be of shape (B, H, N, d)")

        if Q.dtype != K.dtype or K.dtype != V.dtype:
            raise ValueError(f"Q, K, V must have same dtype")

        B, H, N, d = Q.shape
        dtype = convert_triton_dtype(Q.dtype)

        # Support for non-power-of-2 d or d < 16
        d_proper = max(triton.next_power_of_2(d), MIN_TENSOR_SIZE)

        if d_proper != d:
            padded = True
            pad = (0, d_proper - d)
            Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
            K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
            V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)
        else:
            padded = False

        # Initialize output and statistics
        O = torch.empty_like(Q, requires_grad=(Q.requires_grad or K.requires_grad or V.requires_grad))
        L = torch.empty(B, H, N, 1, dtype=Q.dtype, device=dev, requires_grad=False)

        QB_stride, QH_stride, QN_stride, Qd_stride = Q.stride()
        KB_stride, KH_stride, KN_stride, Kd_stride = K.stride()
        VB_stride, VH_stride, VN_stride, Vd_stride = V.stride()
        OB_stride, OH_stride, ON_stride, Od_stride = O.stride()
        LB_stride, LH_stride, _, _ = L.stride()

        fwd_kernel_grid = lambda META: (triton.cdiv(N, META['B_r']), H, B)

        flash_attention_kernels.fwd_kernel[fwd_kernel_grid](
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
                B, H, N, d_proper,
                dtype
                )

        # Store stuff for backward pass
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.padded = padded
        ctx.d_used = d_proper

        if padded:
            return O[:, :, :, 0:d]
        else:
            return O

    @staticmethod
    def backward(ctx, grad_outputs, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q, K, V, O, L = ctx.saved_tensors
        B, H, N, d = Q.shape
        dO = grad_outputs
        if Q.dtype != dO.dtype:
            raise ValueError(f"dO must have same dtype as inputs")

        dtype = convert_triton_dtype(Q.dtype)

        if ctx.padded:
            pad = (0, ctx.d_used - d)
            Q = torch.nn.functional.pad(Q, pad, mode='constant', value=0.0)
            K = torch.nn.functional.pad(K, pad, mode='constant', value=0.0)
            V = torch.nn.functional.pad(V, pad, mode='constant', value=0.0)

        # Allocate output tensors
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        D = torch.empty_like(L)
        written_dQ = torch.zeros(B, H, dtype=torch.int32, device=Q.device)

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
        written_dQ_B_stride, written_dQ_H_stride = written_dQ.stride()

        # Precompute D
        bwd_D_grid = lambda META: (triton.cdiv(N, META['B_r']), H, B)
        flash_attention_kernels.bwd_D_kernel[bwd_D_grid](
                O, dO, D,
                OB_stride, OH_stride, ON_stride, Od_stride,
                dOB_stride, dOH_stride, dON_stride, dOd_stride,
                DB_stride, DH_stride,
                B, H, N, d,
                dtype
                )

        # Launch deterministic bwd kernel
        bwd_kernel_grid = lambda META: (triton.cdiv(N, META['B_c']), H, B)
        flash_attention_kernels.bwd_deterministic_kernel[bwd_kernel_grid](
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
                dtype
                )
        if ctx.padded:
            return dQ[:, :, :, 0:d], dK[:, :, :, 0:d], dV[:, :, :, 0:d]
        else:
            return dQ, dK, dV
