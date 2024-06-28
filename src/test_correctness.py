from flash_attention import flash_attention_forward, flash_attention_backward
from flash_attention_openai_tutorial import attention as openai_attention
import torch

# This file is a script that should test the Flash Attention implementation
# by comparing the outputs and gradients to those of torch's attention implementation.
# Running this takes ~30s on my system.
# Reduce numbers below to reduce runtime.

B = 32
H = 32
N = 256
d = 128
NUM_TESTS = 200

# For NVIDIA GA102 GPUs
# They have 128KB L1 (SRAM) per SM
SRAM = 64 * 1024 # Something is wrong either here or later on. Hotfix
        # TODO Either this value is wrong, some calculation later on,
        # or I'm using more memory than I should.
# Based on triton.runtime.errors.OutOfResources message,
# it seems this is actually just 99*1024 ?
# So maybe 64KB is not that wrong after all?

# Use CUDA. I don't have a non-cuda (eg. ROCm) GPU on which to test this,
# so I have no clue if my implementation supports non-CUDA targets.
gpu = torch.device('cuda')

test_result_fwd = torch.zeros(NUM_TESTS, dtype=torch.int32)
test_result_bwd_Q = torch.zeros(NUM_TESTS, dtype=torch.int32)
test_result_bwd_K = torch.zeros(NUM_TESTS, dtype=torch.int32)
test_result_bwd_V = torch.zeros(NUM_TESTS, dtype=torch.int32)

for test in range(NUM_TESTS):
    torch.manual_seed(test)
    Q = torch.randn(B, H, N, d, device=gpu, requires_grad=True)
    K = torch.randn(B, H, N, d, device=gpu, requires_grad=True)
    V = torch.randn(B, H, N, d, device=gpu, requires_grad=True)
    O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1)
    O_flash, L_flash = flash_attention_forward(
            Q,
            K,
            V,
            M=SRAM,
            dev=gpu
            )
    if torch.allclose(O_torch, O_flash, atol=1e-4, rtol=1e-5):
        test_result_fwd[test] = 1
    else:
        print(O_torch)
        print(O_flash)

    dO = torch.randn_like(O_torch)

    dQ_torch, dK_torch, dV_torch = torch.autograd.grad(O_torch, (Q, K, V), dO)

    dQ_flash, dK_flash, dV_flash = flash_attention_backward(
            Q,
            K,
            V,
            O_flash,
            dO,
            L_flash,
            M=SRAM,
            dev=gpu
            )

    # Somewhere around this ballpark are the tolerances.
    # Some tests begin to fail for a little tighter (absolute) tolerances.
    test_result_bwd_Q[test] = torch.allclose(dQ_torch, dQ_flash, atol=9e-4, rtol=1e-5)
    test_result_bwd_K[test] = torch.allclose(dK_torch, dK_flash, atol=7e-4, rtol=1e-5)
    test_result_bwd_V[test] = torch.allclose(dV_torch, dV_flash, atol=7e-5, rtol=1e-5)

    # TODO
    # Sometimes (rarely), some tests seem to fail nondeterministically.
    # This means that either in my implementation or the torch one,
    # the outputs can differ non-deterministically by a very small amount.
    # I don't know why, how, or what this means, but this being non-deterministic
    # seems bad.
    # Find out if this happens in torch's attention implementation or in my flash attention impl.

print(f"{test_result_fwd.sum().item()} out of {NUM_TESTS} forward tests succeeded!")
print(f"{test_result_bwd_Q.sum().item()} out of {NUM_TESTS} Q backward tests succeeded!")
print(f"{test_result_bwd_K.sum().item()} out of {NUM_TESTS} K backward tests succeeded!")
print(f"{test_result_bwd_V.sum().item()} out of {NUM_TESTS} V backward tests succeeded!")
