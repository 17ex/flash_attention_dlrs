from flash_attention import flash_attention_forward, flash_attention_backward
from flash_attention_openai_tutorial import attention as openai_attention
import torch

N = 256
d = 128
NUM_TESTS = 100

# NVIDIA GA102 GPUs
# They have 128KB L1 (SRAM) per SM
SRAM = 64 * 1024 # Something is wrong either here or later on. Hotfix
        # TODO Either this value is wrong, some calculation later on,
        # or I'm using more memory than I should.
# Based on triton.runtime.errors.OutOfResources message,
# it seems this is actually just 99*1024 ?

gpu = torch.device('cuda')

test_result = torch.zeros(NUM_TESTS)

for test in range(NUM_TESTS):
    torch.manual_seed(test)
    Q = torch.randn(N, d, device=gpu)
    K = torch.randn(N, d, device=gpu)
    V = torch.randn(N, d, device=gpu)
    O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1)
    O_flash, L_flash = flash_attention_forward(
            Q,
            K,
            V,
            N,
            d,
            M=SRAM,
            dev=gpu
            )
    # O_openai = openai_attention(Q[None, None, :, :].to(dtype=torch.float16), K[None, None, :, :].to(dtype=torch.float16), V[None, None, :, :].to(dtype=torch.float16), False, 1.0)
    # Requires very large absolute tolerances. Why? Inexact exp maybe?
    if torch.allclose(O_torch, O_flash, atol=1e-4, rtol=1e-5):
        test_result[test] = 1
    else:
        print(O_torch)
        print(O_flash)
        # print(Q@K.T)
        # print(O_openai)

    dO = torch.randn_like(O_torch)

    flash_attention_backward(
            Q,
            K,
            V,
            O_flash,
            dO,
            L_flash,
            M=SRAM,
            dev=gpu
            )



if torch.all(test_result):
    print(f"All {NUM_TESTS} test runs were completed successfully!")
else:
    print("Errors occurred!")
