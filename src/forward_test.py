from flash_attention_v1 import flash_attention_forward
import torch

N = 256
d = 123
NUM_TESTS = 100

# NVIDIA GA102 GPUs
# They have 128KB L1 (SRAM) per SM
SRAM = 64 * 1024 # Something is wrong either here or later on. Hotfix
        # TODO Either this value is wrong, some calculation later on,
        # or I'm using more memory than I should.

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
    # Requires very large absolute tolerances. Why? Inexact exp maybe?
    if torch.allclose(O_torch, O_flash, atol=1e-4, rtol=1e-5):
        test_result[test] = 1
    else:
        print(O_torch)
        print(O_flash)

if torch.all(test_result):
    print(f"All {NUM_TESTS} test runs were completed successfully!")
else:
    print("Errors occurred!")
