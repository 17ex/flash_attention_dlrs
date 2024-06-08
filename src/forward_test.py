from flash_attention import flash_attention_forward
import numpy as np
import torch
import triton
import triton.language as tl

N = 256
d = 16
NUM_TESTS = 100

# 3090, CHECK AGAIN, idk if this is correct,
# based on it having 128KB L1 per SM
SRAM = 128 * 1024

torch_attention = torch.nn.MultiheadAttention(N, d, bias=False)

for test in range(NUM_TESTS):
    torch.manual_seed(test)
    Q = torch.rand(d, N)
    K = torch.rand(d, N)
    V = torch.rand(d, N)
    O_torch = torch_attention(Q, K, V)
    O_flash, l_flash, m_flash = flash_attention_forward(
            Q,
            K,
            V,
            N,
            d,
            M=SRAM
            )
    print(O_torch)
