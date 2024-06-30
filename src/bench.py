import torch
import triton
from flash_attention_torch import FlashAttention
from flash_attn import flash_attn_func

B = 8
H = 8
d = 128

DTYPE = torch.float32
MODES = ["fwd", "bwd"]
torch.manual_seed(42)

# CUDA only
gpu = torch.device('cuda')

bench_configs: list[triton.testing.Benchmark] = []
for mode in MODES:
    dtype_str = str(DTYPE).split('.')[1]
    bench_configs.append(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(5, 12)],
            line_arg="provider",
            line_vals=[f"my-triton-{dtype_str}", f"daolab-{dtype_str}", f"torch-{dtype_str}"],
            line_names=[f"My Triton [{dtype_str.upper()}]", f"DaoLab FA-2 [{dtype_str.upper()}]", f"Torch Attention [{dtype_str.upper()}]"],
            styles=[("red", "-"), ("blue", "-"), ("black", "-")],
            xlabel="N: Context size/Sequence length",
            ylabel="Mean Runtime [ms]",
            # x_log=False,
            # y_log=False,
            plot_name=f"fused-attention-batch{B}-head{H}-d{d}-{mode}-{dtype_str}",
            args={
                "B": B,
                "H": H,
                "d": d,
                "dtype": DTYPE,
                "mode": mode
            },
        ))


# TODO ensure that all kernels are run once before benching
@triton.testing.perf_report(bench_configs)
def bench_flash_attention(B, H, N, d, mode, dtype, provider):
    assert mode in MODES
    warmup = 25
    rep = 100
    Q = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
    K = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
    V = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
    if "triton" in provider:
        def fwd_fn() -> torch.Tensor: return FlashAttention.apply(Q, K, V)
    elif "daolab" in provider:
        Q = Q.to(dtype=torch.float16)
        K = K.to(dtype=torch.float16)
        V = V.to(dtype=torch.float16)
        def fwd_fn() -> torch.Tensor: return flash_attn_func(Q, K, V)
        # TODO
        # HOTFIX
    elif "torch" in provider:
        def fwd_fn() -> torch.Tensor: return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1)
    else:
        raise ValueError()

    if mode == "bwd":
        O: torch.Tensor = fwd_fn()
        dO = torch.randn_like(O)
        def bench_fn() -> torch.Tensor: return O.backward(dO, retain_graph=True)
    else:
        bench_fn = fwd_fn
    # TODO test
    bench_fn()
    ms = triton.testing.do_bench(bench_fn, warmup=warmup, rep=rep)
    return ms

    # flops_per_matmul = 2.0 * B * H * N * N * d
    # total_flops = 2 * flops_per_matmul
    # if mode == "bwd":
        # total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    # return total_flops / ms * 1e-9


if __name__ == "__main__":
    bench_flash_attention.run(save_path="bench_out/", print_data=True)
