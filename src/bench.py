import torch
import triton
from flash_attention_torch import FlashAttention, FlashAttentionDeterministic
from flash_attn import flash_attn_func
from flash_attention_openai_tutorial import _attention as openai_attention

B = 8
H = 8
d = 32

DTYPE = torch.float16
MODES = ["fwd", "bwd"]
# MODES = ["fwd"]
# MODES = ["bwd"]
torch.manual_seed(42)

# CUDA only
gpu = torch.device('cuda')

bench_configs: list[triton.testing.Benchmark] = []
for mode in MODES:
    dtype_str = str(DTYPE).split('.')[1]
    bench_configs.append(
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[2**i for i in range(7, 15)], # OpenAI Triton requires at least 2^7
            line_arg="provider",
            line_vals=[f"my-triton-det-{dtype_str}", f"my-triton-indet-{dtype_str}", f"daolab-{dtype_str}", f"openai-{dtype_str}",
                       f"torch-fa-{dtype_str}", f"torch-xformers-{dtype_str}", f"torch-math-{dtype_str}"],
            line_names=[f"My Triton ~FA-2 Det. [{dtype_str.upper()}]", f"My Triton ~FA-2 Indet. [{dtype_str.upper()}]",
                        f"DaoLab FA-2 [{dtype_str.upper()}]", f"OpenAI Triton FA-2 [{dtype_str.upper()}]",
                        f"Torch FA-2 [{dtype_str.upper()}]", f"Torch xFormers [{dtype_str.upper()}]", f"Torch Math [{dtype_str.upper()}]"],
            styles=[("red", "-"), ("red", "--"), ("blue", "-"), ("green", "-"),
                    ("black", "-"), ("black", "--"), ("black", "-.")],
            xlabel="N: Context size/Sequence length",
            ylabel="Mean Runtime [ms]",
            # x_log=False,
            # y_log=False,
            plot_name=f"fused-attention-B{B}-B{H}-d{d}-{mode}-{dtype_str}",
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
    try:
        Q = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
        K = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
        V = torch.randn(B, H, N, d, dtype=DTYPE, device=gpu, requires_grad=True)
        if "my-triton" in provider:
            if "my-triton-indet" in provider:
                def fwd_fn() -> torch.Tensor: return FlashAttention.apply(Q, K, V)
            else:
                def fwd_fn() -> torch.Tensor: return FlashAttentionDeterministic.apply(Q, K, V)
        elif "daolab" in provider:
            def fwd_fn() -> torch.Tensor: return flash_attn_func(Q, K, V)
        elif "torch" in provider:
            if "torch-fa" in provider:
                sdpbackend = torch.nn.attention.SDPBackend.FLASH_ATTENTION
            elif "torch-xformers" in provider:
                sdpbackend = torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
            elif "torch-math" in provider:
                sdpbackend = torch.nn.attention.SDPBackend.MATH
            else:
                raise ValueError()
            def fwd_fn() -> torch.Tensor:
                with torch.nn.attention.sdpa_kernel(sdpbackend):
                    return torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1)
        elif "openai" in provider:
            def fwd_fn() -> torch.Tensor: return openai_attention.apply(Q, K, V, False, 1.3)
        else:
            raise ValueError()

        if mode == "bwd":
            O: torch.Tensor = fwd_fn()
            dO = torch.randn_like(O)
            def bench_fn() -> torch.Tensor: return O.backward(dO, retain_graph=True)
        else:
            bench_fn = fwd_fn

        print(f"Benchmarking {mode} (N={N}, H={H}, B={B}, d={d}) for {provider} ...")
        ms = triton.testing.do_bench(bench_fn, warmup=warmup, rep=rep)
    except torch.cuda.OutOfMemoryError:
        ms = float('NaN')
    return ms


if __name__ == "__main__":
    bench_flash_attention.run(save_path="bench_out/", print_data=True)
