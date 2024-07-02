import os
import torch
import triton
from flash_attention_torch import FlashAttention, FlashAttentionDeterministic
from flash_attn import flash_attn_func
from flash_attention_openai_tutorial import _attention as openai_attention

B = 8
H = 16
d = 128
N_MIN_log = 7  # OpenAI Triton requires at least 2^7
N_MAX_log = 15

BENCH_DIR = "bench_out"

# This can't be changed at the moment,
# some implementations don't work with fp8, and some don't work with fp32.
DTYPE = torch.float16

MODES = ["bwd"]
# MODES = ["fwd", "bwd"]

def run_bench(B, H, d, modes=MODES, N_min_log=N_MIN_log, N_max_log=N_MAX_log, dtype=torch.float16) -> None:
    if dtype != torch.float16:
        raise ValueError("Benchmark only supports torch.float16 dtype")
    torch.manual_seed(42)
    # CUDA only
    gpu = torch.device('cuda')
    os.makedirs(BENCH_DIR, exist_ok=True)
    bench_configs: list[triton.testing.Benchmark] = []
    for mode in modes:
        dtype_str = str(DTYPE).split('.')[1]
        bench_configs.append(
            triton.testing.Benchmark(
                x_names=["N"],
                x_vals=[2**i for i in range(N_min_log, N_max_log + 1)],
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
                plot_name=f"fused-attention-B{B}-H{H}-d{d}-{mode}-{dtype_str}",
                args={
                    "B": B,
                    "H": H,
                    "d": d,
                    "dtype": DTYPE,
                    "mode": mode
                },
            ))


    @triton.testing.perf_report(bench_configs)
    def bench_flash_attention(B, H, N, d, mode, dtype, provider):
        assert mode in modes
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
        # except triton.runtime.errorsOutOfMemoryError:
            # ms = float('NN')
        except triton.runtime.errors.OutOfResources:
            ms = float('NaN')
        except RuntimeError as e:
            print(f"{provider} raised the following error:")
            print(e)
            ms = float('NaN')
        return ms


    bench_flash_attention.run(save_path=BENCH_DIR + "/", print_data=True)

if __name__ == "__main__":
    run_bench(B, H, d)
