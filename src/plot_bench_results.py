import os
import re
import torch
import numpy as np
import pandas as pd
from bench import run_bench
import matplotlib.pyplot as plt
from tueplots.constants import markers
from tueplots.constants.color import palettes
from tueplots import axes, bundles, figsizes, cycler

BENCH_DIR = "bench_out"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

B = 8
H = 8
d = 32
mode = "fwd"
dtype = torch.float16

y_log = True

N_MIN = 2**9
N_MAX = 2**16

run_bench_if_not_exists = True
bench_overwrite_if_exists = False

include_det = True
include_indet = True

include_det = include_det and mode != "fwd"

plt.rcParams.update({"figure.dpi": 150})
plt.rcParams.update(figsizes.beamer_169())
plt.rcParams.update(axes.lines())


def get_bench_filename(B, H, d, mode, dtype) -> str:
    dtype_str = str(dtype).split('.')[1]
    return f"{BENCH_DIR}/fused-attention-B{B}-H{H}-d{d}-{mode}-{dtype_str}.csv"


def get_plot_filename(B, H, d, mode, dtype) -> str:
    dtype_str = str(dtype).split('.')[1]
    det_str = "-withdet" if include_det else ""
    log_str = "-ylog" if y_log else ""
    return f"{PLOT_DIR}/plot-fused-attention-B{B}-H{H}-d{d}-{mode}-{dtype_str}{det_str}{log_str}.pdf"


def read_bench_csv(B, H, d, mode, dtype) -> pd.DataFrame:
    return pd.read_csv(
            get_bench_filename(B, H, d, mode, dtype),
            sep=",",
            )


def get_bench_data(B, H, d, mode, dtype) -> pd.DataFrame:
    ran_bench = False
    try:
        if bench_overwrite_if_exists:
            # Run bench regardless whether it exists or not
            ran_bench = True
            run_bench(B, H, d, [mode, ], dtype=dtype)
        df = read_bench_csv(B, H, d, mode, dtype)
    except FileNotFoundError as e:
        if not ran_bench and run_bench_if_not_exists:
            run_bench(B, H, d, [mode, ], dtype=dtype)
            df = read_bench_csv(B, H, d, mode, dtype)
        else:
            raise e
    df['N'] = df['N'].astype(np.int32)
    df = df[df['N'] >= N_MIN]
    df = df[df['N'] <= N_MAX]
    return df


def plot_title(B, H, d, mode, dtype) -> str:
    mode_str = "Forward Pass" if mode == "fwd" else "Backward Pass"
    match dtype:
        case torch.float8_e5m2:
            dtype_str = "FP32"
        case torch.float16:
            dtype_str = "FP16"
        case torch.float32:
            dtype_str = "FP32"
        case _:
            raise ValueError(f"This dtype is not supported: {dtype}")
    return f"Attention {mode_str} (B={B}, H={H}, d={d}), {dtype_str}"


def clean_provider_name(provider):
    return re.sub(r"\s*\[.*\]|\~", r"", provider)


def why_is_this_not_an_option_in_set_xyscale_or_similar(x, pos):
    return str(int(x))


def plot_bench_data(B, H, d, mode, dtype):
    bench_data = get_bench_data(B, H, d, mode, dtype)
    with plt.rc_context(bundles.beamer_moml()):
        fig, ax = plt.subplots()
        providers = list(bench_data.keys())
        providers.reverse()
        for provider in providers:
            if provider == 'N':
                continue
            if "My" in provider:
                if (not include_det and " Det" in provider or
                    not include_indet and " Indet" in provider):
                    continue
            ax.plot(bench_data['N'], bench_data[provider], label=clean_provider_name(provider), linewidth=2)
        ax.set_title(plot_title(B, H, d, mode, dtype))
        ax.set_xlabel("Context size (N)")
        ax.set_xticks(bench_data['N'], labels=bench_data['N'])
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(why_is_this_not_an_option_in_set_xyscale_or_similar)
        if y_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(why_is_this_not_an_option_in_set_xyscale_or_similar)
        ax.set_ylabel("mean execution time [ms]")
        plt.legend()
        plt.savefig(get_plot_filename(B, H, d, mode, dtype))


if __name__ == "__main__":
    plot_bench_data(B, H, d, mode, dtype)
