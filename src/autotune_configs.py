import triton


def is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_fwd_autotune_config_cuda() -> list[triton.Config]:
    return [
        triton.Config({'B_r': 128, 'B_c': 64}, num_stages=2, num_warps=32),
        triton.Config({'B_r': 64, 'B_c': 64}, num_stages=2, num_warps=16),
        triton.Config({'B_r': 64, 'B_c': 64}, num_stages=2, num_warps=8),
        triton.Config({'B_r': 64, 'B_c': 32}, num_stages=2, num_warps=4),
        triton.Config({'B_r': 64, 'B_c': 64}, num_stages=2, num_warps=2),
        # triton.Config({'B_r': 32, 'B_c': 64}, num_stages=2, num_warps=4),
        # triton.Config({'B_r': , 'B_c': 64}, num_stages=2, num_warps=4),
    ]


def fwd_conf_prune(configs, *args, **kwargs) -> list[triton.Config]:
    kernel_args = args[0]
    B, H, N, d = kernel_args['B'], kernel_args['H'], kernel_args['N'], kernel_args['d']
    return configs


def get_fwd_autotune_config() -> list[triton.Config]:
    if is_cuda():
        return get_fwd_autotune_config_cuda()
    else:
        raise NotImplementedError("This flash attention implementation currently only supports CUDA devices!")
