import triton


# ADAPT THIS TO YOUR DEVICE
# For NVIDIA GA102 GPUs
# They have 128KB L1 (SRAM) per SM.
# I set this to 99KB though, because based on
# a triton.runtime.errors.OutOfResources message,
# that seems to be the maximum I can use? TODO find out why.
SRAM = 99 * 1024

# The below calculations for SRAM requirements are super bad
# and only there as a heuristic to filter out configs that would
# result in out of resources errors if included.
# If there are still OutOfResources errors, then that means this
# estimation is garbage, in which case try to increase this factor,
# which directly applies to the estimated required SRAM.
SAFETY_MARGIN_MEM_FACTOR = 1

def is_cuda() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_autotune_config_cuda() -> list[triton.Config]:
    return [
            triton.Config({'B_r': 16, 'B_c': 16}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 16, 'B_c': 16}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 32}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 32}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 16, 'B_c': 32}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 16, 'B_c': 32}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 16}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 16}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 32}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 32}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=3, num_warps=4),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=4, num_warps=4),
            triton.Config({'B_r': 16, 'B_c': 64}, num_stages=4, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=3, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=4, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 16}, num_stages=4, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=3, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=4, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 64}, num_stages=4, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=3, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=4, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 32}, num_stages=4, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 64}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 64}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 32}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 32}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 32}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 32}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 32}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 128}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 128}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 128}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 32, 'B_c': 128}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 32, 'B_c': 128}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 64}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 64}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 64}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 64}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 64}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 128}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 128}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 128}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 128}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 128}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 64}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 64}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 64}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 64}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 64}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 128}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 128}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 128}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 128}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 128}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 64, 'B_c': 128}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 256}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 256}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 256}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 128, 'B_c': 256}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 128, 'B_c': 256}, num_stages=3, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 256}, num_stages=1, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 256}, num_stages=1, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 256}, num_stages=2, num_warps=4),
            triton.Config({'B_r': 256, 'B_c': 256}, num_stages=2, num_warps=8),
            triton.Config({'B_r': 256, 'B_c': 256}, num_stages=3, num_warps=8),
    ]


def fwd_num_stages_mem_factor(num_stages) -> float:
    # Very approximately (I just hand tried a few out)
    # determines how large the memory increase is for
    # a value of num_stages.
    # This is stupid, but I don't know how this would be calculated.
    match num_stages:
        case 1:
            return 1
        case 2:
            return 4/3
        case 3:
            return 2
        case 4:
            return 8/3
        case 5:
            return 640/39
        case 6:
            return 4
        case 7:
            return 896/99
        case 8:
            return 2**10/99
        case _:
            raise ValueError("For num_stages, only values between 1 and 8 are supported.")


def fwd_SRAM_needed(d, B_r, B_c) -> float:
    # This is only in theory. In practice, much more SRAM is used,
    # probably because the code is very non-optimal (eg. still FA-1 math atm),
    # and SRAM is also needed for lots of things other than the block matrices/vectors
    return 2 * (2 * B_r * d + 2 * B_c * d + 6 * B_r + 2*B_r * B_c)


def is_fwd_candidate(config: triton.Config, N, d) -> bool:
    B_r = config.kwargs['B_r']
    B_c = config.kwargs['B_c']
    num_stages = config.num_stages
    num_warps = config.num_warps
    SRAM_needed = fwd_SRAM_needed(d, B_r, B_c)
    SRAM_needed *= fwd_num_stages_mem_factor(num_stages)
    SRAM_needed *= SAFETY_MARGIN_MEM_FACTOR
    return N >= min(B_r, B_c) and SRAM_needed <= SRAM and N % max(B_r, B_c) == 0


def fwd_conf_prune(configs, *args, **kwargs) -> list[triton.Config]:
    kernel_args = args[0]
    B, H, N, d = kernel_args['B'], kernel_args['H'], kernel_args['N'], kernel_args['d']
    return list(filter(lambda config: is_fwd_candidate(config, N, d),
                       configs))


def get_autotune_config() -> list[triton.Config]:
    if is_cuda():
        return get_autotune_config_cuda()
    else:
        raise NotImplementedError("This flash attention implementation currently only supports CUDA devices!")


def bwd_SRAM_needed(N, d, B_r, B_c) -> float:
    # This is only in theory. In practice, much more SRAM is used,
    # probably because the code is very non-optimal (eg. still FA-1 math atm),
    # and SRAM is also needed for lots of things other than the block matrices/vectors
    return 2 * (N//4 + 4 * B_r * d + 4 * B_c * d + 2 * B_r + 3 * B_c * B_r)


def is_bwd_candidate(config: triton.Config, N, d) -> bool:
    B_r = config.kwargs['B_r']
    B_c = config.kwargs['B_c']
    # All configs where B_r or B_c is >= 64 already require too much SRAM
    return N >= min(B_r, B_c) and N % max(B_r, B_c) == 0 and bwd_SRAM_needed(N, d, B_r, B_c) <= SRAM


def bwd_conf_prune(configs, *args, **kwargs) -> list[triton.Config]:
    kernel_args = args[0]
    B, H, N, d = kernel_args['B'], kernel_args['H'], kernel_args['N'], kernel_args['d']
    return list(filter(lambda config: is_bwd_candidate(config, N, d),
                       configs))
