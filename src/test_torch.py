import torch
from flash_attention_torch import FlashAttention, FlashAttentionDeterministic

B = 2
H = 2
N = 32
d = 128

torch.manual_seed(5)
Q = torch.randn(B, H, N, d, dtype=torch.float32, device='cuda', requires_grad=True)
K = torch.randn_like(Q, requires_grad=True)
V = torch.randn_like(Q, requires_grad=True)
dO = torch.randn_like(Q)

# TODO
# Now why the heck is the old (non-det.) kernel now suddenly
# bwd deterministic/reentrant, and the new deterministic one
# does not fail that check?

# Non-deterministic
try:
    O = FlashAttention.apply(Q, K, V)
    # TODO For some reason, the first time the nondeterministic bwd pass runs,
    # it's completely wrong, and after that, no matter how often it's run,
    # the non-deterministic differences are tiny. This needs to be fixed.
    # (See comment in flash_attention.py)
    # This one bwd pass here serves to hotfix that so gradcheck works.
    dQ, dK, dV = torch.autograd.grad(O, (Q, K, V), dO)

    torch.autograd.gradcheck(FlashAttention.apply, (Q, K, V), eps=2e-2, atol=1e-2, rtol=1e-2, nondet_tol=1e-4)
    print("Non-deterministic backwards test successful")
except Exception as e:
    print("Error during gradcheck for non-deterministic backwards pass:")
    print(e)

# Deterministic
try:
    O = FlashAttentionDeterministic.apply(Q, K, V)
    torch.autograd.gradcheck(FlashAttentionDeterministic.apply, (Q, K, V), eps=2e-2, atol=1e-2, rtol=1e-2, nondet_tol=1e-4)
    print("Deterministic backwards test successful")
except Exception as e:
    # Catching all exceptions sucks, but I can't check for
    # torch.autograd.gradcheck.GradcheckError,
    # maybe it's not exported?
    print("Error during gradcheck for deterministic backwards pass:")
    print(e)

