import torch
from einops import rearrange

A = torch.randn((192,192,3,3))
# B = rearrange(A, 'c1 c2 h w-> (c1 h) (c2 w)')
B = rearrange(A, 'c1 c2 h w-> c1 (c2 h w)')
U, S, Vh = torch.linalg.svd(B, full_matrices=False)
print(B.shape, U.shape, S.shape, Vh.shape)
out = U@torch.diag(S)@Vh
print(out.shape)
print(torch.dist(B, out))