import torch
import torch.nn as nn
import torch.nn.functional as F


class FadeAttn(nn.Module):
    def __init__(self, in_channels, num_heads=8, K=3, alpha=0.5, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.K = K
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)
        self.laplacian_cache = {}
    
    def build_laplacian(self, H, W, device):
        key = (H, W, device)
        if key in self.laplacian_cache:
            return self.laplacian_cache[key]
        
        coords = torch.stack(torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).float().to(device)
        coords = coords.view(-1, 2)
        dist = torch.cdist(coords, coords, p=2)
        sigma = 1.0
        A = torch.exp(-dist**2 / (2 * sigma**2))
        D = torch.diag(A.sum(dim=1))
        L = D - A
        self.laplacian_cache[key] = L
        return L
    
    def fractional_diffusion(self, L, V):
        lambda_max = torch.linalg.eigvalsh(L).max()
        L_hat = (2 / lambda_max) * L - torch.eye(L.size(0), device=L.device)
        result = V.clone()
        for _ in range(self.K):
            result = torch.matmul(L_hat, result)
        return result
    
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        q = q.reshape(B, self.num_heads, self.head_dim, H*W).permute(0,1,3,2).reshape(
            B*self.num_heads, H*W, self.head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, H*W).permute(0,1,3,2).reshape(
            B*self.num_heads, H*W, self.head_dim)
        v = v.reshape(B, self.num_heads, self.head_dim, H*W).permute(0,1,3,2).reshape(
            B*self.num_heads, H*W, self.head_dim)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.reshape(B, self.num_heads, H*W, self.head_dim).transpose(1,2).reshape(
            B, H*W, C)
        
        L = self.build_laplacian(H, W, x.device)
        v_diffused = self.fractional_diffusion(L, v)
        v_diffused = v_diffused.reshape(B, self.num_heads, H*W, self.head_dim).transpose(1,2).reshape(
            B, H*W, C)
        
        out = self.alpha * attn_out + (1 - self.alpha) * v_diffused
        out = out.transpose(1, 2).reshape(B, C, H, W)
        out = self.proj(out)
        out = self.dropout(out)
        return out