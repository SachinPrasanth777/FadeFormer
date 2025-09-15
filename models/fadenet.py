import torch
import torch.nn as nn
import timm
from .attention import FadeAttn


class FadeNet(nn.Module):
    def __init__(self, num_classes, dropout=0.5, fade_attn_params=None):
        super().__init__()
        if fade_attn_params is None:
            fade_attn_params = {
                'num_heads': 8,
                'K': 3,
                'alpha': 0.5,
                'dropout': 0.1
            }
        
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.patch_embed = vit.patch_embed
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.blocks = vit.blocks
        self.norm = vit.norm
        
        self.attn = FadeAttn(768, **fade_attn_params)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = x[:, 1:]
        H = W = int(x.shape[1] ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, 768, H, W)
        x = self.attn(x)
        x = x.mean([-2, -1])
        x = self.dropout(x)
        out = self.classifier(x)
        return out