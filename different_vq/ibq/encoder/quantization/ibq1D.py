import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce
import typing as tp


class IndexPropagationQuantize1D(nn.Module):
    """
    1D版本的IndexPropagationQuantize。
    输入维度: (batch, d, length)
    """
    def __init__(self, n_e, e_dim, beta=0.25, use_entropy_loss=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_entropy_loss = use_entropy_loss

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

    def forward(self, z, temp=None, return_logits=False, n_q: tp.Optional[int] = None):
        """
        z.shape = (B, d, L)
        """
        # 1) 计算相似度 logits: (b, n, l) = (b, d, l) x (n, d)
        logits = torch.einsum('bdl,nd->bnl', z, self.embedding.weight)  # (B, n, L)

        # 2) softmax
        soft_one_hot = F.softmax(logits, dim=1)  # (B, n, L)

        # 3) argmax 索引 -> one_hot
        ind = soft_one_hot.argmax(dim=1, keepdim=True)  # (B,1,L)
        hard_one_hot = torch.zeros_like(soft_one_hot).scatter_(1, ind, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        # 4) z_q = (B,d,L) = (B,n,L) x (n,d)
        z_q = torch.einsum('bnl,nd->bdl', one_hot, self.embedding.weight)
        z_q_hard = torch.einsum('bnl,nd->bdl', hard_one_hot, self.embedding.weight)

        # 5) 量化损失: MSE
        quant_loss = torch.mean((z_q - z)**2) \
                   + torch.mean((z_q_hard.detach() - z)**2) \
                   + self.beta * torch.mean((z_q_hard - z.detach())**2)

        diff = quant_loss
        if self.use_entropy_loss:
            # 如果需要熵正则，比如:
            # sample_entropy, avg_entropy, entropy_loss = compute_entropy_loss(...)
            # diff = (quant_loss, sample_entropy, avg_entropy, entropy_loss)
            pass

        # # 6) 将 ind 展平 (B, L) 或保持三维 (B,1,L)
        # ind = ind.squeeze
        diff = diff.unsqueeze(0).unsqueeze(0)

        return z_q, ind, diff
    def get_codebook_entry(self, indices: torch.Tensor, shape: tp.Optional[tp.Tuple[int, int, int]] = None) -> torch.Tensor:
        """
        indices: (B, L)
        shape: (B, d, L)
        """
        indices = indices.squeeze(1)
        one_hot = F.one_hot(indices, num_classes=self.n_e).permute(0, 2, 1).float()  # (B, n, L)
        z_q = torch.einsum('bnl,nd->bdl', one_hot, self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q
    
    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        return self.forward(x)[1]

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        return self.get_codebook_entry(q_indices, None)
