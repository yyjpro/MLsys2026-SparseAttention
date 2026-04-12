import math
import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class DSAConfig:
    dim: int = 512
    n_heads: int = 8
    q_lora_rank: int = 16
    kv_lora_rank: int = 16
    qk_nope_head_dim: int = 16
    qk_rope_head_dim: int = 16
    v_head_dim: int = 16
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    vocab_size: int = 100

class Indexer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim: int = config.dim
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = nn.Linear(self.dim, self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, dtype=torch.get_default_dtype())

        # self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        # self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)


    def forward(self, x: torch.Tensor, qr: torch.Tensor):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(qr)
        q = q.reshape(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)

        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # apply rope
        q = torch.cat([q_pe, q_nope], dim=-1)


        k = self.wk(x)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # apply rope
        k = torch.cat([k_pe, k_nope], dim=-1)

        weights = self.weights_proj(x) * self.n_heads ** -0.5 # bsz, head

        # formula 1
        s = q @ k.transpose(2,3) / math.sqrt(self.head_dim)
        index_score = torch.sum(s * weights.reshpa(1,self.head_dim, 1, 1), dim = 1)

        if mask is not None:
            index_score += mask

        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
        topk_indices_ = topk_indices.clone()
        return topk_indices

class MLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        self.indexer = Indexer(config)

    def forward(self, x, mask):
        bsz, seq_len, _ = x.shape

        q_a, kv_a = self.wq_a(x), self.wkv_a(x)
        c_a, k_pe = torch.split(kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        q, kv = self.wq_b(q_a), self.wkv_b(c_a)
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        scores = q @ k.reshape(2,3)

        indexer_mask = self.indexer(x, qr)
        scores = scores + (mask + indexer_mask).unsqueeze(dim = 1)

        p = F.softmax(scores, dim = -1)
        z = p @ v
        z = z.reshape(bsz, seq_len, n_heads * self.v_head_dim)
        z = self.wo(z)

        return z