import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)

# input sequence parameters
batch_size = 1
seq_len = 32
dim = 128
# compressed kv block length
block_len = 8
# suppose stripe = block_len
stripe = 8
# number of heads if apply mha
heads = 2
head_dim = dim // heads

# number of kv block to compress
# max_idx = round(( seq_len - block_len ) / stripe)
max_idx = round( seq_len / stripe)
block_nums = max_idx
print(f"max_idx: {max_idx}")
print(f"seq start compress idx: {torch.arange(max_idx) * stripe + 1}")
print(f"seq end compress idx: {torch.arange(max_idx) * stripe + block_len}")

X = torch.randn(batch_size, seq_len, dim)
print(f"X shape: {X.shape}")

Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)
Wv = torch.randn(dim, dim)

Q = X @ Wq
K = X @ Wk
V = X @ Wv
print(f"input: Q K V shape: {Q.shape}")

# block len compressed to len 1
W_K_cmp = torch.randn(block_len, 1)
W_V_cmp = torch.randn(block_len, 1)
# position embedding
W_pe = torch.randn(seq_len, dim)

# step1. KV compression
K_cmp = []
V_cmp = []
for i in range(max_idx):
    cur_K = K[:, i * stripe + 0: i * stripe + block_len , :] + W_pe[:block_len, :].unsqueeze(0)
    cur_V = V[:, i * stripe + 0: i * stripe + block_len , :]
    print(f"cur_K.transpose(1, 2) : {cur_K.transpose(1, 2).shape}, W_K_cmp: {W_K_cmp.shape}")
    # torch.Size([1, 128, 8]) @ torch.Size([8, 1]) -> torch.Size([1, 128, 1])
    cur_K = cur_K.transpose(1, 2) @ W_K_cmp
    cur_V = cur_V.transpose(1, 2) @ W_V_cmp
    K_cmp.append(cur_K)
    V_cmp.append(cur_V)

K_cmp = torch.cat(K_cmp, dim = 2).transpose(1,2)
V_cmp = torch.cat(V_cmp, dim = 2).transpose(1,2)
# torch.Size([1, 4, 128])
print(K_cmp.shape)
print(V_cmp.shape)

# muti head attention
Q = Q + W_pe[:seq_len, :].unsqueeze(0)
Q_mha = Q.view(batch_size, seq_len, heads, head_dim).transpose(1,2)
K_cmp_mha = K_cmp.view(batch_size, block_nums, heads, head_dim).transpose(1,2)
V_cmp_mha = V_cmp.view(batch_size, block_nums, heads, head_dim).transpose(1,2)
# Q_mha shape: torch.Size([1, 2, 32, 64]), K_cmp_mha shape: torch.Size([1, 2, 4, 64]), after transpose: torch.Size([1, 2, 64, 4])
print(f"Q_mha shape: {Q_mha.shape}, K_cmp_mha shape: {K_cmp_mha.shape}, after transpose: {K_cmp_mha.transpose(2,3).shape}")
# torch.Size([1, 2, 32, 64]) @ torch.Size([1, 2, 64, 4]) -> torch.Size([1, 4, 32, 4])
# last dimension from seq_len 32 reduce to compressed block nums 4
score_cmp = Q_mha @ K_cmp_mha.transpose(2,3) # batch_size, heads, seq_len, block_nums
# score_cmp.shape: torch.Size([1, 2, 32, 4])
print(f"score_cmp.shape: {score_cmp.shape}")

p_cmp = F.softmax(score_cmp, dim = -1)
# torch.Size([1, 2, 32, 4]) @ torch.Size([1, 2, 4, 64]) -> torch.Size([1, 2, 32, 64])
o_cmp = p_cmp @ V_cmp_mha
print(f"multi head o_cmp shape: {o_cmp.shape}")

o_cmp = o_cmp.transpose(2, 1).reshape(batch_size, seq_len, dim)
# torch.Size([1, 32, 128])
print(f"o_cmp.shape: {o_cmp.shape}")