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

# number of kv blocks to compress
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




# step2. token selection
print("begin token selection")
# torch.Size([1, 2, 32, 4])
print(f"p_cmp.shape: {p_cmp.shape}")
# sum all head to get full attention: shape (1, 32, 4)
p_slc = p_cmp.sum(dim = 1)
print(f"sum all head, p_slc.shape: {p_slc.shape}")

# select top 2 from compressed p and return value idx
select_top_k = 2
value, idx = torch.topk(p_slc, dim = 2, k = select_top_k)
# idx shape: torch.Size([1, 32, 2]), pos 0 idx: tensor([3, 0])
print(f"idx shape: {idx.shape}, pos 0 idx: {idx[0,0,:]}")


# pos of selected token
# idx_slc_start: tensor([[[ 8, 24],
#          [ 8, 24],
#          [24,  0],
# ...
# idx_slc_end: tensor([[[16, 32],
#          [16, 32],
#          [32,  8],
idx_slc_start = idx * stripe
idx_slc_end = idx * stripe + block_len
K_slc = torch.randn(batch_size, seq_len, stripe * select_top_k, dim)
V_slc = torch.randn(batch_size, seq_len, stripe * select_top_k, dim)
# origin K_slc shape: torch.Size([1, 32, 16, 128])
print(f"origin K_slc shape: {K_slc.shape}")
# each token select 2 block_num, here len is 16
for i in range(batch_size):
    for j in range(seq_len):
        for k in range(select_top_k):
            K_slc[i, j, k * stripe : k * stripe + block_len, :] = K[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]
            V_slc[i, j, k * stripe : k * stripe + block_len, :] = V[i, idx_slc_start[i, j, k ] :  idx_slc_end[i, j, k ] , :]
# torch.Size([1, 32, 16, 128])
print(K_slc.shape)
print(V_slc.shape)

# shared head KV (refer MQA using 1-heads and GQA using n-heads)
# IN GQA Group: [1-head KV & N-head Q] ----repeat kv-head---> [N-head KV & N-head Q]

V_slc_mha = V_slc.view(batch_size, seq_len, select_top_k * stripe, heads, head_dim).transpose(2,3)
V_slc = V_slc_mha.sum(dim = 2, keepdim = True)
# torch.Size([1, 32, 1, 16, 64]) : heads sum to 1, mock to 1-head MQA
print(f"after sum all heads, V_slc.shape: {V_slc.shape}") # bs, seq_len, head, select_seq_len, head_dim

K_slc_mha = K_slc.view(batch_size, seq_len, select_top_k * stripe, heads, head_dim).transpose(2,3)
K_slc = K_slc_mha.sum(dim = 2, keepdim = True)
# torch.Size([1, 32, 1, 16, 64])
print(f"after sum all heads, K_slc.shape: {K_slc.shape}") # bs, seq_len, head, select_seq_len, head_dim

# debug Q-1 and KV attention
# Q_mha.shape: torch.Size([1, 2, 32, 64])
# K_slc means each seq in 32 select 16 (select_top_k * stripe) len in KV
print(f"Q_mha.shape: {Q_mha.shape}") # bs, heads, seq, head_dim
print(f"select No.5 Q, Q_mha[:, :, 5, :].shape: {Q_mha[:, :, 5, :].shape}") # seq_len=5 torch.Size([1, 2, 64])
print(f"K_slc[:, 5, :, :, :].shape: {K_slc[:, 5, :, :, :].shape}") # select seq_len=5 from torch.Size([1, 32, 1, 16, 64]) -> torch.Size([1, 1, 16, 64])

print(Q_mha[:, :, 5, :].unsqueeze(dim = 2).repeat(1, 1, select_top_k * stripe, 1).shape) # t=5 torch.Size([1, 2, 16, 64])
print(K_slc[:, 5, :, :, :].repeat(1, heads, 1, 1).shape) # t=5 torch.Size([1, 2, 16, 64])

# mock Q-5 and KV-16 attention
# repeat means MQA: each Q head shares the same KV head
Q_slc_j = Q_mha[:, :, 5, :].unsqueeze(dim = 2)
K_slc_j = K_slc[:, 5, :, :, :].repeat(1, heads, 1, 1)

# torch.Size([1, 2, 1, 64]) @ torch.Size([1, 2, 64, 16])
print(f"Q_slc_j shape: {Q_slc_j.shape}, K_slc_j.transpose(2,3) shape: {K_slc_j.transpose(2,3).shape}")
attn_score_j = Q_slc_j @ K_slc_j.transpose(2,3)
print(attn_score_j.shape) # bs, head, seq_q, seq_slc_k torch.Size([1, 2, 1, 16])

V_slc_j = V_slc[:, 5, :, :, :].repeat(1, heads, 1, 1) # torch.Size([1, 2, 16, 64])
print(f"V_slc_j.shape: {V_slc_j.shape}")

o_j = (attn_score_j @ V_slc_j).transpose(1,2).view(batch_size, 1, dim)
print(f"o_j.shape: {o_j.shape}") # bs, j, dim: torch.Size([1, 1, 128])

# begin token selection
o_slc = torch.zeros(batch_size, seq_len, dim)
for j in range(seq_len):
    Q_slc_j = Q_mha[:, :, j, :].unsqueeze(dim = 2)
    K_slc_j = K_slc[:, j, :, :, :].repeat(1, heads, 1, 1)
    V_slc_j = V_slc[:, j, :, :, :].repeat(1, heads, 1, 1)

    attn_score_j = Q_slc_j @ K_slc_j.transpose(2,3)
    p_slc_j = F.softmax(attn_score_j, dim = -1)
    # print(p_slc.shape)

    o_slc_j = p_slc_j @ V_slc_j # bs, seq, dim
    # print(o_slc_j.shape)

    o_slc_j = o_slc_j.transpose(1,2).view(batch_size, 1, dim)
    o_slc[:, j, :] = o_slc_j

print(o_slc.shape) # torch.Size([1, 32, 128])