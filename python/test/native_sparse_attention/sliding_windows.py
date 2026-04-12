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

X = torch.randn(batch_size, seq_len, dim)

Wq = torch.randn(dim, dim)
Wk = torch.randn(dim, dim)
Wv = torch.randn(dim, dim)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# built sliding window attention
def get_window_mask(seq_len, window):
    mask = torch.ones(seq_len, seq_len, dtype = torch.long)
    mask = torch.tril(mask)
    win_mask = -torch.ones(seq_len - window, seq_len - window, dtype = torch.long)
    win_mask =  torch.tril(win_mask)
    mask[window:, :seq_len - window] += win_mask
    return mask
print(get_window_mask(7, 3)) # test

# begin to set windows mask
window_mask = get_window_mask(seq_len, 8)

window_mask = 1 - window_mask
print(f"after convert, window_mask: {window_mask}")

add_window_mask = window_mask * torch.tensor(float('-inf'))
add_window_mask = torch.nan_to_num(add_window_mask, nan=0.0)
print(f"final add_window_mask: {add_window_mask}")

# simplify multihead attention
S = Q @ K.transpose(1,2) / math.sqrt(dim)
S = S + add_window_mask # using add-style attention mask
print(f"after add window mask S: {S}")
S = F.softmax(S, dim = -1)
# print(f"after softmax S: {S}")
o_win = S @ V
print(o_win.shape) # torch.Size([1, 32, 128])

# final step: Gated Aggregation
W_gated = torch.randn(dim, 3) # 3: cmp, slc, win
gate = X @ W_gated
gate = F.sigmoid(gate) # value in [0, 1] after sigmoid
print(f"gate.shape: {gate.shape}") # torch.Size([1, 32, 3])

# mock o_cmp, o_slc
o_cmp = torch.zeros(batch_size, seq_len, dim)
o_slc = torch.zeros(batch_size, seq_len, dim)

o_list = [o_cmp, o_slc, o_win]
o_star = torch.zeros(batch_size, seq_len, dim)
for i in range(3):
    # print(f"gate[:, :, i].unsqueeze(2) shape: {gate[:, :, i].unsqueeze(2).shape}, o_list[i] shape: {o_list[i].shape}")
    # torch.Size([1, 32, 1]) * torch.Size([1, 32, 128])
    # weight of o_cmp, o_slc, o_win
    o_star += gate[:, :, i].unsqueeze(2) * o_list[i]
print(o_star.shape) # torch.Size([1, 32, 128])