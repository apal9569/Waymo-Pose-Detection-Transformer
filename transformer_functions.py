import torch
import torch.nn as nn
import torch.optim as optim
import math
def positional_encoding_function(maxL, D):
    pe = torch.zeros(maxL, D)
        # pe measures (maxL, D)
    position = torch.arange(0, maxL).unsqueeze(1)
        # (maxL) --> (maxL, 1) via unsqueeze(1)
    coeff = torch.exp(torch.arange(0, D, 2).float() * -(math.log(10000.0) / D))

    pe[:, 0::2] = torch.sin(position * coeff)  # fill in even positions
    pe[:, 1::2] = torch.cos(position * coeff)  # fill in odd positions
    return pe
    
def split_heads(input, H):  # tensor shape is [B, L, D]
    Dh = input.shape[2] // H
    B, L, D = input.shape
    input = input.view(B, L, H, Dh)
    return input.permute(0, 2, 1, 3).contiguous() 

def combine_heads(input):  # tensor shape is [B, H, L, D]
    B, H, L, D = input.shape
    input = input.permute(0, 2, 1, 3).contiguous()
    return input.view(B, L, H * D)

def generate_causality_mask(target):
    L = target.size(1)
    return torch.tril(torch.ones(1, L, L), diagonal=0).int()


def scaled_dot_product_attention(Q, K, V, d_k, mask=None, negative_infinity=-1e9):
    # print(Q,K)
    attn_scores = torch.matmul(Q,K.transpose(-2,-1)) / (d_k**0.5) 
    if mask is not None:
      attn_scores = torch.where(mask == 0, negative_infinity,attn_scores) 
    attn_probabilities = nn.Softmax(dim=-1)
    attn_probabilities = attn_probabilities(attn_scores)
    output = torch.matmul(attn_probabilities, V) 
    return output

def create_embeddings(linear, input):
    batch_size, W, H, C = input.shape 
    patch_size = (W // 8, H // 8, C)
    
    final = []
    for k in range(batch_size):
        img = input[k]
        patches = []
        for i in range(0, W, patch_size[0]):
            for j in range(0, H, patch_size[1]):
                patch = img[i:i+patch_size[0], j:j+patch_size[1], :]
                x = patch.flatten()
                index = int((i/patch_size[0])*8 + j/patch_size[1])
                embed = linear[index](x)
                patches.append(embed)
        final.append(torch.stack(patches, dim=1))
    
    return torch.stack(final)