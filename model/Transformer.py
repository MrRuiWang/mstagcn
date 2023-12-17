import torch.nn as nn
import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.autograd import Variable


class Transformer(nn.Module):
    def __init__(self, input_len, d_model, dropout, max_len, d_k, n_heads, d_v, d_ff, n_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_len, d_model, dropout, max_len, d_k, n_heads, d_v, d_ff, n_layers)

    def forward(self, enc_inputs):
        # [batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        return enc_outputs

class Encoder(nn.Module):
    def __init__(self, input_len, d_model, dropout, max_len, d_k, n_heads, d_v, d_ff, n_layers):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(input_len, d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, n_heads, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.pos_emb(enc_inputs)
        enc_self_attn_mask = None

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, n_heads, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.LayerNorm = LayerNorm(self.d_model)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.LayerNorm(output + residual)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.n_heads = n_heads
        self.d_v = d_v
        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)
        self.LayerNorm = LayerNorm(self.d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = self.ScaledDotProductAttention(Q, K, V, )
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)

        return self.LayerNorm(output + residual), attn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, input_len, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        # log -> exp
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2.0) *
                             -(math.log(10000.0) / d_model))
        # position * div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, src_len, d_model]

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask=None):
        attention = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        if attn_mask:
            attention.masked_fill_(attn_mask, -1e9)  # 一般在decoder层用

        attention = self.softmax(attention)

        context = torch.matmul(attention, V)  # [batch_size, n_heads, len_q, d_v]

        return context, attention
