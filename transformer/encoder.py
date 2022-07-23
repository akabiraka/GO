import sys
from turtle import forward
sys.path.append("../GO")
import copy
import torch
import torch.nn as nn

    
class AttentionSublayerConnection(nn.Module):
    def __init__(self, dim_embed, attention, dropout=0.3) -> None:
        super(AttentionSublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        y = self.norm(x)
        y, attn_weights = self.attention(y, y, y, key_padding_mask, attn_mask)
        x = x + self.dropout(y)
        return x, attn_weights



class FeedForwardSublayerConnection(nn.Module):
    def __init__(self, dim_embed, feed_forward, dropout=0.3) -> None:
        super(FeedForwardSublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(dim_embed)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = feed_forward
    
    def forward(self, x):
        return x + self.dropout(self.feed_forward(self.norm(x)))



class TermsSeqRelationSublayer(nn.Module):
    def __init__(self, dim_embed, termsSeqRelationForward, dropout=0.3) -> None:
        super(TermsSeqRelationSublayer, self).__init__()
        self.batch_norm = nn.BatchNorm1d(dim_embed)
        self.dropout = nn.Dropout(dropout)
        self.termsSeqRelationForward = termsSeqRelationForward
    
    def forward(self, x, seq_reps):
        rel_scores = x.matmul(seq_reps.t())
        return x + self.dropout(self.batch_norm(self.termsSeqRelationForward(rel_scores)))



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"
    def __init__(self, dim_embed, termsSeqRelationForward, attention, feed_forward, dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.dim_embed = dim_embed
        self.terms_seq_relation_sublayer = TermsSeqRelationSublayer(dim_embed, termsSeqRelationForward, dropout)
        self.attn_sublayer = AttentionSublayerConnection(dim_embed, attention, dropout)
        self.feed_forward_sublayer = FeedForwardSublayerConnection(dim_embed, feed_forward, dropout)

    def forward(self, x, seq_reps, key_padding_mask=None, attn_mask=None):
        """seq_reps:[batch_size, embed_dim]"""
        x = self.terms_seq_relation_sublayer(x, seq_reps)
        x, attn_weights = self.attn_sublayer(x, key_padding_mask, attn_mask)
        x = self.feed_forward_sublayer(x)
        return x, attn_weights



class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(enc_layer, n_layers)
        self.norm = nn.LayerNorm(enc_layer.dim_embed)
        
    def forward(self, x, seq_reps, key_padding_mask=None, attn_mask=None, return_attn_weights=False):
        # store and return the attention weights for all layers and heads
        if return_attn_weights:
            all_layers_attn_weights = []
            for _, layer in enumerate(self.layers):
                x, attn_weights = layer(x, key_padding_mask, attn_mask)
                all_layers_attn_weights.append(attn_weights)
            
            all_layers_attn_weights = torch.stack(all_layers_attn_weights, dim=0)
            # print(all_layers_attn_weights.shape) 
            return self.norm(x), all_layers_attn_weights # all_layers_attn_weights: [n_layers, batch_size, n_heads, max_len, max_len]
        
        # does not store attention weights
        else:
            for _, layer in enumerate(self.layers):
                x, _ = layer(x, seq_reps, key_padding_mask, attn_mask)
            return self.norm(x), None 