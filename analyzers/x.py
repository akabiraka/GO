import sys
sys.path.append("../GO")

from transformer.config import Config
import utils as Utils
import torch


class ProjectionLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim, dropout=0.3):
        super(ProjectionLayer, self).__init__()
        self.dropout = dropout
        self.attn_linear = torch.nn.Linear(inp_embed_dim, 1)
        self.projection = torch.nn.Linear(inp_embed_dim, out_embed_dim)

    def forward(self, last_hidden_state):
        """last_hidden_state (torch.Tensor): shape [batch_size, seq_len, dim_embed]"""
        # x = torch.mean(x, dim=1) #global average pooling. shape [batch_size, dim_embed]
        activation = torch.tanh(last_hidden_state) # [batch_size, seq_len, dim_embed]

        score = self.attn_linear(activation) # [batch_size, seq_len, 1]      
        weights = torch.softmax(score, dim=1) # [batch_size, seq_len, 1]
        seq_reps = torch.sum(weights * last_hidden_state, dim=1)  # [batch_size, dim_embed]
        seq_reps = self.projection(seq_reps)
        # seq_reps = F.relu(seq_reps)
        # seq_reps = F.dropout(seq_reps, p=self.dropout)
        return seq_reps

import esm

seq_proj_layer = ProjectionLayer(768, 256)
node_proj_layer = ProjectionLayer(256, 256)

x = torch.randint(low=0, high=21, size=(2, 3, 2, 5)) #batch_size, n_nodes, n_samples, max_seq_len
# print(x)
esm1b, alphabet = esm.pretrained.esm1_t12_85M_UR50S()

batch_size, n_nodes, n_samples, seq_len = x.shape
batches = []
for i in range(batch_size):
    nodes_rep = []
    for j in range(n_nodes):
        with torch.no_grad():
            results = esm1b(x[i, j], repr_layers=[12], return_contacts=False) #n_nodes, max_seq_len, esmb_embed_dim
        rep = results["representations"][12]
        rep = seq_proj_layer(rep) # n_nodes, embed_dim
        nodes_rep.append(rep)
    
    nodes_rep = node_proj_layer(torch.stack(nodes_rep)) 
    batches.append(nodes_rep)

batches = torch.stack(batches) #batch_size, n_nodes, embed_dim
print(batches.shape)