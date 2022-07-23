import sys
sys.path.append("../GO")
import numpy as np
import torch
import torch.nn.functional as F
import esm
from transformer.config import Config
from transformer.factory import build_transformer_model

class Model(torch.nn.Module):
    def __init__(self, config:Config) -> None:
        super(Model, self).__init__()
        self.config = config

        self.GOTopoTransformer = build_transformer_model(config=config, decoder=None) # returns only node embeddings

        self.SeqTransformer, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()

        self.projection_layer = ProjectionLayer(config.emsb_embed_dim, config.embed_dim)

        self.prediction_layer = torch.nn.Linear(config.vocab_size, config.vocab_size)
        

    def forward(self, go_nodes, go_nodes_adj_mat, seq_batch_tokens):

        # print(seq_batch_tokens.shape)
        #with torch.no_grad():
        results = self.SeqTransformer(seq_batch_tokens, repr_layers=[12], return_contacts=False)
        token_reps = results["representations"][12] #n_seq, max_seq_len, esmb_embed_dim
        # print(f"token_reps: {token_reps.shape}")

        seq_reps = self.projection_layer(token_reps) #seq_reps:[batch_size, embed_dim]
        # print(f"seq_reps: {seq_reps.shape}")

        GO_terms_rep = self.GOTopoTransformer(x=go_nodes, seq_reps=seq_reps, key_padding_mask=None, attn_mask=go_nodes_adj_mat)
        # print(f"GO_terms_reps: {GO_terms_rep.shape}")

        scores = seq_reps.matmul(GO_terms_rep.t())
        # print(f"scores: {scores.shape}")

        out = self.prediction_layer(scores)
        return out



class ProjectionLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim):
        super(ProjectionLayer, self).__init__()

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
        seq_reps = F.relu(seq_reps)
        seq_reps = F.dropout(seq_reps, p=self.config.dropout)
        return seq_reps



# from data_preprocess.GO import Ontology
# import utils as Utils

# config = Config()
# go_rels = Ontology('data/downloads/go.obo', with_rels=True)
# studied_terms_dict = Utils.load_pickle(f"data/goa/{config.species}/studied_GO_id_to_index_dicts/{config.GO}.pkl")
# studied_terms_set = set(studied_terms_dict.keys())
# idx_to_term_dict = {i:term for term, i in studied_terms_dict.items()}

# def compute_loss(y_pred, y_true, criterion):
#     rows, cols = y_pred.shape
#     for i in range(rows):
#         for j in range(cols):
#             if y_pred[i, j] > 0.5:
#                 y_pred[i, j] = 1.
#                 GO_id = idx_to_term_dict[j]
#                 ancestors = go_rels.get_anchestors(GO_id)
#                 ancestors = set(ancestors).intersection(studied_terms_set) # taking ancestors if they re in the studied terms
#                 for term in ancestors:
#                     y_pred[i, studied_terms_dict[term]] = 1.

#     # print(y_pred)
#     batch_loss = criterion(y_pred, y_true)
#     return batch_loss, y_pred


def train(model, data_loader, go_topo_data, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    go_nodes, go_nodes_adj_mat = go_topo_data["src"].to(device), go_topo_data["attn_mask"].to(device)
    for i, (seq_tokens, y_true) in enumerate(data_loader):
        seq_tokens, y_true = seq_tokens.to(device), y_true.to(device)
        # print(seq_tokens.shape, y_true.shape)

        model.zero_grad(set_to_none=True)
        y_pred = model(go_nodes, go_nodes_adj_mat, seq_tokens)
        
        batch_loss, _ = criterion(y_pred, y_true) #compute_loss(y_pred, y_true, criterion) 
        batch_loss.backward()
        optimizer.step()
        
        train_loss = train_loss + batch_loss.item()
        # print(f"    train batch: {i}, loss: {batch_loss.item()}")
    return train_loss/len(data_loader)



@torch.no_grad()
def val(model, data_loader, go_topo_data, criterion, device):
    model.eval()
    val_loss = 0.0
    pred_scores, true_scores = [], []

    go_nodes, go_nodes_adj_mat = go_topo_data["src"].to(device), go_topo_data["attn_mask"].to(device)
    for i, (seq_tokens, y_true) in enumerate(data_loader):
        seq_tokens, y_true = seq_tokens.to(device), y_true.to(device)
        # print(seq_tokens.shape, y_true.shape)

        model.zero_grad(set_to_none=True)
        y_pred = model(go_nodes, go_nodes_adj_mat, seq_tokens)
        
        batch_loss, y_pred = criterion(y_pred, y_true) #compute_loss(y_pred, y_true, criterion) 
        val_loss = val_loss + batch_loss.item()
        
        pred_scores.append(y_pred.detach().cpu().numpy())
        true_scores.append(y_true.detach().cpu().numpy())

        # print(f"    val batch: {i}, loss: {batch_loss.item()}")

    true_scores, pred_scores = np.vstack(true_scores), np.vstack(pred_scores)
    return val_loss/len(data_loader), true_scores, pred_scores



