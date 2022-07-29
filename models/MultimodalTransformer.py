import sys
from turtle import forward
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

        self.esm1b, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = alphabet.get_batch_converter()

        self.GOTopoTransformer = build_transformer_model(config=config, decoder=None) # returns only node embeddings
        self.term_embedding_layer = TermEmbeddingLayer(config)

        self.seq_projection_layer = ProjectionLayer(config.emsb_embed_dim, config.embed_dim, config.dropout)

        self.prediction_refinement_layer = PredictionRefinementLayer(config.vocab_size, config.vocab_size, config.dropout)
        

    def forward(self, go_nodes, terms_ancestors_rel_mat, terms_children_rel_mat, seq_batch_tokens):
        # print(seq_batch_tokens.shape)
        with torch.no_grad():
            results = self.esm1b(seq_batch_tokens, repr_layers=[12], return_contacts=False)
        token_reps = results["representations"][12] #n_seq, max_seq_len, esmb_embed_dim
        # print(f"token_reps: {token_reps.shape}")
        
        seqs_reps = self.seq_projection_layer(token_reps) #seqs_reps:[batch_size, embed_dim]
        # print(f"seqs_reps: {seqs_reps.shape}")
        
        terms_reps = self.term_embedding_layer(x=go_nodes, esm1b=self.esm1b)
        terms_reps = self.GOTopoTransformer(x=go_nodes, key_padding_mask=None, attn_mask=terms_ancestors_rel_mat)
        # print(f"terms_reps: {terms_reps.shape}")
        
        scores = self.prediction_refinement_layer(seqs_reps, terms_reps, terms_children_rel_mat)
        return scores



class PredictionRefinementLayer(torch.nn.Module):
    def __init__(self, inp_embed_dim, out_embed_dim, dropout=0.3) -> None:
        super(PredictionRefinementLayer, self).__init__()
        self.dropout = dropout
        # self.w1 = torch.nn.Linear(inp_embed_dim, out_embed_dim)

    def forward(self, seqs_reps, terms_reps, terms_children_rel_mat):
        scores = seqs_reps.matmul(terms_reps.t()) # shape: n_seqs, n_terms
        # scores = scores.matmul(terms_children_rel_mat.t())
        # scores = self.w1(scores)

        return scores



class TermEmbeddingLayer(torch.nn.Module):
    def __init__(self, config:Config) -> None:
        super(TermEmbeddingLayer, self).__init__()
        self.seq_proj_layer = ProjectionLayer(config.emsb_embed_dim, config.embed_dim, config.dropout)
        self.node_proj_layer = ProjectionLayer(config.embed_dim, config.embed_dim, config.dropout)

    def forward(self, x, esm1b):
        batch_size, n_nodes, n_samples, seq_len = x.shape
        batches = []
        for i in range(batch_size):
            nodes_rep = []
            for j in range(n_nodes):
                with torch.no_grad():
                    results = esm1b(x[i, j], repr_layers=[12], return_contacts=False) #n_nodes, max_seq_len, esmb_embed_dim
                rep = results["representations"][12]
                rep = self.seq_proj_layer(rep) # n_nodes, embed_dim
                print(rep.shape)
                nodes_rep.append(rep)
            
            nodes_rep = self.node_proj_layer(torch.stack(nodes_rep)) 
            batches.append(nodes_rep)

        batches = torch.stack(batches) #batch_size, n_nodes, embed_dim
        return batches



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






def train(model, data_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0

    for i, (seq_tokens, y_true, terms) in enumerate(data_loader):
        y_true = y_true.to(device)
        # print(y_true.shape)

        model.zero_grad(set_to_none=True)
        y_pred = model(terms["nodes"].to(device), terms["ancestors_rel_matrix"].to(device), terms["children_rel_matrix"].to(device), seq_tokens.to(device))
        
        # batch_loss, _ = compute_loss(y_pred, y_true, criterion) 
        batch_loss = criterion(y_pred, y_true)

        batch_loss.backward()
        optimizer.step()
        
        train_loss = train_loss + batch_loss.item()
        print(f"    train batch: {i}, loss: {batch_loss.item()}")
        break
    return train_loss/len(data_loader)



@torch.no_grad()
def val(model, data_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    pred_scores, true_scores = [], []

    for i, (seq_tokens, y_true, terms) in enumerate(data_loader):
        y_true = y_true.to(device)
        # print(y_true.shape)

        model.zero_grad(set_to_none=True)
        y_pred = model(terms["nodes"].to(device), terms["ancestors_rel_matrix"].to(device), terms["children_rel_matrix"].to(device), seq_tokens.to(device))
        
        # batch_loss, y_pred = compute_loss(y_pred, y_true, criterion) 
        batch_loss = criterion(y_pred, y_true)

        val_loss = val_loss + batch_loss.item()
        
        pred_scores.append(torch.sigmoid(y_pred).detach().cpu().numpy())
        true_scores.append(y_true.detach().cpu().numpy())

        # print(f"    val batch: {i}, loss: {batch_loss.item()}")
        break
    true_scores, pred_scores = np.vstack(true_scores), np.vstack(pred_scores)
    return val_loss/len(data_loader), true_scores, pred_scores



