import sys
sys.path.append("../GO_prediction")

import torch
import numpy as np
from transformer.config import Config
from models.GOTopoDataset import get_GO_topo_dataset, get_batched_dataset
from models.MultimodalTransformer import MultimodalTransformer
import utils as Utils

torch.cuda.empty_cache()

config = Config()

go_topo_data = get_GO_topo_dataset(config.species, config.GO)
uniprotid_seq_label_batches = get_batched_dataset(config, dataset="train")
uniprotid_seq_pairs, Y_labels = get_batched_dataset(config, dataset="train")


vocab_size = go_topo_data["src"].shape[0]
config.vocab_size = vocab_size
print(config.get_model_name())             


model = Model(config=config).to(config.device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

# for i, batch in enumerate(uniprotid_seq_label_batches):
#     go_nodes, go_nodes_adj_mat = go_topo_data["src"].to(config.device), go_topo_data["attn_mask"].to(config.device)
#     uniprotid_seq_pairs, Y_labels = batch[0], batch[1]

    
#     batch_labels, batch_strs, seq_batch_tokens = model.batch_converter(uniprotid_seq_pairs)
#     seq_batch_tokens = seq_batch_tokens.to(config.device)

#     # uniprotid_seq_pairs = [
#     #    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     #    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     #    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     #    ("protein3",  "K A <mask> I S Q"),
#     # ]
#     # batch_labels, batch_strs, seq_batch_tokens = model.batch_converter(uniprotid_seq_pairs)
#     # seq_batch_tokens = seq_batch_tokens.to(config.device)

#     y_pred = model(go_nodes, go_nodes_adj_mat, seq_batch_tokens, uniprotid_seq_pairs)
#     y_true = torch.tensor(np.array(batch[1])).to(config.device, dtype=torch.float32)
#     loss = criterion(y_pred, y_true)
#     loss.backward()
#     optimizer.step()
#     print('Loss: {:.3f}'.format(loss.item()))
#     break
    


go_nodes, go_nodes_adj_mat = go_topo_data["src"].to(config.device), go_topo_data["attn_mask"].to(config.device)

batch_labels, batch_strs, seq_batch_tokens = model.batch_converter(uniprotid_seq_pairs)
seq_batch_tokens = seq_batch_tokens.to(config.device)

y_pred = model(go_nodes, go_nodes_adj_mat, seq_batch_tokens)
y_true = torch.tensor(np.array(Y_labels)).to(config.device, dtype=torch.float32)
loss = criterion(y_pred, y_true)
loss.backward()
optimizer.step()
print('Loss: {:.3f}'.format(loss.item()))
