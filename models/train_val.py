import sys
sys.path.append("../GO")

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from transformer.config import Config
from models.Dataset import SeqAssociationDataset, get_terms_to_dataset
import models.MultimodalTransformer as MultimodalTransformer


config = Config()
out_filename = config.get_model_name()
print(out_filename)


# loading model, criterion, optimizer, summarywriter
model = MultimodalTransformer.Model(config=config).to(config.device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")



# loading dataset
go_topo_data = get_terms_to_dataset(config.species, config.GO)
train_dataset = SeqAssociationDataset(config.species, config.GO, model.batch_converter, config.max_len_of_a_seq, dataset="train")
# print(train_dataset.__getitem__(0))
val_dataset = SeqAssociationDataset(config.species, config.GO, model.batch_converter, config.max_len_of_a_seq, dataset="val")
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")





best_loss = np.inf
for epoch in range(1, config.n_epochs+1):
    train_loss = MultimodalTransformer.train(model, train_loader, go_topo_data, criterion, optimizer, config.device)
    val_loss, metrics = MultimodalTransformer.val(model, val_loader, go_topo_data, criterion, config.device)

    print(f"Epoch: {epoch:03d}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    writer.add_scalar('TrainLoss', train_loss, epoch)
    writer.add_scalar('ValLoss', val_loss, epoch)
    writer.add_scalar('MicroAvgF1', metrics["MicroAvgF1"], epoch)
    writer.add_scalar('MicroAvgPrecision', metrics["MicroAvgPrecision"], epoch)

    # save model dict
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}.pth")

    