import sys
sys.path.append("../GO")

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from transformer.config import Config
from models.Dataset import SeqAssociationDataset, TermsGraph, get_class_weights, get_positive_class_weights
import models.MultimodalTransformer as MultimodalTransformer

import eval_metrics as eval_metrics


config = Config()
out_filename = config.get_model_name()
print(out_filename)



# loading model, criterion, optimizer, summarywriter
model = MultimodalTransformer.Model(config=config).to(config.device)
class_weights = get_class_weights(config.species, config.GO).to(config.device)
# pos_class_weights = get_positive_class_weights(config.species, config.GO).to(config.device)
label_pred_criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")
print("log: model loaded")


# loading dataset
terms_graph = TermsGraph(config.species, config.GO, config.n_samples_from_pool)
train_dataset = SeqAssociationDataset(config.species, config.GO, dataset="train")
val_dataset = SeqAssociationDataset(config.species, config.GO, dataset="val")
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")




best_loss, best_f1 = np.inf, np.inf
for epoch in range(1, config.n_epochs+1):
    train_loss = MultimodalTransformer.train(model, train_loader, terms_graph, label_pred_criterion, optimizer, config.device)
    val_loss, true_scores, pred_scores = MultimodalTransformer.val(model, val_loader, terms_graph, label_pred_criterion, config.device)

    print(f"Epoch: {epoch:03d}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1_TPR(true_scores, pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1(true_scores, pred_scores)
    # micro_avg_precision = eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
    # fmax = eval_metrics.Fmax(true_scores, pred_scores)

    writer.add_scalar('TrainLoss', train_loss, epoch)
    writer.add_scalar('ValLoss', val_loss, epoch)
    # writer.add_scalar('MicroAvgF1', micro_avg_f1, epoch)
    # writer.add_scalar('MicroAvgPrecision', micro_avg_precision, epoch)
    writer.add_scalar('Fmax', fmax, epoch)

    # save model dict based on loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}_loss.pth")


    # save model dict based on performance
    if fmax < best_f1:
        best_fmax = fmax
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}_pref.pth")

    
