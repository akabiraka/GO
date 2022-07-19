import sys
sys.path.append("../GO")

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from transformer.config import Config
from models.Dataset import SeqAssociationDataset, get_terms_to_dataset
import models.MultimodalTransformer as MultimodalTransformer

import eval_metrics as eval_metrics

config = Config()
out_filename = config.get_model_name()
print(out_filename)


# loading model, criterion, optimizer, summarywriter
model = MultimodalTransformer.Model(config=config).to(config.device)
criterion = torch.nn.BCEWithLogitsLoss()


# loading dataset
go_topo_data = get_terms_to_dataset(config.species, config.GO)
test_dataset = SeqAssociationDataset(config.species, config.GO, model.batch_converter, config.max_len_of_a_seq, dataset="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


test_loss, true_scores, pred_scores = MultimodalTransformer.val(model, test_loader, go_topo_data, criterion, config.device)
eval_metrics.Fmax_Smin_AUPR(pred_scores, species="yeast", GO="CC", eval_dataset="test")
eval_metrics.MicroAvgF1(true_scores, pred_scores)
eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
eval_metrics.Fmax(true_scores, pred_scores)
eval_metrics.AUROC(true_scores, pred_scores)
eval_metrics.AUPR(true_scores, pred_scores)