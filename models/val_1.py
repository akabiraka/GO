import sys
sys.path.append("../GO")

import models.MultimodalTransformer as MultimodalTransformer
import eval_metrics_1 as eval_metrics
from transformer.config import Config
from data_preprocess.GO import Ontology
import utils as Utils
from torch.utils.data import DataLoader
from models.Dataset_1 import SeqAssociationDataset

config = Config()
eval_set = "val" #test, val

# for evaluation purposes
go_rels = Ontology('data/downloads/go.obo', with_rels=True)
term_to_idx_dict = Utils.load_pickle(f"data/goa/{config.species}/studied_GO_id_to_index_dicts/{config.GO}.pkl")
idx_to_term_dict = {i:term for term, i in term_to_idx_dict.items()}
terms_set = set(term_to_idx_dict.keys())

train_dataset = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.GO}/train.pkl") # list of uniprot_id, set([terms])
print(f"Length of train set: {len(train_dataset)}")

val_set = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.GO}/{eval_set}.pkl")
print(f"Length of eval set: {len(val_set)}")


val_annotations = [annots for uniprot_id, annots in val_set]
train_annotations = [annots for uniprot_id, annots in train_dataset]
go_rels.calculate_ic(train_annotations + val_annotations)

print("Log: finished computing ic")

val_dataset = SeqAssociationDataset(config.species, config.GO, dataset="val")
val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)
print(f"val batches: {len(val_loader)}")

def run_val(model, terms_graph, label_pred_criterion):
    val_loss, true_scores, pred_scores = MultimodalTransformer.val(model, val_loader, terms_graph, label_pred_criterion, config.device)

    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores, val_set, idx_to_term_dict, go_rels, terms_set, val_annotations)
    # tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1_TPR(true_scores, pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1(true_scores, pred_scores)
    # micro_avg_precision = eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
    # fmax = eval_metrics.Fmax(true_scores, pred_scores)

    return val_loss, tmax, fmax, smin, aupr