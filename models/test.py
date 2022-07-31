import sys
sys.path.append("../GO")

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from transformer.config import Config
from models.Dataset import SeqAssociationDataset, TermsGraph, get_class_weights
import models.MultimodalTransformer as MultimodalTransformer

import utils as Utils

config = Config()
out_filename = config.get_model_name()
out_filename = out_filename+"_pref" #_loss
print(f"Running test: {out_filename}")


# loading model, criterion, optimizer, summarywriter
model = MultimodalTransformer.Model(config=config).to(config.device)
class_weights = get_class_weights(config.species, config.GO).to(config.device)
# pos_class_weights = get_positive_class_weights(config.species, config.GO).to(config.device)
criterion = torch.nn.BCEWithLogitsLoss(class_weights)

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


# loading dataset
terms_graph = TermsGraph(config.species, config.GO, config.n_samples_from_pool)
test_dataset = SeqAssociationDataset(config.species, config.GO, dataset="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


test_loss, true_scores, pred_scores = MultimodalTransformer.val(model, test_loader, terms_graph, criterion, config.device)
Utils.save_as_pickle(true_scores, f"outputs/predictions/{out_filename}_true_scores.pkl")
Utils.save_as_pickle(pred_scores, f"outputs/predictions/{out_filename}_pred_scores.pkl")
