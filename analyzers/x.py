import sys
sys.path.append("../GO")

from transformer.config import Config
import utils as Utils
import torch

config = Config()
out_filename = config.get_model_name()
out_filename = out_filename+"_pref" #_loss
print(f"Running test: {out_filename}")
pred_scores = Utils.load_pickle(f"outputs/predictions/{out_filename}_pred_scores.pkl")
print(torch.sigmoid(torch.tensor(pred_scores)))