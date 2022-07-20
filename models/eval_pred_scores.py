import sys
sys.path.append("../GO")

from transformer.config import Config
import eval_metrics as eval_metrics
import utils as Utils

config = Config()
out_filename = config.get_model_name()
out_filename = out_filename+"_perf" #_loss
print(f"Running evaluation: {out_filename}")

true_scores = Utils.load_pickle(f"outputs/predictions/{out_filename}_true_scores.pkl")
pred_scores = Utils.load_pickle(f"outputs/predictions/{out_filename}_pred_scores.pkl")
print(true_scores.shape, pred_scores.shape)

eval_metrics.Fmax_Smin_AUPR(pred_scores, species="yeast", GO="CC", eval_dataset="test")
eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
eval_metrics.MicroAvgF1(true_scores, pred_scores)
eval_metrics.AUROC(true_scores, pred_scores, pltpath=f"outputs/images/{out_filename}_auroc.pdf")
eval_metrics.Fmax(true_scores, pred_scores)
eval_metrics.AUPR(true_scores, pred_scores, pltpath=f"outputs/images/{out_filename}_aupr.pdf")