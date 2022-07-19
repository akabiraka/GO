import sys
sys.path.append("../GO")

from transformer.config import Config
import eval_metrics as eval_metrics
import utils as Utils

out_filename = "Model_yeast_CC_1e-05_14_500_244_512_256_1024_2_8_0.3_True_False_cuda"

true_scores = Utils.load_pickle(f"outputs/predictions/{out_filename}_true_scores.pkl")
pred_scores = Utils.load_pickle(f"outputs/predictions/{out_filename}_pred_scores.pkl")

eval_metrics.Fmax_Smin_AUPR(pred_scores, species="yeast", GO="CC", eval_dataset="test")
eval_metrics.MicroAvgF1(true_scores, pred_scores)
eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
eval_metrics.Fmax(true_scores, pred_scores)
eval_metrics.AUROC(true_scores, pred_scores)
eval_metrics.AUPR(true_scores, pred_scores)