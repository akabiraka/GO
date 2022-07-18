import numpy as np

import sklearn.metrics as metrics
# Article1: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6 (An expanded evaluation ... )


def Fmax(y_true:np.ndarray, y_scores:np.ndarray):
    fmax=0.0
    decision_th = 0.0

    for t in range(1, 101):
        th = t/100 # according to Article1
        y_pred = np.where(y_scores>th, 1, 0)

        prec = metrics.precision_score(y_true, y_pred, average="micro", zero_division=1)
        rec = metrics.recall_score(y_true, y_pred, average="micro", zero_division=1)

        f = (2*prec*rec) / (prec+rec)
        if f > fmax: 
            fmax = f
            decision_th = th
            
    print(f"    Fmax: {fmax} at decision_th: {decision_th}")



def AUROC(y_true:np.ndarray, y_scores:np.ndarray):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    score1 = metrics.roc_auc_score(y_true, y_scores)
    print(f"    AUROC: {score1}")
    


def AUPR(y_true:np.ndarray, y_scores:np.ndarray):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    prec, rec, t = metrics.precision_recall_curve(y_true, y_scores)
    v = metrics.auc(rec, prec)
    print(f"    AUPR: {v}")



def MicroAvgF1(y_true:np.ndarray, y_scores:np.ndarray):
    y_pred = np.where(y_scores>0.7, 1, 0)
    micro_avg_f1 = metrics.f1_score(y_true, y_pred, average="micro")
    print(f"    MicroAvgF1: {micro_avg_f1}")



def MicroAvgPrecision(y_true:np.ndarray, y_scores:np.ndarray):
    micro_avg_prec = metrics.average_precision_score(y_true, y_scores, average="micro")
    print(f"    MicroAvgPrecision: {micro_avg_prec}")


# y_true = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
# y_scores = torch.tensor([[1., .6, 1.], [0., 1., 3.]])
# y_true, y_scores = y_true.detach().cpu().numpy(), y_scores.detach().cpu().numpy()

y_true = np.load("outputs/y_true_Deepour_pretrain_gcn_yeast_cc.npy") # shape: [n_terms, n_protein]
y_scores = np.load("outputs/y_scores_Deepour_pretrain_gcn_yeast_cc.npy") # shape: [n_terms, n_protein]

y_true_p, y_scores_p = y_true.transpose(), y_scores.transpose() # shape: [n_protein, n_terms]
print(y_true.shape, y_scores.shape)

# according to DeepGOA and Article1

# term centric
AUROC(y_true, y_scores)
AUPR(y_true, y_scores)

# protein centric
Fmax(y_true_p, y_scores_p)
MicroAvgPrecision(y_true_p, y_scores_p)
MicroAvgF1(y_true_p, y_scores_p)
