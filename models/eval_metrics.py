import sys
sys.path.append("../GO")

import numpy as np
import pandas as pd
import math
from data_preprocess.GO import Ontology
import utils as Utils
import sklearn.metrics as metrics



def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns



def Fmax_Smin_AUPR(pred_scores, species="yeast", GO="CC", eval_dataset="test"):

    go_rels = Ontology('data/downloads/go.obo', with_rels=True)
    train_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/train.pkl")
    train_annotations = train_df['GO_id'].values
    train_annotations = list(map(lambda x: set(x), train_annotations))
    print("Length of train set: " + str(len(train_df)))

    test_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/{eval_dataset}.pkl")
    test_annotations = test_df['GO_id'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    print("Length of evaluation set: " + str(len(test_df)))

    go_rels.calculate_ic(train_annotations + test_annotations)
    print("Log: finished computing ic")
    terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")

    # pred_scores = np.random.rand(869, 244)
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    for t in range(1, 101): # the range in this loop has influence in the AUPR output
        threshold = t / 100.0
        preds = []
        for i, row in enumerate(test_df.itertuples()):
            pred_terms_indies = np.where(pred_scores[i] > threshold)[0]
            annots = set([terms_dict.get(i) for i in pred_terms_indies])
            new_annots = set()
            for go_id in annots:
                new_annots = new_annots | go_rels.get_anchestors(go_id)
            preds.append(new_annots)

        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, test_annotations, preds)
        # print(fscore, prec, rec, s, ru, mi)

        avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
        avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
        #print(f'{avg_fp} {avg_ic}')
        precisions.append(prec)
        recalls.append(rec)
        # print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s

    print(f'    threshold: {tmax}')
    print(f'    Smin: {smin:0.3f}')
    print(f'    Fmax: {fmax:0.3f}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'    AUPR: {aupr:0.3f}')
    return tmax, fmax, smin, aupr



def MicroAvgF1(true_scores:np.ndarray, pred_scores:np.ndarray):
    best_micro_avg_f1 = 0.0
    for t in range(1, 101):
        th = t/100
        pred_scores = np.where(pred_scores>th, 1, 0)
        micro_avg_f1 = metrics.f1_score(true_scores, pred_scores, average="micro")
        if micro_avg_f1 > best_micro_avg_f1:
            best_micro_avg_f1 = micro_avg_f1
    print(f'    MicroAvgF1: {best_micro_avg_f1:0.3f}')
    return best_micro_avg_f1



def MicroAvgPrecision(true_scores:np.ndarray, pred_scores:np.ndarray):
    micro_avg_prec = metrics.average_precision_score(true_scores, pred_scores, average="micro")
    print(f'    MicroAvgPrecision: {micro_avg_prec:0.3f}')
    return micro_avg_prec



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
    return fmax



def AUROC(y_true:np.ndarray, y_scores:np.ndarray):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    auroc = metrics.roc_auc_score(y_true, y_scores)
    print(f"    AUROC: {auroc}")
    return auroc



def AUPR(y_true:np.ndarray, y_scores:np.ndarray):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    prec, rec, t = metrics.precision_recall_curve(y_true, y_scores)
    aupr = metrics.auc(rec, prec)
    print(f"    AUPR: {aupr}")
    return aupr