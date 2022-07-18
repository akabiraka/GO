import sys
sys.path.append("../GO")

import pandas as pd
import math
from data_preprocess.GO import Ontology


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


go_rels = Ontology('data/downloads/go.obo', with_rels=True)


train_df = pd.read_pickle('data/train_data.pkl')
test_df = pd.read_pickle('data/test_data.pkl')
print("Length of test set: " + str(len(test_df)))

annotations = train_df['prop_annotations'].values
annotations = list(map(lambda x: set(x), annotations))
test_annotations = test_df['prop_annotations'].values
test_annotations = list(map(lambda x: set(x), test_annotations)) # list(list) -> list(set)
go_rels.calculate_ic(annotations + test_annotations)

fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, test_annotations, test_annotations)
print(fscore, prec, rec, s, ru, mi)