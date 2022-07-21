import sys
sys.path.append("../GO")

import pandas as pd
import numpy as np
import collections
import torch

species = "yeast"
GO_type = "CC"

#Checking if the proteins in the train, val and test set have at least 1 annotation from studied GO terms. 
# for dataset in ["train", "val", "test"]:
#     df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO_type}/{dataset}.pkl")
#     for i, row in df.iterrows():
#         annots = row["GO_id"]
#         # print(i, row["uniprot_id"], len(annots))
#         if len(annots) == 0:
#             print(dataset, GO_type, i, row["uniprot_id"], len(annots))


# checking num of annotations per terms in the train set 
# dataset = "train"
# df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO_type}/{dataset}.pkl")        
# all_labels = np.hstack(df["GO_id"].tolist())

# print(f"#-annotations: {len(all_labels)}")

# counter=collections.Counter(all_labels)
# print(len(counter))
# print(counter)

from data_preprocess.GO import Ontology
import utils as Utils

go_rels = Ontology('data/downloads/go.obo', with_rels=True)
studied_terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO_type}.pkl")
studied_terms_set = set(studied_terms_dict.keys())
idx_to_term_dict = {i:term for term, i in studied_terms_dict.items()}



y_pred = torch.rand(size=(2, 244)) #batch_size, vocab_size
# print(y_pred)
# y_labels = torch.where(y_pred>0.5, 1, 0)
# print(y_labels)

rows, cols = y_pred.shape

for i in range(rows):
    for j in range(cols):
        if y_pred[i, j] > 0.5:
            y_pred[i, j] = 1.
            GO_id = idx_to_term_dict[j]
            ancestors = go_rels.get_anchestors(GO_id)
            ancestors = set(ancestors).intersection(studied_terms_set) # taking ancestors only in the studied terms
            for term in ancestors:
                y_pred[i, studied_terms_dict[term]] = 1.

print(y_pred)
# pred_terms_indies = np.where(pred_scores[i] > threshold)[0]
# annots = set([idx_to_term_dict.get(i) for i in pred_terms_indies])

# new_annots = set()
# for go_id in annots:
#     new_annots = new_annots | go_rels.get_anchestors(go_id)
# preds.append(new_annots)