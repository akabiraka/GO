import sys
sys.path.append("../GO")

import os.path as osp
import numpy as np
import pandas as pd
import utils as Utils
from data_preprocess.GO import Ontology


go_rels = Ontology('data/downloads/go.obo', with_rels=True)

def create_studied_GO_terms_dict(GO_id_list, species, GO):
    GO_dict = {}
    for i, GO_id in enumerate(GO_id_list):
        GO_dict[GO_id] = i
    Utils.save_as_pickle(GO_dict, f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
    return GO_dict


def create_GO_topo_adj_matrix(GO_dict, species, GO):
    n_GO_terms = len(GO_dict)
    relation_matrix = np.zeros(shape=(n_GO_terms, n_GO_terms), dtype=np.int16) # realtion_matrix: R
    np.fill_diagonal(relation_matrix, 1) # encoding thyself as ancestor using 1

    for GO_id, i in GO_dict.items():
        ancestors = go_rels.get_anchestors(GO_id)
        for ancestor in ancestors:
            ancestor_index = GO_dict.get(ancestor)
            relation_matrix[i, ancestor_index] = 1

    Utils.save_as_pickle(relation_matrix, f"data/goa/{species}/studied_GO_terms_adj_matrix/{GO}.pkl")
    print(f"{species}-{GO}: {relation_matrix.shape}")
    

species = "yeast" # human, yeast

for GO in ["BP", "CC", "MF"]:
    dev_df = pd.read_csv(f"data/goa/{species}/dev_test_set_cutoff/{GO}/dev.csv")

    GO_id_list = dev_df["GO_id"].unique()
    GO_dict = create_studied_GO_terms_dict(GO_id_list, species, GO)
    
    
    create_GO_topo_adj_matrix(GO_dict, species, GO)

