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



def get_related_terms(GO_id, relation="ancestors"):
    if relation=="ancestors":
        terms = go_rels.get_anchestors(GO_id)
    elif relation=="children":
        terms = go_rels.get_children(GO_id)
    elif relation=="parents":
        terms = go_rels.get_parents(GO_id)
    elif relation=="adjacency":
        terms = go_rels.get_parents(GO_id)
    else:
        raise NotImplementedError(f"Given relation={relation} is not implemented yet.")
    
    return terms




# i-th row denotes the ancestor/children-indices of i if corresponding entry is 1
def create_terms_relation_matrix(GO_dict, species, GO, relation="ancestors"):
    # relation could be [ancestors, children, parents]

    studied_terms_set = set(GO_dict.keys())

    n_GO_terms = len(GO_dict)
    relation_matrix = np.zeros(shape=(n_GO_terms, n_GO_terms), dtype=np.int16) # realtion_matrix: R
    np.fill_diagonal(relation_matrix, 1) # adding self loop

    for GO_id, i in GO_dict.items():
        terms = get_related_terms(GO_id, relation)
        terms = studied_terms_set.intersection(terms)
        for term in terms:
            term_i = GO_dict.get(term)
            relation_matrix[i, term_i] = 1
            if relation=="adjacency": relation_matrix[term_i, i] = 1

    Utils.save_as_pickle(relation_matrix, f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_{relation}.pkl")
    print(f"{species}-{GO}: {relation_matrix.shape}")
    print(f"Is it symmetric: {(relation_matrix==relation_matrix.T).all()}")



    

species = "yeast" # human, yeast

for GO in ["BP", "CC", "MF"]:
    dev_df = pd.read_csv(f"data/goa/{species}/dev_test_set_cutoff/{GO}/dev.csv")

    GO_id_list = dev_df["GO_id"].unique()
    GO_dict = create_studied_GO_terms_dict(GO_id_list, species, GO)
    
    
    create_terms_relation_matrix(GO_dict, species, GO, relation="ancestors")
    create_terms_relation_matrix(GO_dict, species, GO, relation="parents")
    create_terms_relation_matrix(GO_dict, species, GO, relation="children")
    create_terms_relation_matrix(GO_dict, species, GO, relation="adjacency")
    # break

