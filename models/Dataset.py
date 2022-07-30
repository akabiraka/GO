import sys
sys.path.append("../GO")

import pandas as pd
import torch
from torch.utils.data import Dataset
import utils as Utils
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random

class SeqAssociationDataset(Dataset):
    def __init__(self, species, GO, n_samples_from_pool=5, max_seq_len=512, dataset="train") -> None:
        super(SeqAssociationDataset, self).__init__()
        self.species = species
        self.GO = GO
        self.max_seq_len = max_seq_len
        self.n_samples = n_samples_from_pool


        self.df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/{dataset}.pkl")
        self.seq_db_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")
        self.terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
        

        self.dev_df = pd.read_csv(f"data/goa/{species}/dev_test_set_cutoff/{GO}/dev.csv")
        self.GOid_vs_uniprotid_list_df = self.dev_df.groupby("GO_id")["uniprot_id"].apply(list).reset_index() # GO-id vs list of uniprotid
        # GOid_vs_uniprotid_list_df["features"] = GOid_vs_uniprotid_list_df["uniprot_id"].map(lambda x: get_go_seq_features(x, crnt_uniprot_id))#random.sample(x, n_samples))
        # print(GOid_vs_uniprotid_list_df.head())
        self.terms_ancestors = Utils.load_pickle(f"data/goa/{self.species}/studied_GO_terms_relation_matrix/{self.GO}_ancestors.pkl")
        self.terms_children = Utils.load_pickle(f"data/goa/{self.species}/studied_GO_terms_relation_matrix/{self.GO}_children.pkl")
    

    def __len__(self):
        return self.df.shape[0]


    def generate_true_label(self, GO_terms):
        y_true = torch.zeros(len(self.terms_dict), dtype=torch.float32)
        for term in GO_terms:
            y_true[self.terms_dict[term]] = 1.
        return y_true



    def __getitem__(self, i):
        row = self.df.loc[i]
        uniprot_id, GO_terms = row["uniprot_id"], row["GO_id"]

        y_true = self.generate_true_label(GO_terms) # shape: [n_terms]
        seq_rep = self.get_seq_representation(uniprot_id) # shape: [max_seq_len+1, esm1b_embed_dim]
        terms_graph = self.get_terms_graph(uniprot_id) 

        return seq_rep, terms_graph, y_true


    def get_seq_representation(self, uniprot_id):
        seq_rep = Utils.load_pickle(f"data/uniprotkb/{self.species}_sequences_rep/{uniprot_id}.pkl") # shape: max_seq_len, esmb_embed_dim
        return seq_rep


    def get_terms_graph(self, crnt_uniprot_id):
        # the pool excludes crnt_uniprot_id
        nodes = []
        for term, id in self.terms_dict.items():
            # print(term, id)
            uniprotid_list = self.GOid_vs_uniprotid_list_df[self.GOid_vs_uniprotid_list_df["GO_id"]==term]["uniprot_id"].item()
            
            term_seq_features = self.get_term_seq_features(uniprotid_list, crnt_uniprot_id) 
            # print(term_seq_features.shape)
            nodes.append(term_seq_features)
            # break

        data = {}
        data["nodes"] = torch.stack(nodes)
        data["ancestors_rel_matrix"] = torch.logical_not(torch.tensor(self.terms_ancestors, dtype=torch.bool))
        # data["children_rel_matrix"] = torch.tensor(self.terms_children, dtype=torch.float32)

        return data    
    

    def get_term_seq_features(self, uniprotid_list, crnt_uniprot_id):
        uniprotid_list = list(filter((crnt_uniprot_id).__ne__, uniprotid_list)) # removing current uniprotid from seq-feature pool
        uniprotid_list = random.sample(uniprotid_list, self.n_samples)
        features = []
        for uniprot_id in uniprotid_list:
            seq_rep = Utils.load_pickle(f"data/uniprotkb/{self.species}_sequences_rep/{uniprot_id}.pkl") # shape: max_seq_len, esmb_embed_dim
            features.append(seq_rep)
        
        features = torch.stack(features)
        # print(features.shape) # n_samples, max_seq_len+1, esm1b_embed_dim
        return features


# sample usage
# val_dataset = SeqAssociationDataset("yeast", "CC", dataset="val")
# seq_rep, terms_graph, y_true = val_dataset.__getitem__(0)
# print(seq_rep.shape, y_true.shape, terms_graph["nodes"].shape, terms_graph["ancestors_rel_matrix"].shape)
# torch.Size([513, 768]) torch.Size([244]) torch.Size([244, 5, 513, 768]) torch.Size([244, 244])



# deprecated
# def get_terms_dataset(species, GO):
#     GO_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")

#     data = {}
#     data["nodes"] = torch.tensor(list(GO_dict.values())) # node embeddings from 0 to vocab_size-1
    
#     ancestors = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_ancestors.pkl")
#     data["ancestors_rel_matrix"] = torch.logical_not(torch.tensor(ancestors, dtype=torch.bool))

#     children = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_children.pkl")
#     data["children_rel_matrix"] = torch.tensor(children, dtype=torch.float32)


#     print(f"#-terms: {data['nodes'].shape}")
#     print(f"ancestors_rel_matrix: {data['ancestors_rel_matrix'].shape}")
#     print(f"children_rel_matrix: {data['children_rel_matrix'].shape}")
#     return data

    

def get_class_weights(species, GO):
    # computing class weights from the train data
    terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
    train_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/train.pkl")
    
    classes = np.array([value for key, value in terms_dict.items()])
    all_labels = [terms_dict[term] for term in np.hstack(train_df["GO_id"].tolist())]

    class_weights = compute_class_weight("balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def get_positive_class_weights(species, GO):
    terms_to_idx_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
    train_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/train.pkl")

    def generate_true_label(GO_terms):
        y_true = np.zeros(len(terms_to_idx_dict), dtype=np.int32)
        for term in GO_terms:
            y_true[terms_to_idx_dict.get(term)] = 1
        return y_true


    all_labels = []
    for i, row in train_df.iterrows():
        GO_terms = row["GO_id"]
        y_true = generate_true_label(GO_terms)
        all_labels.append(y_true)

    all_labels = np.array(all_labels)

    positive_cls_weights = []
    for i in range(all_labels.shape[1]):
        n_pos = (all_labels[:, i]==1).sum()
        n_neg = (all_labels[:, i]==0).sum()
        weight = n_neg / n_pos
        positive_cls_weights.append(weight)


    return torch.tensor(positive_cls_weights, dtype=torch.float32)

# get_positive_class_weights("yeast", "BP")