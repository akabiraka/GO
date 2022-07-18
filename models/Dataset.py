import sys
sys.path.append("../GO")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils as Utils

class SeqAssociationDataset(Dataset):
    def __init__(self, species, GO, esm1b_batch_converter, max_len_of_a_seq=512, dataset="train") -> None:
        super(SeqAssociationDataset, self).__init__()
        self.species = species
        self.GO = GO
        self.max_len_of_a_seq = max_len_of_a_seq
        self.esm1b_batch_converter = esm1b_batch_converter

        self.df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/{dataset}.pkl")
        self.seq_db_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")
        self.terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
        


    
    def __len__(self):
        return self.df.shape[0]


    def generate_true_label(self, GO_terms):
        y_true = torch.zeros(len(self.terms_dict), dtype=torch.float32)
        for term in GO_terms:
            y_true[self.terms_dict[term]] = 1.
        return y_true



    def __getitem__(self, i):
        row = self.df.loc[i]
        uniprotid_seq, GO_terms = [(row["uniprot_id"], self.seq_db_dict.get(row["uniprot_id"])["seq"][:self.max_len_of_a_seq])], row["GO_id"]
        # print(uniprotid_seq)

        y_true = self.generate_true_label(GO_terms)

        uniprotid, batch_strs, seq_tokens = self.esm1b_batch_converter(uniprotid_seq)
        seq_tokens = seq_tokens[0]

        seq_int_rep = torch.ones(self.max_len_of_a_seq+1, dtype=torch.int32) ## NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        seq_int_rep[:seq_tokens.shape[0]] = seq_tokens
        # print(seq_int_rep.shape, y_true.shape)
        return seq_int_rep, y_true



def get_terms_to_dataset(species, GO):
    GO_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")

    data = {}
    data["src"] = torch.tensor(list(GO_dict.values())) # node embeddings from 0 to vocab_size-1
    adj_mat = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_adj_matrix/{GO}.pkl")
    data["attn_mask"] = torch.logical_not(torch.tensor(adj_mat, dtype=torch.bool))

    print(f"#-terms: {data['src'].shape}")
    print(f"terms relations: {data['attn_mask'].shape}")
    return data

    
