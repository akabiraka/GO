import sys
sys.path.append("../GO")
import pandas as pd
import utils as Utils
from sklearn.model_selection import train_test_split


def create_train_val_set(species, GO_type):
    dev_df = pd.read_csv(f"data/goa/{species}/dev_test_set_cutoff/{GO_type}/dev.csv")

    uniprotid_vs_GO_list_df = dev_df.groupby("uniprot_id")["GO_id"].apply(list).reset_index() # uniprotid vs list of GO-id
    # print(uniprotid_vs_GO_list_df)

    train_df, val_df = train_test_split(uniprotid_vs_GO_list_df, test_size=0.15)
    train_df, val_df = train_df.reset_index(), val_df.reset_index()
    train_df, val_df = train_df.drop(["index"], axis=1), val_df.drop(["index"], axis=1)
    
    
    print(f"{species}-{GO_type}-train: {train_df.shape}")
    print(f"{species}-{GO_type}-val: {val_df.shape}")
    # print(train_df)

    pd.to_pickle(train_df, f"data/goa/{species}/train_val_test_set/{GO_type}/train.pkl")
    pd.to_pickle(val_df, f"data/goa/{species}/train_val_test_set/{GO_type}/val.pkl")
    
    # train_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO_type}/train.pkl")
    # print(train_df)
    # break
    return train_df, val_df


def create_test_set(species, GO_type):
    terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO_type}.pkl")
    test_df = pd.read_csv(f"data/goa/{species}/dev_test_set_expanded/{GO_type}/test.csv")

    test_df = test_df[test_df["GO_id"].isin(terms_dict)].reset_index().drop(["index"], axis=1) # removing annotations which are not in studied GO-terms
    print(test_df.shape)
    test_df = test_df.groupby("uniprot_id")["GO_id"].apply(list).reset_index() # uniprotid vs list of GO-id

    print(f"{species}-{GO_type}-test: {test_df.shape}")
    pd.to_pickle(test_df, f"data/goa/{species}/train_val_test_set/{GO_type}/test.pkl")
    return test_df

species = "yeast"
for GO_type in ["BP", "CC", "MF"]:
    create_train_val_set(species, GO_type)
    create_test_set(species, GO_type)
    print()