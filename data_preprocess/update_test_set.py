import sys
sys.path.append("../GO")
import pandas as pd
import utils as Utils

species = "yeast" # human, yeast

for GO in ["BP", "CC", "MF"]:
    terms_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")
    test_df = pd.read_csv(f"data/goa/{species}/dev_test_set_expanded/{GO}/test.csv")
    test_df = test_df[test_df["GO_id"].isin(terms_dict)].reset_index().drop(["index"], axis=1) # removing annotations which are not in studied GO-terms


    output_filepath = f"data/goa/{species}/dev_test_set_cutoff/{GO}/test.csv"
    test_df.to_csv(output_filepath, index=False)

    print(f"for {GO}")
    print(f"    #-of annotations: {test_df.shape[0]}")
    print(f"    #-of unique proteins: {len(test_df['uniprot_id'].unique())}")
    print(f"    #-of unique GO terms: {len(test_df['GO_id'].unique())}")
    print()