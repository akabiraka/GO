import sys
sys.path.append("../GO")
import pandas as pd


species = "yeast" # human, yeast
for GO in ["BP", "CC", "MF"]:
    dataset = "test"  # "dev", "test"
    which_set = "dev_test_set_expanded" #dev_test_set, dev_test_set_expanded, dev_test_set_cutoff

    # filepath = f"data/goa/{species}/separated_annotations/{GO}.csv" # for separated annotations
    filepath = f"data/goa/{species}/{which_set}/{GO}/{dataset}.csv" # for other dev-test set
    

    df = pd.read_csv(filepath)
    # print(df.columns) #['line_no', 'uniprot_id', 'GO_id', 'date']

    print(f"for {GO}")
    print(f"    #-of annotations: {df.shape[0]}")
    print(f"    #-of unique proteins: {len(df['uniprot_id'].unique())}")
    print(f"    #-of unique GO terms: {len(df['GO_id'].unique())}")
    print()