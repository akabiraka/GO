import sys
sys.path.append("../GO")
import os
import pandas as pd
from data_preprocess.GO import Ontology

species = "yeast" # human, yeast
dataset = "test"  # "dev", "test"
GO_types = ["BP", "CC", "MF"]

go_rels = Ontology('data/downloads/go.obo', with_rels=True)


def is_redundent(goa_GO_df, goa_with_uniprot_info):
    return ((goa_GO_df["uniprot_id"]==goa_with_uniprot_info["uniprot_id"]) & (goa_GO_df["GO_id"]==goa_with_uniprot_info["GO_id"])).any()


def expand(GO):
    input_filepath = f"data/goa/{species}/dev_test_set/{GO}/{dataset}.csv"
    output_filepath = f"data/goa/{species}/dev_test_set_expanded/{GO}/{dataset}.csv"

    df = pd.read_csv(input_filepath)

    expanded_df = pd.DataFrame(columns=df.columns)
    # print(df)

    for i, row in df.iterrows():
        print(f"expanding {GO}:{i}")
        expanded_df = expanded_df.append(df.loc[i], ignore_index=True)

        ancestors = go_rels.get_anchestors(row["GO_id"])
        for ancestor in ancestors:
            goa_with_ancestor_info = {"line_no": "-", "uniprot_id": row["uniprot_id"], "GO_id": ancestor, "date": "-"}
            if not is_redundent(expanded_df, goa_with_ancestor_info):
                expanded_df = expanded_df.append(goa_with_ancestor_info, ignore_index=True)
        
        # break

    expanded_df.to_csv(output_filepath, index=False)
    print(f"{species}-{GO}-{dataset}: {expanded_df.shape}")



if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
    i = int(os.environ["SLURM_ARRAY_TASK_ID"]) 
    expand(GO_types[i])
else:
    for GO in GO_types:
        expand(GO)
