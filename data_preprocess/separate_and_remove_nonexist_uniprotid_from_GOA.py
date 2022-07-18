import sys
sys.path.append("../GO")
import utils as Utils
import pandas as pd
from data_preprocess.GO import Ontology, NAMESPACES


species = "yeast" # yeast, human

go_rels = Ontology('data/downloads/go.obo', with_rels=True)
bp_set = go_rels.get_namespace_terms(NAMESPACES["bp"])
cc_set = go_rels.get_namespace_terms(NAMESPACES["cc"])
mf_set = go_rels.get_namespace_terms(NAMESPACES["mf"])


species_uniprot_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")


columns = ["line_no", "uniprot_id", "GO_id", "date"]
goa_BP_df = pd.DataFrame(columns=columns, dtype=object)
goa_CC_df = pd.DataFrame(columns=columns, dtype=object)
goa_MF_df = pd.DataFrame(columns=columns, dtype=object)


def update(goa_GO_df, goa_with_uniprot_info):
    query = (goa_GO_df["uniprot_id"]==goa_with_uniprot_info["uniprot_id"]) & (goa_GO_df["GO_id"]==goa_with_uniprot_info["GO_id"])
    if query.any():  #  if a pair of (uniprot_id, GO_id) is already in the dataset
        i = goa_GO_df[query].index
        if goa_GO_df.loc[i, "date"].item() < goa_with_uniprot_info["date"]: # update with the newer date
            goa_GO_df.loc[i, "date"] = goa_with_uniprot_info["date"]
    else: # else append the data
        goa_GO_df = goa_GO_df.append(goa_with_uniprot_info, ignore_index=True)
    return goa_GO_df




f = open(f"data/goa/{species}/goa.gpa", "r")
for i, line in enumerate(f.readlines()):
    print(f"line no: {i}")
    # if i<28536: continue # for debugging

    if line.startswith("UniProtKB"):
        line_items = line.split()
        uniprot_id = line_items[1]
        GO_id = line_items[3]


        # validate GO_id
        if not GO_id.startswith("GO:"):
            print(f"GO id issue detected at line {i}: {GO_id}")
            break

        # validate date
        if line_items[-3].isdigit() and len(line_items[-3])==8:
            date = int(line_items[-3])
        elif line_items[-4].isdigit() and len(line_items[-4])==8:
            date = int(line_items[-4])
        else: 
            print(f"Date issue detected at line {i}: {date}")
            break

        # removing those annotations that does not have valid uniprot-id
        if uniprot_id in species_uniprot_dict:
            record = species_uniprot_dict.get(uniprot_id)
        else:
            print(f"UniProtKB {uniprot_id} not found")
            continue

        
        goa_with_uniprot_info = {"line_no":i, "uniprot_id":uniprot_id, "GO_id":GO_id, "date":date}


        # separating the annotations according to the GO={BP, CC, MF} types
        if GO_id in bp_set: goa_BP_df = update(goa_BP_df, goa_with_uniprot_info)
        elif GO_id in cc_set: goa_CC_df = update(goa_CC_df, goa_with_uniprot_info)
        elif GO_id in mf_set: goa_MF_df = update(goa_MF_df, goa_with_uniprot_info)

    # if i==28600: break # for debugging

print(goa_BP_df.shape)
print(goa_CC_df.shape)
print(goa_MF_df.shape)

# print(goa_BP_df)
# print(goa_CC_df)
# print(goa_MF_df)


goa_BP_df.to_csv(f"data/goa/{species}/separated_annotations/BP.csv", index=False)
goa_CC_df.to_csv(f"data/goa/{species}/separated_annotations/CC.csv", index=False)
goa_MF_df.to_csv(f"data/goa/{species}/separated_annotations/MF.csv", index=False)
    

    