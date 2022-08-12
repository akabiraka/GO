import sys
sys.path.append("../GO")
from data_preprocess.GO import Ontology, NAMESPACES
import utils as Utils
import numpy as np
np.random.seed(123456)
import collections
from sklearn.model_selection import train_test_split
import pandas as pd
import statistics

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

species = "yeast"
t0 = 20000101 # 1 Jan 2000
t1 = 20220114 # 14 Jan 2022

timeline = list(range(t0, t1, 3000))
timeseries = [(timeline[i], timeline[i+1]) for i in range(len(timeline)-1)]
# print(timeseries)
train_timeseries, test_timeseries = train_test_split(timeseries, test_size=0.20)
train_timeseries, val_timeseries = train_test_split(train_timeseries, test_size=0.10)
print(len(train_timeseries), len(val_timeseries), len(test_timeseries))
# print(train_timeseries, val_timeseries, test_timeseries)


go_rels = Ontology('data/downloads/go.obo', with_rels=True)
bp_set = go_rels.get_namespace_terms(NAMESPACES["bp"])
cc_set = go_rels.get_namespace_terms(NAMESPACES["cc"])
mf_set = go_rels.get_namespace_terms(NAMESPACES["mf"])
# print(len(bp_set)) #30365
# print(len(cc_set)) #4423
# print(len(mf_set)) #12360
# no intersetion among these sets


species_uniprot_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")


def print_summary(dataset_annots:list):
    all_annots = np.hstack([list(annots) for unitprot_id, annots in dataset_annots])
    prots = [unitprot_id for unitprot_id, annots in dataset_annots]
    terms = set(all_annots)
    print(f"    #-proteins: {len(prots)}, #-annotations: {len(all_annots)}, #-terms: {len(terms)}")

    num_of_labels_list = [len(annots) for unitprot_id, annots in dataset_annots]
    print(f"    num_of_labels_per_protein_distribution: mean, std: {statistics.mean(num_of_labels_list):.3f}, {statistics.stdev(num_of_labels_list):.3f}")


def save_studied_terms(studied_terms_list, go):
    GO_dict = {}
    for i, GO_id in enumerate(studied_terms_list):
        GO_dict[GO_id] = i
    Utils.save_as_pickle(GO_dict, f"data/goa/{species}/studied_GO_id_to_index_dicts/{go}.pkl")
    # print(GO_dict)


def remove_proteins_annotated_to_n_or_less_terms(go_dev_annots:dict, n):
    uniprot_ids_to_remove = set()
    for uniprot_id, annots in go_dev_annots.items():
        if len(annots) <= n:
            uniprot_ids_to_remove = uniprot_ids_to_remove | set([uniprot_id])
    
    for id in uniprot_ids_to_remove: # removing proteins which is not annotated with at least 3 terms
        del go_dev_annots[id]

def update_annots_with_studied_terms(go_dev_annots:dict, studied_terms:set):
    for uniprot_id, annots in go_dev_annots.items():
        new_annots = set(annots).intersection(studied_terms)
        go_dev_annots[uniprot_id] = new_annots

    return go_dev_annots


def compute_studied_terms(annots:dict, cutoff_value):
    all_annots = np.hstack([list(annots) for unitprot_id, annots in annots.items()])
    term_freq_dict = collections.Counter(all_annots)

    studied_terms = set()
    for GO_id, count in term_freq_dict.items():
        if count>cutoff_value:
            studied_terms = studied_terms | set([GO_id])
            # print(count)

    return studied_terms


def apply_true_path_rule(go_dataset_annots:dict):
    for uniprot_id, annots in go_dataset_annots.items():
        expanded_annots = set()
        for go_id in annots:
            ancestors = go_rels.get_anchestors(go_id)
            expanded_annots = expanded_annots | ancestors # set union
        go_dataset_annots[uniprot_id] = set(expanded_annots)
    return go_dataset_annots


def remove_nonexist_uniprotids_from_dev_test(annots:dict):
    uniprotids_to_remove = set(annots.keys()) - set(species_uniprot_dict.keys())
    for key in uniprotids_to_remove:
        del annots[key]


def remove_dev_uniprotids_from_test(dev_annots:dict, test_annots:dict): # inplace operation
    uniprotids_to_remove_from_test = set(dev_annots.keys()).intersection(set(test_annots.keys()))
    for key in uniprotids_to_remove_from_test:
        dev_annots[key] = dev_annots[key] | test_annots[key] # annotations which got into test/val from time series are adding into train
        del test_annots[key]
    return dev_annots, test_annots


def update_annot_dict(uniprot_id, GO_id, date, annot_dict:dict):
    if uniprot_id in annot_dict.keys():
        annot_dict[uniprot_id] = annot_dict[uniprot_id] | set([GO_id])
    else: 
        annot_dict[uniprot_id] = set([GO_id])
    return annot_dict


def check_timeseries(date:int):
    for (start, end) in train_timeseries:
        if date>=start and date<end:
            return "train"
    for (start, end) in val_timeseries:
        if date>=start and date<end:
            return "val"
    for (start, end) in test_timeseries:
        if date>=start and date<end:
            return "test"
        

def generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=125, atleast_n_annots=0):
    train_set, val_set, test_set = {}, {}, {} # uniprot_id, set of annots

    f = open(f"data/downloads/{species}_goa.gpa", "r")
    for i, line in enumerate(f.readlines()):
        # print(f"line no: {i}")

        if not line.startswith("UniProtKB"): continue

        line_items = line.split()
        uniprot_id = line_items[1]
        GO_id = line_items[3]
        evidence = line_items[-1].split("=")[1].upper()

        # validate evidence code
        if evidence not in EXP_CODES: continue

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

        # print(uniprot_id, GO_id, date, evidence)
        

        if GO_id in GO_terms_set and date<=t1: 
            if check_timeseries(date) == "train": train_set = update_annot_dict(uniprot_id, GO_id, date, train_set)
            elif check_timeseries(date) == "val": val_set = update_annot_dict(uniprot_id, GO_id, date, val_set)
            elif check_timeseries(date) == "test": test_set = update_annot_dict(uniprot_id, GO_id, date, test_set)
        
        # if i==32: break # for debugging

    print(t0,t1)
    print(f"#-prots in train, val, test: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    print_summary(list(train_set.items()))

    train_set, val_set = remove_dev_uniprotids_from_test(train_set, val_set) # inplace operation
    train_set, test_set = remove_dev_uniprotids_from_test(train_set, test_set) # inplace operation
    print(f"#-prots in train, val, test after keeping only no-knowledge proteins in val and test: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    print_summary(list(train_set.items())) # the number of annotations is increased in train_set, becasue no-knowledge proteins annotations are be added from val/test into train.


    train_set = apply_true_path_rule(train_set)
    val_set = apply_true_path_rule(val_set)
    test_set = apply_true_path_rule(test_set)
    print("\nSummary of sets after applying true-path-rule: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))


    studied_terms = compute_studied_terms(train_set, cutoff_value)
    save_studied_terms(list(studied_terms), go=GOname)
    print(f"\n#-of studied terms: {len(studied_terms)}")

    train_set = update_annots_with_studied_terms(train_set, studied_terms)
    val_set = update_annots_with_studied_terms(val_set, studied_terms)
    test_set = update_annots_with_studied_terms(test_set, studied_terms)
    print("\nSummary of sets after updating annotations with studied GO terms: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))

    remove_proteins_annotated_to_n_or_less_terms(train_set, n=atleast_n_annots)
    remove_proteins_annotated_to_n_or_less_terms(val_set, n=atleast_n_annots)
    remove_proteins_annotated_to_n_or_less_terms(test_set, n=atleast_n_annots)
    print("\nSummary of sets after removing proteins having <=n annotations: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))

    Utils.save_as_pickle(list(train_set.items()), f"data/goa/{species}/train_val_test_set/{GOname}/train.pkl")
    Utils.save_as_pickle(list(val_set.items()), f"data/goa/{species}/train_val_test_set/{GOname}/val.pkl")
    Utils.save_as_pickle(list(test_set.items()), f"data/goa/{species}/train_val_test_set/{GOname}/test.pkl")


# generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=150, atleast_n_annots=0)
# generate_dataset(GOname="CC", GO_terms_set=cc_set, cutoff_value=25, atleast_n_annots=0)
generate_dataset(GOname="MF", GO_terms_set=mf_set, cutoff_value=25, atleast_n_annots=0)

