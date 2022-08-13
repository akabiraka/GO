import sys
sys.path.append("../GO")
from data_preprocess.GO import Ontology, NAMESPACES
import utils as Utils
import numpy as np
import collections
from sklearn.model_selection import train_test_split
import pandas as pd
import statistics

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

species = "yeast"
data_generation_process = "time_delay_no_knowledge"

t0 = 20200811 # 11 Aug 2020, dev deadline
t1 = 20220114 # 14 Jan 2022, test deadline

go_rels = Ontology('data/downloads/go.obo', with_rels=True)
bp_set = go_rels.get_namespace_terms(NAMESPACES["bp"])
cc_set = go_rels.get_namespace_terms(NAMESPACES["cc"])
mf_set = go_rels.get_namespace_terms(NAMESPACES["mf"])
# print(len(bp_set)) #30365
# print(len(cc_set)) #4423
# print(len(mf_set)) #12360
# no intersetion among these sets


species_uniprot_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")

bp_dev_annots = {} # uniprot_id, set of annots
cc_dev_annots = {}
mf_dev_annots = {}

bp_test_annots = {}
cc_test_annots = {}
mf_test_annots = {}


# def print_summary(annots:dict):
#     prots = [unitprot_id for unitprot_id, annots in annots.items()]
#     # print(prots) 
#     all_annots = np.hstack([list(annots) for unitprot_id, annots in annots.items()])
    
#     n_annots = len(all_annots)
#     n_terms = len(set(all_annots))
#     print(len(prots), n_annots, n_terms)

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
        del test_annots[key]


def update_annot_dict(uniprot_id, GO_id, annot_dict:dict):
    if uniprot_id in annot_dict.keys():
        annot_dict[uniprot_id] = annot_dict[uniprot_id] | set([GO_id])
    else: 
        annot_dict[uniprot_id] = set([GO_id])
    
    return annot_dict


f = open(f"data/downloads/{species}_goa.gpa", "r")
for i, line in enumerate(f.readlines()):
    # print(f"line no: {i}")

    if not line.startswith("UniProtKB"): continue

    line_items = line.split()
    uniprot_id = line_items[1]
    GO_id = line_items[3]
    evidence = line_items[-1].split("=")[1].upper()

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
    # at this point, we extracted 4 things: uniprot_id, GO_id, date, evidence


    # separating annotations according to the GO={BP, CC, MF} types
    if GO_id in bp_set: 
        if date<=t0: bp_dev_annots = update_annot_dict(uniprot_id, GO_id, bp_dev_annots)  # bp_dev
        elif date>t0 and date<=t1: bp_test_annots = update_annot_dict(uniprot_id, GO_id, bp_test_annots)# possible bp_test
    
    if GO_id in cc_set: 
        if date<=t0: cc_dev_annots = update_annot_dict(uniprot_id, GO_id, cc_dev_annots)  # cc_dev
        elif date>t0 and date<=t1: cc_test_annots = update_annot_dict(uniprot_id, GO_id, cc_test_annots)# possible cc_test
    
    if GO_id in mf_set:
        if date<=t0: mf_dev_annots = update_annot_dict(uniprot_id, GO_id, mf_dev_annots)  # mf_dev
        elif date>t0 and date<=t1: mf_test_annots = update_annot_dict(uniprot_id, GO_id, mf_test_annots)# possible mf_test



    # if i==32: break # for debugging


def do(dev_annots, test_annots, terms_cutoff_value, n_annots, go):
    print(f"#-seqs in dev: {len(dev_annots)}")
    print(f"#-seqs in test: {len(test_annots)}")

    # print(dev_annots["P32916"])
    # print(test_annots["P32916"])

    remove_dev_uniprotids_from_test(dev_annots, test_annots) # inplace operation
    print(f"#-seqs after keeping only no-knowledge proteins in test: {len(test_annots)}")

    remove_nonexist_uniprotids_from_dev_test(dev_annots)
    print(f"#-seqs after removing nonexist uniprotids from dev: {len(dev_annots)}")

    remove_nonexist_uniprotids_from_dev_test(test_annots)
    print(f"#-seqs after removing nonexist uniprotids from test: {len(test_annots)}")


    # apply true path rule
    dev_annots = apply_true_path_rule(dev_annots)
    test_annots = apply_true_path_rule(test_annots)


    # computing studied terms based on term frequency
    studied_terms = compute_studied_terms(dev_annots, terms_cutoff_value)
    save_studied_terms(list(studied_terms), go)
    print(f"#-of studied terms: {len(studied_terms)}")

    dev_annots = update_annots_with_studied_terms(dev_annots, studied_terms)
    remove_proteins_annotated_to_n_or_less_terms(dev_annots, n_annots)
    print(f"#-seqs after updating with studied terms and removing proteins annotated to only n or less terms for dev: {len(dev_annots)}")


    test_annots = update_annots_with_studied_terms(test_annots, studied_terms)
    remove_proteins_annotated_to_n_or_less_terms(test_annots, n_annots)
    print(f"#-seqs after updating with studied terms: {len(test_annots)}")


    train_annots, val_annots = train_test_split(list(dev_annots.items()), test_size=0.10)

    print_summary(list(dev_annots.items()))    
    print_summary(train_annots)
    print_summary(val_annots)
    print_summary(list(test_annots.items()))

    Utils.save_as_pickle(train_annots, f"data/goa/{species}/train_val_test_set/{data_generation_process}/{go}/train.pkl")
    Utils.save_as_pickle(val_annots, f"data/goa/{species}/train_val_test_set/{data_generation_process}/{go}/val.pkl")
    Utils.save_as_pickle(list(test_annots.items()), f"data/goa/{species}/train_val_test_set/{data_generation_process}/{go}/test.pkl")
    print()


    

# applying multiview-GCN cutoff values
# do(bp_dev_annots, bp_test_annots, terms_cutoff_value=125, n_annots=0, go="BP")
# do(cc_dev_annots, cc_test_annots, terms_cutoff_value=25, n_annots=0, go="CC")
# do(mf_dev_annots, mf_test_annots, terms_cutoff_value=25, n_annots=0, go="MF")


# applying DeepGO cutoff values
# do(bp_dev_annots, bp_test_annots, terms_cutoff_value=250, n_annots=0, go="BP")
# do(cc_dev_annots, cc_test_annots, terms_cutoff_value=50, n_annots=0, go="CC")
# do(mf_dev_annots, mf_test_annots, terms_cutoff_value=50, n_annots=0, go="MF")


# mine
do(bp_dev_annots, bp_test_annots, terms_cutoff_value=150, n_annots=0, go="BP")
do(cc_dev_annots, cc_test_annots, terms_cutoff_value=25, n_annots=0, go="CC")
do(mf_dev_annots, mf_test_annots, terms_cutoff_value=25, n_annots=0, go="MF")
