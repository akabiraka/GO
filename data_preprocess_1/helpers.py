import sys
sys.path.append("../GO")

from data_preprocess.GO import Ontology, NAMESPACES
import utils as Utils
import statistics
import numpy as np
import collections

species = "yeast"
EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])

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


def update_annot_dict(uniprot_id, GO_id, annot_dict:dict):
    if uniprot_id in annot_dict.keys():
        annot_dict[uniprot_id] = annot_dict[uniprot_id] | set([GO_id])
    else: 
        annot_dict[uniprot_id] = set([GO_id])
    return annot_dict


