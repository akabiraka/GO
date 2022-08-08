import sys
sys.path.append("../GO")

import utils as Utils
# bp_test_prots = ['SYC1', 'YB92', 'HHY1', 'APQ13', 'BOP3', 'IRC9', 'MAL33', 'IRC16', 'HCS1', 'ARG56']
# cc_test_prots = ['SRPR', 'RRT1', 'ATG4', 'YB92', 'MAL11', 'MNN4', 'HHY1', 'APQ13', 'IRC9', 'MAL33', 'IRC16', 'ARG56']
# mf_test_prots = ['AHC2', 'TOP2', 'YPK2', 'YB92', 'RSB1', 'CTK3', 'RCR1', 'OST1', 'PDR18', 'LSB1', 'PEA2', 'OST2', 'NVJ1', 'HHY1', 'SFP1', 'APQ13', 'NIS1', 'MDY2', 'MRX6', 'MAL33', 'ENV9', 'IRC16', 'ECM3', 'EAF3', 'ARG56']

# # their test set
# bp_uniprot_ids, cc_uniprot_ids, mf_uniprot_ids = [], [], []
# yeast_db = Utils.load_pickle("data/uniprotkb/yeast.pkl")
# for uniprot_id, info in yeast_db.items():
#     prot_name = info["id"].split("|")[2][:-6]

#     if prot_name in bp_test_prots: 
#         bp_uniprot_ids.append(uniprot_id)
#     if prot_name in cc_test_prots: 
#         cc_uniprot_ids.append(uniprot_id)
#     if prot_name in mf_test_prots: 
#         mf_uniprot_ids.append(uniprot_id)

# # print(len(bp_uniprot_ids), len(cc_uniprot_ids), len(mf_uniprot_ids))
# # print(bp_uniprot_ids, cc_uniprot_ids, mf_uniprot_ids)
# bp_uniprot_ids, cc_uniprot_ids, mf_uniprot_ids = set(bp_uniprot_ids), set(cc_uniprot_ids), set(mf_uniprot_ids)

# my test set
bp_uniprot_ids = set(['P0C5R9', 'P32903', 'P36119', 'P38162', 'P40856', 'P40975', 'P40985', 'P53137', 'Q06010', 'Q12025'])
cc_uniprot_ids = set(['P0C5R9', 'P14180', 'P36123', 'P39101', 'P40856', 'P43612', 'P53961', 'Q12433'])
mf_uniprot_ids = set(['P03871', 'P03872', 'P08640', 'P0C5R9', 'P25574', 'P28625', 'P28737', 'P32366', 'P36039', 'P38836', 'P40492', 'P40540', 'P40851', 'P40975', 'P42845', 'P43534', 'P47133', 'P53012', 'P53073', 'Q03063', 'Q03373', 'Q03530', 'Q06891', 'Q12025', 'Q12094', 'Q12142', 'Q12208', 'Q12431', 'Q12508'])


from data_preprocess.GO import Ontology, NAMESPACES
species = "yeast"
EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP'])
go_rels = Ontology('data/downloads/go.obo', with_rels=True)
bp_set = go_rels.get_namespace_terms(NAMESPACES["bp"])
cc_set = go_rels.get_namespace_terms(NAMESPACES["cc"])
mf_set = go_rels.get_namespace_terms(NAMESPACES["mf"])
t0 = 20200811 # 11 Aug 2020, dev deadline
t1 = 20220114 # 14 Jan 2022, test deadline

f = open(f"data/downloads/{species}_goa.gpa", "r")
train_uniprot_ids_set, test_uniprot_ids_set = set(), set()
go_uniprot_ids_to_observe = mf_uniprot_ids # bp_uniprot_ids, cc_uniprot_ids, mf_uniprot_ids
go_set_to_observe = mf_set # bp_set, cc_set, mf_set
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

    

    if uniprot_id in go_uniprot_ids_to_observe and GO_id in go_set_to_observe:
        if date <= t0: 
            train_uniprot_ids_set.add(uniprot_id)
            date_rel = str(date) + "<=" + str(t0)

        elif date>t0 and date<=t1: 
            test_uniprot_ids_set.add(uniprot_id)
            date_rel = str(t0) + "<" + str(date) + "<=" + str(t1)

        
        print(uniprot_id, GO_id, date_rel, evidence)

test_uniprot_ids_set = test_uniprot_ids_set - train_uniprot_ids_set

print(train_uniprot_ids_set) # should be in the train set
print(test_uniprot_ids_set) # should be in the test set
print((go_uniprot_ids_to_observe - train_uniprot_ids_set) - test_uniprot_ids_set) # proteins which do not have any annotations in between t0 and t1