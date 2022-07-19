

## GO processing tasks:
* `python data_preprocess/separate_and_remove_nonexist_uniprotid_from_GOA.py`

## UniProtKB-SwissProt sequence processing tasks:
* `python data_preprocess/create_uniprot_species_dict.py`
    * input files: 
    * output files: 

## GO annotation processing tasks:
* `python data_preprocess/separate_GOA_and_remove_nonexist_uniprotid_from_GOA.py`
    * input files: 
    * output files: 
* `python data_preprocess/create_dev_test_set.py`
    * input files: 
    * output files: 
* `python data_preprocess/expand_dev_test_set.py`
    * input files: 
    * output files: 
* `python data_preprocess/cutoff_dev_set_on_term_frequency.py`
    * input files: 
    * output files: 
* `python data_preprocess/update_test_set.py`
    * input files: 
    * output files: 
* `python data_preprocess/compute_studied_GO_terms_adj_matrix.py`
    * input files: 
    * output files: 
* `python data_preprocess/compute_studied_terms_dict_and_relation_matrix.py`
    * input files: 
    * output files: 
* `python data_preprocess/create_train_val_test_set.py`
    * input files: 
    * output files: 

## Model development
* `python models/train_val.py`
* `python models/test.py`
* `python models/eval_pred_scores.py`

tensorboard --logdir=outputs/tensorboard_runs/
scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/GO/outputs/tensorboard_runs/* outputs/tensorboard_runs/
scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/GO/outputs/models/* outputs/models/
scp -r akabir4@argo.orc.gmu.edu:/scratch/akabir4/GO/outputs/predictions/* outputs/predictions/