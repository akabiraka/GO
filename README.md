

## GO preprocessing tasks:
* `python data_preprocess/separate_and_remove_nonexist_uniprotid_from_GOA.py`

## UniProtKB-SwissProt sequence processing tasks:
* `python data_preprocess/create_uniprot_species_dict.py`
    * input files: data/uniprotkb/{species}.fasta
    * output files: data/uniprotkb/dicts/{species}.pkl

## GO annotation processing tasks:
* `python data_preprocess/separate_GOA_and_remove_nonexist_uniprotid_from_GOA.py`
    * input files: data/uniprotkb/dicts/{species}.pkl, data/GO/dicts/{GO}.pkl
    * output files: data/goa/{species}/separated_annotations/{GO}.csv
* `python data_preprocess/create_dev_test_set.py`
    * input files: data/goa/{species}/separated_annotations/{GO}.csv
    * output files: data/goa/{species}/dev_test_set/{GO}/dev.csv, data/goa/{species}/dev_test_set/{GO}/test.csv
* `python data_preprocess/expand_dev_test_set.py`
    * input files: data/goa/{species}/dev_test_set/{GO}/{dataset}.csv
    * output files: data/goa/{species}/expanded_dev_test_set/{GO}/{dataset}.csv

* `python data_preprocess/cutoff_dev_set_on_term_frequency.py`

* `python data_preprocess/update_test_set.py`

* `python data_preprocess/compute_studied_GO_terms_adj_matrix.py`

* `python data_preprocess/generate_dev_train_val_test_dataset_and_label_vec.py`


## Model development

