#!/usr/bin/sh

#SBATCH --job-name=GO
#SBATCH --output=/scratch/akabir4/GO/outputs/argo_logs/argo-%j.out
#SBATCH --error=/scratch/akabir4/GO/outputs/argo_logs/argo-%j.err
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

##--------------CPU jobs------------------
##SBATCH --partition=all-LoPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=16000MB

##python data_preprocess/compute_GO_terms_topo_matrix.py.py


##--------------CPU array jobs------------------
##SBATCH --partition=all-LoPri
##SBATCH --cpus-per-task=4
##SBATCH --mem=16000MB
##SBATCH --array=0-2

##python data_preprocess/expand_dev_test_set.py


##--------------GPU jobs------------------
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:4
#SBATCH --mem=64000MB

##nvidia-smi
python models/train_val.py

##python models/test.py
##python models/eval_pred_scores.py

##python models/example_esm_1b.py
