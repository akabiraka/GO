import sys
sys.path.append("../GO")

import pandas as pd
import matplotlib.pyplot as plt
import statistics

# checking why BP/MF is hard than CC.
species = "yeast"

labels = []
for dataset in ["train", "val", "test"]:
    plt.cla()
    for GO in ["BP", "CC", "MF"]:
        df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{GO}/{dataset}.pkl")
        num_of_labels_list = df["GO_id"].apply(lambda labels: len(labels)).tolist()
        labels.append(num_of_labels_list)

        print(f"{dataset} mean: {statistics.mean(num_of_labels_list):.3f}, std: {statistics.stdev(num_of_labels_list):.3f}")
        
        bins = range(0, 150, 5)
        plt.hist(num_of_labels_list, bins, alpha=0.4, label=GO)
        plt.legend()

    # plt.show()
    plt.savefig(f"outputs/images/num_of_labels_distribution_on_GO_types_{dataset}.png", dpi=300, format="png", bbox_inches='tight', pad_inches=0.0)


# df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/BP/train.pkl")
# for i, row in df.iterrows():
#     n_labels = len(row["GO_id"])
#     # print(n_labels)
#     if n_labels in range(0, 3):
#         print(row["uniprot_id"], n_labels, row["GO_id"]) # this prints uniprot_id which is annotated with 1 or 2 terms
