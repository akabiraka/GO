import sys
sys.path.append("../GO")
import pandas as pd

species = "yeast" # human, yeast

t0 = 20200811 # 11 Aug 2020
t1 = 20220114 # 14 Jan 2022

for GO in ["BP", "CC", "MF"]:
    annotations_df = pd.read_csv(f"data/goa/{species}/separated_annotations/{GO}.csv")

    dev_df = annotations_df[annotations_df["date"] <= t0]
    test_df = annotations_df[(annotations_df["date"] > t0) & (annotations_df["date"] <= t1)]

    print(f"for {GO}: {annotations_df.shape}, {dev_df.shape}, {test_df.shape}")
    print(f"#-of lost GOA: {annotations_df.shape[0] - dev_df.shape[0] - test_df.shape[0]}")

    dev_df.to_csv(f"data/goa/{species}/dev_test_set/{GO}/dev.csv", index=False)
    test_df.to_csv(f"data/goa/{species}/dev_test_set/{GO}/test.csv", index=False)
