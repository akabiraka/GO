import sys
sys.path.append("../GO")
import pandas as pd

species = "yeast" # human, yeast
dataset = "dev"
GO_term_freq_cutoff_th = [125, 25, 25]

for i, GO in enumerate(["BP", "CC", "MF"]):
    input_filepath = f"data/goa/{species}/dev_test_set_expanded/{GO}/{dataset}.csv"
    output_filepath = f"data/goa/{species}/dev_test_set_cutoff/{GO}/{dataset}.csv"

    df = pd.read_csv(input_filepath)

    vc_mask = df["GO_id"].value_counts() >= GO_term_freq_cutoff_th[i]

    df = df[df["GO_id"].map(vc_mask)]

    df["freq"] = df["GO_id"].map(df["GO_id"].value_counts())

    df.to_csv(output_filepath, index=False)
    print(f"{species}-{GO}-{dataset}: {df.shape}")


# an example of "map" usage
# students = [['jackma', 34, 'Sydeny', 'Australia'],
#             ['Ritika', 30, 'Delhi', 'India'],
#             ['Vansh', 31, 'Delhi', 'India'],
#             ['Nany', 32, 'Tokyo', 'Japan'],
#             ['May', 16, 'New York', 'US'],
#             ['Michael', 17, 'las vegas', 'US']]
  
# # Create a DataFrame object
# df = pd.DataFrame(students, columns=['Name', 'Age', 'City', 'Country'])
# print(df)

# print(df["Country"].value_counts())
# vc_mask = df["Country"].value_counts() >=2

# df = df[df['Country'].map(vc_mask)]

# df["freq"] = df['Country'].map(df["Country"].value_counts())

# print(df)