import torch
import esm

device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#

# Load ESM-1b model
model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
model = model.to(device)
# print(model)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print(batch_labels)
print(batch_labels)
print(batch_tokens)
batch_tokens = batch_tokens.to(device)

# Extract per-residue representations (on CPU)
#with torch.no_grad():
results = model(batch_tokens, repr_layers=[12], return_contacts=True)
token_representations = results["representations"][12]
print(token_representations)

## Generate per-sequence representations via averaging
## NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
#sequence_representations = []
#for i, (_, seq) in enumerate(data):
#    sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
#print(sequence_representations[0].shape)
#
## Look at the unsupervised self-attention map contact predictions
#import matplotlib.pyplot as plt
#for (_, seq), attention_contacts in zip(data, results["contacts"]):
#    plt.matshow(attention_contacts[: len(seq), : len(seq)])
#    plt.title(seq)
#    plt.show()
