import sys
sys.path.append("../GO")
import esm
import utils as Utils
import torch

species = "yeast"
max_seq_len = 512



esm1b, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
esm1b_batch_converter = alphabet.get_batch_converter()

seq_db_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")


for uniprot_id, all_info in seq_db_dict.items():
    seq = all_info["seq"][:max_seq_len]
    uniprotid_seq = [(uniprot_id, seq)]
    uniprotid, batch_strs, seq_tokens = esm1b_batch_converter(uniprotid_seq)

    seq_int_rep = torch.ones((1, max_seq_len+1), dtype=torch.int32) # esm1b padding token is 1
    seq_int_rep[0, :seq_tokens.shape[1]] = seq_tokens # shape: [1, max_seq_len]
    
    with torch.no_grad():
        results = esm1b(seq_int_rep, repr_layers=[12], return_contacts=False)
    seq_rep = results["representations"][12] #1, max_seq_len, esmb_embed_dim
    seq_rep.squeeze_(0)
    

    Utils.save_as_pickle(seq_rep, f"data/uniprotkb/{species}_sequences_rep/{uniprot_id}.pkl")
    # seq_rep = Utils.load_pickle(f"data/uniprotkb/{species}_sequences_rep/{uniprot_id}.pkl")

    print(uniprot_id, seq_rep.shape)
    # break