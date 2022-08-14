import sys
sys.path.append("../GO")
import torch

class Config(object):
    def __new__(cls):
        # Singleton class instance
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self, 
                 species="yeast", 
                 GO="CC", 
                 lr=1e-4, 
                 batch_size=32, 
                 n_epochs=500,
                 max_len_of_a_seq=512, 
                 embed_dim=256,
                 n_samples_from_pool=5, 
                 n_encoder_layers=3, 
                 n_attn_heads=2, 
                 dropout=0.5,
                 add_node_embed_layer=False, 
                 add_positional_encoding_layer=False) -> None:
        super(Config, self).__init__()

        self.species = species #"yeast"
        self.GO = GO #["BP", "CC", "MF"]

        # training specific config
        self.lr = lr #1e-4
        self.batch_size = batch_size #32
        self.n_epochs = n_epochs #1000

        # Input configs
        vocab_sizes = {"BP": 287, "CC": 246, "MF": 432}
        self.vocab_size = vocab_sizes[self.GO] #[0, 20] inclusive for 20 amino acids [1, 20] and 0 for padding
        self.max_num_of_nodes = self.vocab_size # this will not be used
        self.max_len_of_a_seq = max_len_of_a_seq #1024, this also means number of nodes for graphs
        self.embed_dim = embed_dim #256, embed_dim must be divisible by n_attn_heads
        self.n_samples_from_pool = n_samples_from_pool

        # Encoder configs
        self.dim_ff =  4*self.embed_dim # if dim_ff is None else dim_ff
        self.n_encoder_layers = n_encoder_layers
        self.n_attn_heads = n_attn_heads
        self.dropout = dropout
        self.add_node_embed_layer = add_node_embed_layer
        self.add_positional_encoding_layer = add_positional_encoding_layer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # "cpu"#
        self.esm1b_embed_dim = 768

        self.data_generation_process = "time_delay_no_knowledge" # time_series_no_knowledge, time_delay_no_knowledge, random_split_leakage
        

    def get_model_name(self, task="Modelv3.4") -> str:
        return f"{task}_{self.species}_{self.GO}_{self.lr}_{self.batch_size}_{self.n_epochs}_{self.vocab_size}_{self.max_len_of_a_seq}_{self.embed_dim}_{self.n_samples_from_pool}" +\
               f"_{self.dim_ff}_{self.n_encoder_layers}_{self.n_attn_heads}_{self.dropout}_{self.add_node_embed_layer}_{self.add_positional_encoding_layer}" +\
               f"_{self.device}"


# config = Config(max_len=2708)
# print(config.get_model_name())


# separate decoder config 
# Decoder related configs
# self.n_classes = n_classes

