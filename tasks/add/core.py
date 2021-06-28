import torch
import torch.nn as nn
from torch.nn.modules import activation
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from tasks.add.config import CONFIG, config
import numpy as np

class StateEncoder(nn.Module):
    """
    Build the Encoder Network (f_enc) taking the environment state (env_in) and the program
    arguments (arg_in), feeding through a Multilayer Perceptron, to generate the state encoding
    (s_t).

    Reed, de Freitas only specify that the f_enc is a Multilayer Perceptron => As such we use
    two ELU Layers, up-sampling to a state vector with dimension 128.

    Reference: Reed, de Freitas [9]
    """
    def __init__(self, hidden_dim:int=100, state_dim=128, batch_size=1, method:str='maxout'):
        super(StateEncoder, self).__init__()
        self.env_row, self.env_col, self.env_depth = config.env_shape
        self.arg_num, self.arg_depth = config.arg_shape
        self.prog_num, self.prog_key_size, self.prog_embedding_size = config.prog_num, config.prog_key_size, config.prog_embedding_size
        self.hidden_dim, self.state_dim, self.batch_size = hidden_dim, state_dim, batch_size
        self.env_dim = self.env_row * self.env_depth # one hot encoding for one digit * row
        self.args_dim = self.arg_num * self.arg_depth # one hot encoding for argument * num of args
        self.prog_dim = self.prog_embedding_size    # program dimention for output categories
        self.method = method
        self._trainable = True
        if self.method == 'linear':
            self.f_enc = nn.Sequential(
                nn.Linear(self.env_dim + self.args_dim, self.hidden_dim),
                activation.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                activation.ELU(),
                nn.Linear(self.hidden_dim, self.state_dim)
            )
        elif self.method == 'maxout':
            self.nb_features:int = 4
            # self.layers = nn.ModuleList([nn.Linear(self.env_dim + self.prog_dim + self.args_dim, self.hidden_dim) for _ in range(4)])
            self.layers = nn.Linear(self.env_dim + self.args_dim, self.state_dim * self.nb_features)
            # self.layers = nn.ModuleList([nn.Linear(self.env_dim + self.args_dim, self.hidden_dim) for _ in range(self.nb_features)])

    def forward(self, feature):
        if self.method == 'linear':
            return self.f_enc(feature)
        elif self.method == 'maxout':
            # x = [out(feature.clone()) for out in self.layers]
            x = self.layers(feature)
            output, _ = torch.max(x.view(self.nb_features, -1), dim=0)
            return output
    
    @property
    def trainable(self):
        return self._trainable
    
    @trainable.setter
    def trainable(self, flag:bool):
        for param in self.parameters():
            param.requires_grad = flag

# TODO: add embedding
"""
Build the Program Embedding (M_prog) that takes in a specific Program ID (prg_in), and
returns the respective Program Embedding.

Reference: Reed, de Freitas [4]
"""
program_embedding = nn.Embedding(config.prog_num, config.prog_embedding_size).to(config.device) #CONFIG['PROG_EMBEDDING_SIZE'])