from torch.nn.modules.sparse import Embedding
from tasks.add.env import AdditionEnv
from tasks.add.core import StateEncoder, program_embedding
import torch
from torch import optim, tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import activation
from tasks.add.config import config
import numpy as np

class NPITerm(nn.Module):
    def __init__(self, input_dim:int):
        super(NPITerm, self).__init__()
        self._fcn = nn.Linear(in_features=input_dim, out_features=1)  # Dense(1, W_regularizer=l2(0.001))

    def forward(self, x):
        x = self._fcn(x)
        x = torch.sigmoid(x)
        return x

class NPIProg(nn.Module):
    def __init__(self, input_dim:int, prog_key_dim:int, prog_num:int,):
        super(NPIProg, self).__init__()
        
        self._fcn1 = nn.Linear(in_features=input_dim, out_features=prog_key_dim)
        self._fcn2 = nn.Linear(in_features=prog_key_dim, out_features=prog_num)
    def forward(self, x):
        x = self._fcn1(x)
        x = self._fcn2(F.relu_(x))
        x = F.log_softmax(x.view(1, -1), dim=1)
        return x

class NPIArg(nn.Module):
    def __init__(self, input_dim:int, arg_dim:int):
        super(NPIArg, self).__init__()
        self.f_arg = nn.Linear(input_dim, arg_dim)
        
    def forward(self, x):
        x = self.f_arg(x)
        x = F.log_softmax(x.view(1, -1), dim=1)
        return x

class NPICore(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, layers:int=1):
        super(NPICore, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.h_0, self.c_0 = torch.zeros(1, self.hidden_dim).to(config.device), torch.zeros(1, self.hidden_dim).to(config.device)
        self.cell_inpt = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.cells = nn.ModuleList([nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim) for _ in range(layers)])

    def forward(self, x):
        h_x, c_x = self.cell_inpt(x.view(1, -1), (self.h_0, self.c_0))
        for cell in self.cells:
            h_x, c_x = cell(F.relu(h_x), (h_x, c_x))
        self.h_0, self.c_0 = h_x.clone().detach(), c_x.clone().detach()
        return F.relu(h_x)

    def reset_states(self):
        self.h_0, self.c_0 = torch.zeros(1, self.hidden_dim).to(config.device), torch.zeros(1, self.hidden_dim).to(config.device)
        for layer in self.cell_inpt.modules():
            if hasattr(layer, 'reset_states'):  # could use `isinstance` instead to check if it's an RNN layer
                layer.reset_states()
        for cell in self.cells:
            for layer in cell.modules():
                if hasattr(layer, 'reset_states'):  # could use `isinstance` instead to check if it's an RNN layer
                    layer.reset_states()
        

class NPI(nn.Module):
    def __init__(self, state_encoder:StateEncoder, model_path:str=None):
        super(NPI, self).__init__()
        self.steps = 0
        self.max_depth, self.max_steps = 10, 1000

        self.prog_dim, self.prog_key_size, self.prog_num = config.prog_embedding_size, config.prog_key_size, config.prog_num
        
        self.arg_num, self.arg_depth = config.arg_shape
        self.arg_dim = self.arg_num * self.arg_depth

        state_in, prog_in = state_encoder.state_dim, state_encoder.prog_dim
        self.f_enc, self.prog_embedding = state_encoder, program_embedding
        # LSTM configs
        self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_hidden_layer = state_in + prog_in, 256, 1
        self.f_lstm = NPICore(self.lstm_input_dim, self.lstm_hidden_dim, self.lstm_hidden_layer)

        self.f_term = NPITerm(self.lstm_hidden_dim)
        self.f_prog = NPIProg(self.lstm_hidden_dim, self.prog_key_size, self.prog_num)
        self.f_args:list = []
        for _ in range(self.arg_num):
            self.f_args.append(NPIArg(self.lstm_hidden_dim, self.arg_depth))
        # try nn.ModuleDict
        self.f_args = nn.ModuleList(self.f_args)
        
    def forward(self, state, prog, args_embedding):
        prog_embedding = self.prog_embedding(prog)
        # prog_embedding = tensor(to_one_hot(prog, self.prog_dim)).to(config.device)
        z_state = self.f_enc(torch.cat([state, args_embedding]))
        
        inpt = torch.cat([z_state, prog_embedding])
        # out, (h_state, c_state) = self.f_lstm(inpt.view(1, 1, -1))  # find out how to use program embedding
        # # h_state = F.relu(out)
        # h_state = F.relu(h_state[-1])
        h_state = self.f_lstm(inpt.view(1, -1))  # find out how to use program embedding
        # h_state = F.relu(out)
        # h_state = h_state[-1]
        # h_state = out
        t_state = self.f_term(h_state.clone())
        p_state = self.f_prog(h_state.clone())
        args_state:list = []
        for f_arg in self.f_args:
            a_state = f_arg(h_state.clone())
            args_state.append(a_state)
        return t_state.view([]), p_state.view(1, -1), args_state
    
    def reset(self):
        # reset LSTM states
        self.steps:int = 0
        self.f_lstm.reset_states()
        
    def predict(self, env, pgid, args):
        args = [to_one_hot(arg, self.arg_depth) for arg in args]
        args += [to_one_hot(self.arg_depth - 1, self.arg_depth) for _ in range(len(args) + 1, self.arg_num + 1)]   # fill with 10
        term, prog_key, args_out = self(tensor(env).flatten().type(torch.FloatTensor).to(config.device), tensor(pgid).to(config.device), tensor(args).flatten().type(torch.FloatTensor).to(config.device))
        return term.item(), torch.argmax(prog_key).item(), [torch.argmax(arg).item() for arg in args_out]

    def step(self, env:AdditionEnv, pgid:int, args:list, term_thresh:float=0.5, depth:int=0):
        term_prob:float = 0
        if depth > self.max_depth:
            return
        while term_prob < term_thresh:
            self.steps += 1
            if self.steps > self.max_steps:
                return
            env_state = env.get_observation()
            # term_prob, pgid_out, args_out = self(env_state, pgid, args)
            term_prob, pgid_pred, args_pred = self.predict(env_state, pgid, args)
            # env.display_step(pgid, args, term_prob)
            if pgid in [0, 1]:
                env.exec(pgid, args)
                # env.display(pgid, args, term_prob)
            else:
                if pgid_pred != 6:    # here need modify
                    self.step(env, pgid_pred, args_pred, depth=depth+1)
    def save(self, modelpath:str=f'{config.basedir}/weights/npi.weights'):
        torch.save(self.state_dict(), modelpath)
def to_one_hot(x:int, dim:int, dtype=np.int8):
    one_hot = np.zeros((dim,), dtype=dtype)
    if 0 <= x < dim:
        one_hot[x] = 1
    return one_hot
