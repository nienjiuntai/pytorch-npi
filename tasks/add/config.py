'''
Configuration for NPI task addition
'''
import os
import torch

CONFIG = {
    'ENV_ROW': 4,       # Addend, Augend, Carry, Output
    'ENV_COL': 10,      # Max-digit for addition task
    'ENV_DEPTH': 11,    # Number of elements per cell(0-9 and non), one-hot encoding
    'ARG_NUM': 2,       # Max number of arguments
    'ARG_DEPTH': 11,    # Number of elements per argument cell(0-9 and non), one-hot encoding
    'DEFUALT_ARG_VAL': 10,  # Default argument value
    'PROG_NUM': 7,      # Number of programs
    'PROG_KEY_SIZE': 5, # Size of program keys
    'PROG_EMBEDDING_SIZE': 7   # Size of program embeddings
}
PROGRAMS = [
    ('MVPTR', 4, 2),    # Move pointer left or right
    ('WRITE', 2, 10),   # Given Carry/Out pointer write digit
    ('ADD',),           # Top-level add
    ('SADD',),          # Single-digit add
    ('CARRY',),         # Carry operation
    ('LSHIFT',),        # Shifts all pointers left(only apper after single-digit add)
    ('EOP',),           # End of Program
]

class Cfg():
    basedir:str = os.path.realpath(os.path.dirname(__file__))
    env_shape:tuple = (CONFIG['ENV_ROW'], CONFIG['ENV_COL'], CONFIG['ENV_DEPTH'])
    arg_shape:tuple = (CONFIG['ARG_NUM'], CONFIG['ARG_DEPTH'])
    prog_num:int = CONFIG['PROG_NUM']
    prog_key_size:int = CONFIG['PROG_KEY_SIZE']
    prog_embedding_size:int = CONFIG['PROG_EMBEDDING_SIZE']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Cfg()