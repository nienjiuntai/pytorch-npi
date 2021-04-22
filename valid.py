from random import random
from model.npi import NPI
from tasks.add.core import StateEncoder
from tasks.add.config import config
from tasks.add.env import AdditionEnv
from tqdm import tqdm
import torch

def valid_npi(questions:list, pretrained_encoder_weights:str, pretrained_npi_weights:str):
    arg_num, arg_depth = config.arg_shape
    state_encoder = StateEncoder().to(config.device)
    state_encoder.load_state_dict(torch.load(pretrained_encoder_weights))
    npi = NPI(state_encoder).to(config.device)
    npi.load_state_dict(torch.load(pretrained_npi_weights))
    env = AdditionEnv()
    add_program:dict = {
        'pgid': 2,
        'args': []
    }
    wc:int = 0
    correct:int = 0
    npi.eval().to(config.device)
    loop = tqdm(questions, postfix='correct: {correct} wrong: {wrong}')
    loop
    for addend, augend in loop:
        npi.reset()
        with torch.no_grad():
            env.setup(addend, augend)
            # run npi algorithm
            npi.step(env, add_program['pgid'], add_program['args'])
        if env.result != (addend + augend):
            wc += 1
            loop.write('{:>5} + {:>5} = {:>5}'.format(addend, augend, env.result))
        else:
            correct +=1
        loop.set_postfix(correct=correct, wrong=wc)

if __name__ == '__main__':
    max = 100000000
    q = [[int(random() * max), int(random() * max)]  for _ in range(1000)]
    valid_npi(q, f'{config.basedir}/weights/f_enc.weights', f'{config.basedir}/weights/npi.weights')

