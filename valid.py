from random import random
from model.npi import NPI
from tasks.add.core import StateEncoder
from tasks.add.config import config
from tasks.add.env import AdditionEnv
from tqdm import tqdm
import torch

def valid_npi(questions:list, pretrained_encoder_weights:str, pretrained_npi_weights:str):
    state_encoder = StateEncoder().to(config.device)
    state_encoder.load_state_dict(torch.load(pretrained_encoder_weights))
    npi = NPI(state_encoder, max_depth=20, max_steps=10000).to(config.device)
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
    return correct, wc

if __name__ == '__main__':
    import os, sys
    uuid = sys.argv[1]
    assert(os.path.exists(f'{config.basedir}/outputs/{uuid}'))
    for idx in range(1, 40):
        max = 1e4 ** idx
        q = [[int(random() * max), int(random() * max)]  for _ in range(100)]
        cc, wc = valid_npi(q, f'{config.basedir}/weights/f_enc.weights', f'{config.basedir}/outputs/{uuid}/weights/npi.weights')
        with open(f'{config.basedir}/outputs/{uuid}/accuracy.csv', 'a') as f:
            f.write('{}, {}, {}\n'.format(idx * 4, cc, wc))

