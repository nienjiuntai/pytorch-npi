import torch.nn as nn
from torch import optim, tensor
from tqdm import tqdm
from model.npi import NPI
import torch.nn.functional as F
from tasks.add.core import StateEncoder, program_embedding
from tasks.add.config import config
from tasks.add.env import AdditionEnv
import numpy as np
import torch
from random import random

def npi_criteria(term_out:torch.Tensor, term:torch.Tensor, prog_out:torch.Tensor, prog:torch.Tensor, args_out:torch.Tensor, args, weights:list=[1] * 4):
    assert(len(weights) == 4)
    total_losses = F.binary_cross_entropy(term_out, term) * weights[0]
    total_losses += F.nll_loss(prog_out, prog.view(-1)) * weights[1]
    for arg_out, arg, weight in zip(args_out, args, weights[2:]):
        total_losses += F.nll_loss(arg_out, torch.argmax(arg).view(-1)) * weight
    return total_losses

def encoder_criteria(sum, sum_target, carry, carry_target):
    sum_loss = F.nll_loss(sum, sum_target.view(-1))
    carry_loss = F.nll_loss(carry, carry_target.view(-1))
    return sum_loss + carry_loss

class EncodeTrainer(nn.Module):
    '''
    An encoder-decoder for tranning StateEncoder used by NPI
    '''
    def __init__(self, hidden_dim, state_dim, batch_size, output_dim):
        super(EncodeTrainer, self).__init__()
        self.encoder = StateEncoder(hidden_dim, state_dim, batch_size)
        self.sum = nn.Linear(state_dim, output_dim)
        self.carry = nn.Linear(state_dim, output_dim)

    def forward(self, env, args):
        feature = torch.cat([env, args])
        z_state = self.encoder(feature)
        sum = F.log_softmax(self.sum(z_state.clone()).view(1, -1), dim=1)
        carry = F.log_softmax(self.carry(z_state.clone()).view(1, -1), dim=1)
        return sum, carry
        
def train_f_enc(steplists, epochs:int=100, batch_size:int=1):
    env_row, env_col, env_depth = config.env_shape
    arg_num, arg_depth = config.arg_shape
    hidden_dim, state_dim = 100, 128
    trainer = EncodeTrainer(hidden_dim, state_dim, batch_size, env_depth).to(config.device)
    optimizer = optim.Adam(trainer.parameters())
    for epoch in range(epochs):
        total_loss:list = []
        for idx, steps in enumerate(steplists):
            question, trace = steps['question'], steps['trace']
            prev = None
            for step in trace:    # get batched step list
                optimizer.zero_grad()
                env, (pgid, args), _, term = step  # env, args in, args out, terminate
                addend, augend, carry, _ = np.argmax(env, axis=1)
                sadd = addend + augend + carry
                if addend == augend == carry == (arg_depth - 1):
                    sum = carry = arg_depth - 1
                else:
                    sum = sadd % 10
                    carry = int(sadd / 10)
                if prev == (addend, augend, carry):
                    continue
                prev = (addend, augend, carry)
                sum_out, carry_out = trainer(tensor(env).flatten().type(torch.FloatTensor).to(config.device), tensor(args).flatten().type(torch.FloatTensor).to(config.device))
                loss = encoder_criteria(sum_out, tensor(sum).to(config.device), carry_out, tensor(carry).to(config.device))
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
        avg_loss:float = np.average(total_loss)
        print('epoch: {}, avg loss: {:2f}'.format(epoch + 1, avg_loss))
        if avg_loss < 1e-6:
            break
    torch.save(trainer.encoder.state_dict(), f'{config.basedir}/weights/f_enc.weights')

def test_question(question:list, npi:NPI)->int:
    env = AdditionEnv()
    addend, augend = question
    add_program:dict = {
        'pgid': 2,
        'args': []
    }
    wc:int = 0
    correct:int = 0
    npi.reset()
    with torch.no_grad():
        env.setup(addend, augend)
        # run npi algorithm
        npi.step(env, add_program['pgid'], add_program['args'])
        # get environment observation
        return env.result

def train(npi:NPI, optimizer, steplists:list, epochs:int=100, skip_correct:bool=False):
    arg_num, arg_depth = config.arg_shape
    npi.train().to(config.device)
    for epoch in range(epochs):
        # initialize corrent / wrong count
        losses:list = []
        np.random.shuffle(steplists)
        correct = wc = 0
        loop = tqdm(steplists, postfix='loss: {loss} correct: {correct} wrong: {wrong}')
        for idx, step in enumerate(loop):
            question, trace = step['question'], step['trace']
            res = test_question(question, npi)
            if res == np.sum(question):
                correct += 1
            else:
                wc += 1
                if skip_correct:
                    loop.set_description('epoch: {}'.format(epoch + 1))
                    loop.set_postfix(loss=np.average(losses), correct=correct, worng=wc)
                    continue
            with open('log.txt', 'a') as f:
                f.write('epoch: {:>3} idx: {:>3} {:>4} + {:>4} = {:>4}\n'.format(epoch + 1, idx, question[0], question[1], res))

            npi.reset()
            for env, (pgid, args), (pgid_out, args_out), term_out in trace:
                optimizer.zero_grad()
                weights = [1] + [1 if 0 <= pgid < 6 else 1e-10] + [1e-10 if np.argmax(arg) == (arg_depth - 1) else 1 for arg in args_out]
                # get environment observation
                term_pred, pgid_pred, args_pred = npi(tensor(env).flatten().type(torch.FloatTensor).to(config.device), tensor(pgid).to(config.device), tensor(args).flatten().type(torch.FloatTensor).to(config.device))
                total_loss = npi_criteria(term_pred, tensor(term_out).type(torch.FloatTensor).to(config.device), pgid_pred, tensor(pgid_out).to(config.device), args_pred, tensor(args_out).type(torch.FloatTensor).to(config.device), weights)
                total_loss.backward()
                optimizer.step()
                losses.append(total_loss.item())
            loop.set_description('epoch: {}'.format(epoch + 1))
            # loop.set_description(epoch=epoch + 1)
            loop.set_postfix(loss=np.average(losses), correct=correct, worng=wc)
        with open('log.txt', 'a') as f:
            f.write('-' * 20 + '\n')
        loop.close()
        npi.save()
        if wc == 0:
            return True

def train_npi(steplists, epochs:int=100, batch_size:int=1, pretrained_encoder_weights:str=None):
    state_encoder = StateEncoder().to(config.device)
    warm_up:list = list(filter(lambda question: 0 <= question['question'][0] < 100 and 0 <= question['question'][1] < 100, steplists)) 
    if not pretrained_encoder_weights:
        print('start trainning f_enc model')
        train_f_enc(warm_up)
        pretrained_encoder_weights=f'{config.basedir}/weights/f_enc.weights'
    state_encoder.load_state_dict(torch.load(pretrained_encoder_weights))
    for param in state_encoder.parameters():
        param.requires_grad = False
        
    npi = NPI(state_encoder).to(config.device)
    optimizer = optim.Adam(npi.parameters(), lr=1e-4)

    # warm up with single digit add
    for _ in range(10):
        if train(npi, optimizer, warm_up, epochs=100):
            break
    train(npi, optimizer, steplists, epochs=100, skip_correct=False)
    

if __name__ == '__main__':
    import pickle
    with open(f'{config.basedir}/data/train.pik', 'rb') as data:
        steps = pickle.load(data)
    # train_npi(steps)
    train_npi(steps, pretrained_encoder_weights=f'{config.basedir}/weights/f_enc.weights')

