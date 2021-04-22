from random import random
import numpy as np
from tasks.add.config import CONFIG, PROGRAMS, config
import pickle

class AdditionEnv():
    PGID:list = [idx for idx, _ in enumerate(PROGRAMS)]
    def __init__(self):
        self.env_row, self.env_col, self.env_depth = config.env_shape
        self.arg_num, self.arg_depth = config.arg_shape
        self.prog_num, self.prog_key_size, self.prog_embedding_size = config.prog_num, config.prog_key_size, config.prog_embedding_size
        
        self.addend_ptr, self.augend_ptr, self.carry_ptr, self.out_ptr = [(x, -1) for x in range(self.env_row)]
        self.ptrs:list = [self.addend_ptr, self.augend_ptr, self.carry_ptr, self.out_ptr]
        # self.scratchpad:np.array = np.zeros((self.env_row, self.env_col), dtype=np.int8)
        self.scratchpad:np.array = None

    def __getitem__(self, item):
        assert(self.scratchpad is not None)
        return self.scratchpad[item]

    def __setitem__(self, key, value):
        assert(self.scratchpad is not None)
        self.scratchpad[key] = value

    def get_observation(self):
        return [self.to_one_hot(self[row, self.ptrs[row][1]], self.env_depth) for row in range(self.env_row)]

    def to_one_hot(self, x, dim:int, dtype=np.int8):
        one_hot = np.zeros((dim,), dtype=dtype)
        if 0 <= x < dim:
            one_hot[x] = 1
        return one_hot

    def setup(self, addend:int, augend:int):
        '''
        Setup question, maximum 10-digits

        Args:
            addend (int): number
            augend (int): number
        '''
        self.reset()
        numbers:list = [str(addend), str(augend)]
        self.env_col = len(max(numbers, key=len)) + 2   # one for extra carry and one for termination
        self.scratchpad:np.array = np.zeros((self.env_row, self.env_col), dtype=np.int8)
        # set first column to np.nan
        self.scratchpad[:,0] = self.arg_depth - 1
        for row, number in enumerate(numbers):
            for idx, digit in enumerate(number[::-1], 1):
                self[row, -idx] = digit

    def reset(self):
        self.scratchpad:np.array = None
        self.addend_ptr, self.augend_ptr, self.carray_ptr, self.out_ptr = self.ptrs = [(idx, -1) for idx in range(self.env_row)]


    @property
    def result(self):
        return int(''.join([str(i) for i in self.scratchpad[-1][1:]]))

    def exec(self, prog, args):
        # print('exec', prog, args)
        if prog == 0:
            ptr, lr = args
            if 0 <= ptr < self.env_row:
                # print('mvptr', ptr, lr)
                pos:int = self.ptrs[ptr][1] + (-1 if lr == 0 else 1)
                self.ptrs[ptr] = (ptr, (pos % self.env_col) - self.env_col)
                self.addend_ptr, self.augend_ptr, self.carray_ptr, self.out_ptr = self.ptrs
        elif prog == 1:
            ptr, val = args
            if 0 <= ptr < len(self.ptrs) and 0 <= val < self.env_depth:
                if ptr == 0:
                    self[self.out_ptr] = val
                elif ptr == 1:
                    self[self.carray_ptr] = val
    
    def display_step(self, prog, args, term):
        if prog == None:
            return
        print(['MVPTR', 'WRITE', 'ADD', 'SADD', 'CARRY', 'LSHIFT', 'EOP'][prog], args, 'alpha:', term)

    def display(self, prog, args, term):
        row_name:list = ['addend', 'augend', 'carry ', 'output']
        # self.display_step(prog, args, term)
        if prog not in [0, 1, -1]:
            return
        for name, row in zip(row_name, self.scratchpad):
            print(name, *row, sep=' ')
        print(self.ptrs)
    
class NPIExpert():
    WRITE_OUT, CARRY_OUT = 0, 1
    LEFT, RIGHT = 0, 1
    MVPTR, WRITE, ADD, SADD, CARRY, LSHIFT, EOP = [idx for idx, _ in enumerate(PROGRAMS)]
    ADDEND_PTR, AUGEND_PTR, CARRY_PTR, OUT_PTR = range(4)

    def __init__(self, addend:int, augend:int):
        self.term_thresh:float = 0.9
        self.trace:list = []
        self.stack:list = None
        self.stack_queue:list = []
        self.env:AdditionEnv = AdditionEnv()
        self.env.setup(addend, augend)
        self.prog:list = [self.mvptr, self.write, self.add, self.sadd, self.carry, self.lshift, self.eop]
        # self.trace.append((self.env.get_observation(), self.ADD, [], 0))
        self.npi(self.ADD, [])
        assert(self.env.result == addend + augend)

    def eop(self, env:list, pgid:int, args:list):
        '''
        End of program
        '''
        return None

    def mvptr(self, env:list, pgid:int, args:list):
        '''
        primitive action, always return None
        '''
        return None

    def write(self, env:list, pgid:int, args:list):
        '''
        primitive action, always return None
        '''
        return None

    def add(self, env:list, pgid:int, args:list):
        '''
        Add is combination of single digit add and shift all ptr to left
        '''
        addend, augend, carry, out = np.argmax(env, axis=1) # TODO arg max will fail when it's an zeros list
        if addend == augend == carry == (self.env.arg_depth - 1):
            return None
        ret = []
        ret.append((0, self.SADD, []))
        ret.append((0, self.LSHIFT, []))
        return ret
        
    def sadd(self, env:list, pgid:int, args:list):
        '''
        SADD is combination of write to output and carry 
        '''
        addend, augend, carry, out = np.argmax(env, axis=1)
        result = addend + augend + carry
        ret = []
        ret.append((0, self.WRITE, [self.WRITE_OUT, result % 10]))
        if result > 9:
            ret.append((0, self.CARRY, []))
        ret[-1] = (1, ret[-1][1], ret[-1][2])
        return ret

    def carry(self, env:list, pgid:int, args:list):
        '''
        CARRY is combination of move carry ptr write result and shift back
        '''
        ret = []
        ret.append((0, self.MVPTR, [self.CARRY_PTR, self.LEFT]))
        ret.append((0, self.WRITE, [self.CARRY_OUT, 1]))
        ret.append((1, self.MVPTR, [self.CARRY_PTR, self.RIGHT]))
        return ret

    def lshift(self, env:list, pgid:int, args:list):
        '''
        LSHIFT is combination of move all ptr left
        '''
        ret = []
        ret.append((0, self.MVPTR, [self.ADDEND_PTR, self.LEFT]))
        ret.append((0, self.MVPTR, [self.AUGEND_PTR, self.LEFT]))
        ret.append((0, self.MVPTR, [self.CARRY_PTR, self.LEFT]))
        ret.append((1, self.MVPTR, [self.OUT_PTR, self.LEFT]))
        return ret

    def save_ctx(self):
        self.stack_queue.append(self.stack or [])
        self.stack:list = None

    def restore_ctx(self):
        self.stack  = self.stack_queue.pop()
    
    def step(self, env:list, pgid:int, args:list):
        if not self.stack:
            self.stack = self.prog[pgid](env, pgid, args)
        if self.stack:
            return self.stack.pop(0)
        else:
            return (1, self.EOP, [])

    def npi(self, pgid:int, args:list):
        self.save_ctx()
        term_prob:int = 0
        while term_prob < self.term_thresh:
            env = self.env.get_observation()
            term_prob, pgid_out, args_out = self.step(env, pgid, args)
            self.trace.append((env, (pgid, self.args_encode(args)), (pgid_out, self.args_encode(args_out)), term_prob))
            # self.env.display_step(pgid, args, term_prob)
            if pgid in [0, 1]:
                self.env.exec(pgid, args)
                # self.env.display(pgid, args, term_prob)
            else:
                if pgid_out != self.EOP:   # is't not EOP
                    self.npi(pgid_out, args_out)
        self.restore_ctx()
    
    def args_encode(self, args:list):
        dim:int = self.env.arg_depth
        encoded_args:list = []
        null_arg = one_hot = np.zeros((dim,), dtype=np.int8)
        null_arg[dim - 1] = 1
        for arg in args:
            one_hot = np.zeros((dim,), dtype=np.int8)
            if 0 <= arg < dim:
                one_hot[arg] = 1
            encoded_args.append(one_hot)
        encoded_args += [null_arg] * (self.env.arg_num - len(encoded_args))
        return encoded_args

def create_questions(num:int, max:int=10000)->list:
    '''
    Randomly create `num` of questions

    Args:
        num (int): num of questions
        `option` max (int): maximum of input number 
    '''
    questions:list = []
    for _ in range(num):
        questions.append({
            'addend': int(random() * max),
            'augend': int(random() * max)
        })
    return questions

def create_basic_questions(num:int, max:int=10000):
    questions:list = []
    for addend in range(10):
        for augend in range(10):
            questions.append({
                'addend': addend,
                'augend': augend
            })
    questions.extend(create_questions(100, 100))
    questions.extend(create_questions(100, 1000))
    questions.extend(create_questions(num, max))
    return questions

if __name__ == '__main__':
    questions = create_basic_questions(1000)
    data:list = []
    for q in questions:
        exp = NPIExpert(q['addend'], q['augend'])
        data.append({
            'question': [q['addend'], q['augend']],
            'trace': exp.trace
        })
    import os, pickle
    prefix:str = 'train'
    with open('{}/data/{}.pik'.format(config.basedir, prefix), 'wb') as f:
        pickle.dump(data, f)
