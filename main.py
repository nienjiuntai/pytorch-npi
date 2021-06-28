from tasks.add.config import config
from tasks.add import env
import argparse, pickle
from trainer import train_npi
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('type', type=str, help='train | test | valid | gen')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--prefix', type=str, default='test')
    return parser.parse_args()

def create_dataset(dataset_size:int, prefix:str, output_path:str=config.basedir):
    questions = env.create_questions(dataset_size)
    data:list = []
    for q in questions:
        exp = env.Expert(q['addend'], q['augend'])
        data.append({
            'question': [q['addend'], q['augend']],
            'trace': exp.solve()
        })
        
    with open('{}/data/{}.pik'.format(output_path, prefix), 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    args = parse_args()
    if args.type == 'gen':
        create_dataset(args.size, args.prefix, config.basedir)
    elif args.type == 'train':
        with open(f'{config.basedir}/data/train.pik', 'rb') as data:
            steps = pickle.load(data)
        train_npi(steps, pretrained_encoder_weights=f'{config.basedir}/weights/f_enc.weights')
    elif args.type == 'test':
        pass
    elif args.type == 'valid':
        pass
    else:
        raise NotImplemented
        