import yaml, os


class Config(object):
    def __init__(self, config_path:str):
        assert(os.path.exists(config_path))
        with open(config_path, 'r') as f:
            config = yaml.load(config_path)
    
    def _check_keys(self, cfg:dict):
        keys:list = ['env_row', 'env_col', 'env_depth', 'arg_num', 'arg_depth', 'prog_num', 'prog_key_size', 'prog_embedding_size']
        