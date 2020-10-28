import os
import yaml


def get_logdir(root_path):
    return os.path.join(root_path, 'exp{}'.format(len(os.listdir(root_path))+1))


def read_config():
    current = os.path.dirname(__file__)
    cfg_file = os.path.join(current, "../config/cfg.yaml")
    assert os.path.isfile(cfg_file), "no cfg file"
    # print("loaded cfg from", cfg_file)
    with open(cfg_file, 'r')  as f:
        cfg = yaml.safe_load(f)
    return cfg


class Config(object):
    def __init__(self):
        config = read_config()

        # ds info
        dataset_info = config['dataset']
        self.root_folder = dataset_info['root_folder']
        training_info = config['training']
        self.bs = training_info['bs']
        self.lr = training_info['lr']
        self.epoch = training_info['epochs']
        self.warmup_lr = training_info['warmup_lr']
        self.warmup_epoch = training_info['warmup_epoch']



if __name__ == '__main__':
    cfg = Config()
    print(cfg.root_folder+'/train_train.csv')