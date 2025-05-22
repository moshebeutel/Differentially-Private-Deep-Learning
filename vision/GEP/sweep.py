import argparse
import logging
import subprocess
from functools import partial
import wandb
from utils import set_seed
from main import main


def sweep_train(sweep_id, args, train_fn, config=None):
    with wandb.init(config=config):
        config = wandb.config
        config.update({'sweep_id': sweep_id})
        set_seed(config.seed)

        for k, v in config.items():
            if k in args:
                setattr(args, k, v)

        wandb.run.name = '_'.join([f'{k}_{v}' for k, v in config.items()])
        train_fn(args)


def init_sweep(config):
    sweep_id = wandb.sweep(sweep=config, project="GEP")
    return sweep_id


def start_sweep(sweep_id, f_sweep):
    wandb.agent(sweep_id=sweep_id, function=f_sweep)


def sweep(sweep_config, args, train_fn):
    # logger = logging.getLogger(args.log_name)
    # logger.info(f'sweep {sweep_config}')
    sweep_id = init_sweep(sweep_config)
    f_sweep = partial(sweep_train, sweep_id=sweep_id, args=args, train_fn=train_fn)
    wandb.agent(sweep_id=sweep_id, function=f_sweep)
    start_sweep(sweep_id, f_sweep)


parser = argparse.ArgumentParser(description='Sweep Differentially Private learning with GEP')
## general arguments
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--filters', default=4, type=int, help='TinyCifarNet num of filters')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--batchsize', default=500, type=int, help='batch size')
parser.add_argument('--n_epoch', default=200, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
## arguments for learning with differential privacy
# parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--override', '-o', action='store_true', help='zero sigma')
parser.add_argument('--perp', '-v', action='store_true', help='add perpendicular vector')
parser.add_argument('--eps', default=8., choices=[8., 3., 1.], type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
parser.add_argument('--public_perp_split', default=0.75, type=float, help='split public data for perp train')

parser.add_argument('--rgp', action='store_true', help='use residual gradient perturbation or not')
parser.add_argument('--clip0', default=1., type=float, help='clipping threshold for gradient embedding')
parser.add_argument('--clip1', default=2., type=float, help='clipping threshold for residual gradients')
parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
parser.add_argument('--num_groups', default=1, type=int, help='number of parameters groups')
parser.add_argument('--num_bases', default=1000, type=int, help='dimension of anchor subspace')
parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
parser.add_argument('--aux_dataset', default='imagenet', type=str,
                    help='name of the public dataset, [cifar10, cifar100, imagenet]')
parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')

args = parser.parse_args()

args.private = True
perp = args.perp
override = args.override

sweep_configuration = {
    "name": f"GEP_SEED_2_TINY_{'Perp' if perp else 'NoPerp'}",
    # "name": f"GEP_SEED_2_TINY_COMPARE",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "test_acc"},
    "parameters": {
        "lr": {"values": [0.00001]},
        "num_groups": {"values": [1]},
        "seed": {"values": [2]},
        "clip0": {"values": [30.0]},
        "eps": {"values": [8.0]},
        "public_perp_split": {"values": [0.95]},
        "momentum": {"values": [0.95]},
        "filters": {"values": [16]},
        "n_epoch": {"values": [30]},
        "num_bases": {"values": [800]},
        "aux_data_size": {"values": [2000]},
        "batchsize": {"values": [128]},
        "perp": {"values": [perp]},
        "override": {"values": [False]}
    },
}


def namespace_to_cmd_args(namespace):
    args_list = []
    for key, value in vars(namespace).items():
        if isinstance(value, bool):
            if value:
                args_list.append(f'--{key}')
        else:
            args_list.extend([f'--{key}', str(value)])
    return args_list


wandb.login()

sweep(sweep_config=sweep_configuration, args=args,
      train_fn=main)
