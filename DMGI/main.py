import os
import numpy as np
np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main(args):
    #args, unknown = parse_args()
    # args.dataset = 'dblp'
    # args.metapaths = 'APA,APTPA,APCPA'
    try:
        os.mkdir(args.save_path)
    except:
        print('model folder exists!')

    if args.embedder == 'DMGI':
        from models import DMGI
        embedder = DMGI(args)
    elif args.embedder == 'DGI':
        from models import DGI
        embedder = DGI(args)

    embedder.training()

if __name__ == '__main__':
    # input arguments
    parser = argparse.ArgumentParser(description='DMGI')

    parser.add_argument('--embedder', nargs='?', default='DMGI')
    parser.add_argument('--dataset', nargs='?', default='imdb')
    parser.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    parser.add_argument('--save_path', default='saved_model/', help='folder path of saved model.')
    parser.add_argument('--metapaths', nargs='?', default='MAM,MDM')
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hid_units', type=int, default=64)
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--reg_coef', type=float, default=0.001)
    parser.add_argument('--sup_coef', type=float, default=0.1)
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--patience2', type=int, default=100)
    parser.add_argument('--nheads', type=int, default=1)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=False)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=False)
    parser.add_argument('--num_iter', type=int, default=500, help='Max epochs to run for downstream task.')
    parser.add_argument("--num_test", type=int, default=30, help='num of runs to test accuracy.')
    args = parser.parse_args()
    main(args)
