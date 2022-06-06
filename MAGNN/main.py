from run_ACM import run_model_ACM
from run_IMDB import run_model_IMDB
from run_DBLP import run_model_DBLP
import argparse
import os

def main(args):
    if args.dataset == 'ACM':
        f1s = run_model_ACM(args)
    elif args.dataset == 'IMDB':
        f1s = run_model_IMDB(args)
    elif args.dataset == 'DBLP':
        f1s = run_model_DBLP(args)
    return f1s

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing')
    ap.add_argument('--dataset', type=str, default='ACM', help='Name of dataset.')
    ap.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    ap.add_argument('--save_path', default='saved_model/', help='folder path of saved model.')
    ap.add_argument('--feats_type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn_type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    ap.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.')
    ap.add_argument('-wd', '--weight_decay', type=float, default=0.001, help='weight decay in Adam optimizer.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    ap.add_argument('--batch_size', type=int, default=8, help='Batch size. Default is 8, used by DBLP dataset only')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100, used by DBLP dataset only.')
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    ap.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    ap.add_argument('--log_step', type=int, default=5, help='training log step.')
    
    args = ap.parse_args()
    f1s = main(args)

