import numpy as np
import torch
import torch.nn as nn
from models import GCN, GAT
from utils import norm_adj, remove_edge_pts
#from collections import Counter
import time
from sklearn import metrics
import argparse
import pickle
#import sys

def train(model, x_dict, y, g, node_type_order, train_mask, val_mask, test_mask, args):
    device = args.device
    model.to(device)
    for node_type in x_dict:
        x_dict[node_type] = x_dict[node_type].to(device)
    y = y.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_f1 = 0
    test_f1 = 0
    count = 0
    
    tic = time.time()
    t0 = time.time()
    training_time = 0
    for epoch in range(1, args.num_iter + 1):  
        t1 = time.time()
        model.train()
        y_pred = model(x_dict, g, node_type_order)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dict, g, node_type_order)
            y_pred_val = y_pred[val_mask]
            y_pred_test = y_pred[test_mask]
            pred_val = y_pred_val.argmax(1)
            pred_test = y_pred_test.argmax(1)
            Val_f1 = metrics.f1_score(y[val_mask].cpu().numpy(), pred_val.cpu().numpy(), average=args.average)
            Test_f1 = metrics.f1_score(y[test_mask].cpu().numpy(), pred_test.cpu().numpy(), average=args.average)
        # Save the best validation f1 and the corresponding test f1.
        if Val_f1 > best_val_f1:
            best_val_f1 = Val_f1
            test_f1 = Test_f1
            count = 0
        elif count >= args.patience:
            break
        else:
            count += 1
        if epoch % args.log_step == 0:
            print('Epoch {}, time elapsed: {:.3f}, loss: {:.3f}, val f1: {:.3f} (best {:.3f}), test f1: {:.3f} (test_f1 at best val: {:.3f})'.
                  format(epoch, time.time() - tic, loss.detach().item(), Val_f1, best_val_f1, Test_f1, test_f1))
            tic = time.time()
    print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 ({}) = {:.3f}, test_f1 ({}) = {:.3f}'.
          format(time.time() - t0, training_time/epoch, args.average, best_val_f1, args.average, test_f1))
    #model = model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_f1


def main(args):
    load_data_name = args.data + '.pkl'
    with open(args.data_path + load_data_name, 'rb') as fp:
        data = pickle.load(fp)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_out = data['labels'].max().item() + 1
    # transfer to homogeneous graph
    x_dict, edge_index_dict, y = data['x_dict'], data['edge_index_dict'], data['labels']
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']
    if args.data == 'ACM':
        # nodes aranged in the order of paper, author, subject
        node_type_order = ['paper', 'author', 'subject']
        # combine edge_index_dict to one edge_index in homogeneous graph
        A_pa = edge_index_dict[('paper', 'to', 'author')].clone()
        A_ps = edge_index_dict[('paper', 'to', 'subject')].clone()
        s1, t1 = A_pa
        s2, t2 = A_ps
        t1 += x_dict['paper'].size(0)
        t2 += x_dict['paper'].size(0) + x_dict['author'].size(0)
        s, t = torch.cat((s1, s2, t1, t2)), torch.cat((t1, t2, s1, s2)) # consider undirected
        edge_index = torch.stack([s, t])
    elif args.data == 'DBLP':
        # nodes aranged in the order of author, paper, conference
        node_type_order = ['author', 'paper', 'term', 'conference']
        # combine edge_index_dict to one edge_index in homogeneous graph
        A_ap = edge_index_dict[('author', 'to', 'paper')].clone()
        A_pt = edge_index_dict[('paper', 'to', 'term')].clone()
        A_pc = edge_index_dict[('paper', 'to', 'conference')].clone()
        s1, t1 = A_ap
        s2, t2 = A_pt
        s3, t3 = A_pc
        t1 += x_dict['author'].size(0)
        s2 += x_dict['author'].size(0)
        t2 += x_dict['author'].size(0) + x_dict['paper'].size(0)
        s3 += x_dict['author'].size(0)
        t3 += x_dict['author'].size(0) + x_dict['paper'].size(0) + x_dict['term'].size(0)
        s, t = torch.cat((s1, s2, s3, t1, t2, t3)), torch.cat((t1, t2, t3, s1, s2, s3)) # consider undirected
        edge_index = torch.stack([s, t])
    elif args.data == 'IMDB':
        # nodes aranged in the order of movie, director, actor
        node_type_order = ['movie', 'director', 'actor']
        # combine edge_index_dict to one edge_index in homogeneous graph
        A_md = edge_index_dict[(('movie', 'to', 'director'))].clone()
        A_ma = edge_index_dict[('movie', 'to', 'actor')].clone()
        s1, t1 = A_md
        s2, t2 = A_ma
        t1 += x_dict['movie'].size(0)
        t2 += x_dict['movie'].size(0) + x_dict['director'].size(0)
        s, t = torch.cat((s1, s2, t1, t2)), torch.cat((t1, t2, s1, s2)) # consider undirected
        edge_index = torch.stack([s, t])
    
    n_in_dict = {}
    N = 0 # total node number
    for node_type in data['x_dict']:
        N += data['x_dict'][node_type].size(0)
        n_in_dict[node_type] = data['x_dict'][node_type].size(-1)
                
    f1s = []
    g = {}
    for i in range(args.num_test):
        if args.model == 'GCN':
            model = GCN(n_in_dict, args.n_hid, n_out, args.dropout, args.hop)
            A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (N, N))
            g['A'] = norm_adj(A, self_loop=True).to(args.device)
            # if i == 0:
            #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
        elif args.model == 'GAT':
            model = GAT(n_in_dict, args.n_hid, n_out, args.dropout, args.dropout2, args.hop, args.num_heads, args.num_out_heads)
            g['edge_index'] = edge_index.to(args.device)
            # if i == 0:
            #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
        f1 = train(model, x_dict, y, g, node_type_order, train_mask, val_mask, test_mask, args)
        f1s.append(f1) 
    f1s = np.array(f1s)
    print('test {}-f1 (mean, std): '.format(args.average), f1s.mean(), f1s.std())
    f1 = remove_edge_pts(f1s, pct=args.filter_pct)
    print('test {}-f1 (mean, std) after filter: '.format(args.average), f1.mean(), f1.std())
    return f1s

if __name__ == "__main__":
    """
        Heterogeneous GNN Model Hyperparameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DBLP', help='Name of dataset.')
    parser.add_argument('--data_path', default='data/preprocessed/', help='folder path of saved preprocessed data.')
    parser.add_argument('--model', default='GAT', help='Heterogeneous GNN model.')
    parser.add_argument('--n_hid', type=int, default=64, help='num of hidden features')
    parser.add_argument('--num_heads', type=int, default=1, help='num heads for attention layer')
    parser.add_argument('--num_out_heads', type=int, default=1, help='num heads for GAT output layer')
    parser.add_argument('--dropout', type=float, default=0, help='MLP1 dropout')
    parser.add_argument('--dropout2', type=float, default=0, help='Intermediate layer dropout')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00005, help='weight decay in Adam optimizer.')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience.')
    parser.add_argument('--num_iter', type=int, default=500, help='Max epochs to run.')
    parser.add_argument("--num_test", type=int, default=30, help='num of runs to test accuracy.')
    parser.add_argument("--hop", type=int, default=2, help='hop of GNN models.')
    parser.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    parser.add_argument('--log_step', type=int, default=1000, help='training log step.')
    parser.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    args = parser.parse_args()
    f1s = main(args)

