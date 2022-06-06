import torch
import numpy as np
import torch.nn as nn
from run import SimpleHGN
import pickle
import argparse
import time
from sklearn import metrics
import gc

def evaluate_f1(y_val_pred, y_val_true, y_test_pred, y_test_true, average='macro'):
    val_f1 = metrics.f1_score(y_val_true.cpu().numpy(), y_val_pred.argmax(1).cpu().numpy(), average=average)
    test_f1 = metrics.f1_score(y_test_true.cpu().numpy(), y_test_pred.argmax(1).cpu().numpy(), average=average)
    return val_f1, test_f1

def remove_edge_pts(accs, pct=0.1):
    accs = sorted(list(accs))
    N = len(accs)
    M = int(N * pct)
    accs = np.array(accs[M:N-M])
    return accs
    
def train(node_features, edges, labels, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_nodes = edges[0].shape[0]
    A = []
    
    for i,edge in enumerate(edges):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[0], edge.nonzero()[1]))).type(torch.int64).to(device)
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.float32).to(device)
        A.append((edge_tmp,value_tmp))
    edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.int64).to(device)
    value_tmp = torch.ones(num_nodes).type(torch.float32).to(device)
    A.append((edge_tmp,value_tmp))

    node_features = torch.from_numpy(node_features).type(torch.float32).to(device)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.int64).to(device)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.int64).to(device)

    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.int64).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.int64).to(device)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.int64).to(device)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.int64).to(device)
    num_classes = torch.max(train_target).item()+1
    
    heads = [args.num_heads] * args.hop + [1]
    model = SimpleHGN(in_dims=node_features.shape[1], num_classes=num_classes, edge_dim=args.edge_dim, num_etypes=len(A), 
                      num_hidden=args.n_hid, num_layers=args.hop, heads=heads, feat_drop=args.dropout, 
                      attn_drop=args.dropout2, negative_slope=args.negative_slope, residual=True, alpha=args.beta)
    print('#Parameters:', sum(p.numel() for p in model.parameters()))
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    best_val_f1 = 0
    best_test_f1 = 0
    count = 0
    tic = time.time()
    t0 = time.time()
    training_time = 0
    for epoch in range(1, args.epoch + 1):
        t1 = time.time()
        model.train()
        y_pred = model(node_features, A)
        loss = criterion(y_pred[train_node], train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        model.eval()
        # Valid
        with torch.no_grad():
            y_pred = model(node_features, A)
            y_valid, y_test = y_pred[valid_node].detach(), y_pred[test_node].detach()
            val_f1, test_f1 = evaluate_f1(y_valid, valid_target, y_test, test_target, average=args.average)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_test_f1 = test_f1
                count = 0
            elif count >= args.patience:
                break
            else:
                count += 1
            if epoch % args.log_step == 0:
                print('Epoch {}, time elapsed: {:.3f}, loss: {:.3f}, val macro_f1: {:.3f} (best {:.3f}), test macro_f1: {:.3f} (test_f1 at best val: {:.3f})'.
                      format(epoch, time.time() - tic, loss.detach().item(), val_f1, best_val_f1, test_f1, best_test_f1))
                tic = time.time()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 (macro) = {:.3f}, best_test_f1 (macro) = {:.3f}'.
          format(time.time() - t0, training_time/epoch, best_val_f1, best_test_f1))
    return best_test_f1
    
    
def main(args):
    with open(args.data_path+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open(args.data_path+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open(args.data_path+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)

    test_f1s = []
    for cnt in range(args.num_test):
        best_test_f1 = train(node_features, edges, labels, args)
        test_f1s.append(best_test_f1)
    test_f1s = np.array(test_f1s)
    print('test macro-f1 (mean, std): ', test_f1s.mean(), test_f1s.std())
    test_f1s_2 = remove_edge_pts(test_f1s, pct=args.filter_pct)
    print('test macro-f1 (mean, std) after filter: ', test_f1s_2.mean(), test_f1s_2.std())
    return test_f1s
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM', help='Dataset')
    parser.add_argument('--data_path', default='../GTN/data/', help='folder path of saved preprocessed data.')
    parser.add_argument('--epoch', type=int, default=500, help='Training Epochs')
    parser.add_argument('--patience', type=int, default=100, help='Training Epochs')
    parser.add_argument('--n_hid', type=int, default=64, help='num of hidden features.')
    parser.add_argument('--edge_dim', type=int, default=64, help='num of edge features.')
    parser.add_argument('--num_heads', type=int, default=8, help='num heads for attention layer')
    parser.add_argument('--dropout', type=float, default=0.5, help='MLP1 dropout')
    parser.add_argument('--dropout2', type=float, default=0.5, help='Intermediate layer dropout')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--hop', type=int, default=2, help='number of layer')
    parser.add_argument('--negative_slope', type=float, default=0.05, help='leakyrelu negative slope')
    parser.add_argument('--beta', type=float, default=0.05, help='edge residue factor')
    parser.add_argument("--num_test", type=int, default=30, help='num of runs to test accuracy.')
    parser.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    parser.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    parser.add_argument('--log_step', type=int, default=1000, help='training log step.')

    args = parser.parse_args()
    print(args)
    f1s = main(args)
    

