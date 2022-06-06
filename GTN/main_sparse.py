import torch
import numpy as np
import torch.nn as nn
# import torch.nn.functional as F
from model_sparse import GTN
# from matplotlib import pyplot as plt
# import pdb
# from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
# from torch_geometric.data import Data
# import torch_sparse
import pickle
#from mem import mem_report
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
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
    if args.use_cpu:
        device = torch.device('cpu')
    #norm = args.norm
    adaptive_lr = args.adaptive_lr
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
    model = GTN(num_edge=len(A),
                num_channels=args.num_channels,
                w_in = node_features.shape[1],
                w_out = args.node_dim,
                num_class=num_classes,
                num_nodes = node_features.shape[0],
                num_layers= args.num_layers)
    # print('#Parameters:', sum(p.numel() for p in model.parameters()))
    model.to(device)
    
    if adaptive_lr == 'false':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params':model.gcn.parameters()},
                                      {'params':model.linear1.parameters()},
                                      {'params':model.linear2.parameters()},
                                      {"params":model.layers.parameters(), "lr":0.5}
                                      ], lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss().to(device)
    best_val_f1 = 0
    best_test_f1 = 0
    count = 0
    tic = time.time()
    t0 = time.time()
    training_time = 0
    for epoch in range(1, args.epoch + 1):
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        t1 = time.time()
        model.train()
        model.zero_grad()
        loss, y_train, _ = model(A, node_features, train_node, train_target)
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        model.eval()
        # Valid
        with torch.no_grad():
            val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
            test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
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
    parser.add_argument('--dataset', type=str, default='DBLP', help='Dataset')
    parser.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    parser.add_argument('--epoch', type=int, default=50, help='Training Epochs')
    parser.add_argument('--patience', type=int, default=50, help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64, help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2, help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
    parser.add_argument('--norm', type=str, default='true', help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false', help='adaptive learning rate')
    parser.add_argument("--num_test", type=int, default=30, help='num of runs to test accuracy.')
    parser.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    parser.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    parser.add_argument('--log_step', type=int, default=1000, help='training log step.')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='adaptive learning rate')

    args = parser.parse_args()
    print(args)
    f1s = main(args)
    

