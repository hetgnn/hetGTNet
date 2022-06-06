import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model import GTN
import pdb
import pickle
import argparse
from utils import f1_score
import time
from sklearn import metrics

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

def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.use_cpu:
        device = torch.device('cpu')
    node_dim = args.node_dim
    num_channels = args.num_channels
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open(args.data_path+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open(args.data_path+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open(args.data_path+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
        
        
    num_nodes = edges[0].shape[0]
    A = []
    for i,edge in enumerate(edges):
        if i ==0:
            A = torch.from_numpy(edge.todense()).float().unsqueeze(-1).to(device)
        else:
            A = torch.cat([A,torch.from_numpy(edge.todense()).float().unsqueeze(-1).to(device)], dim=-1)
    A = torch.cat([A,torch.eye(num_nodes).unsqueeze(-1).to(device)], dim=-1)
    
    node_features = torch.from_numpy(node_features).type(torch.float32).to(device)
    train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.int64).to(device)
    train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.int64).to(device)
    valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.int64).to(device)
    valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.int64).to(device)
    test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.int64).to(device)
    test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.int64).to(device)
    num_classes = torch.max(train_target).item()+1

    # train_losses = []
    # train_f1s = []
    # val_losses = []
    # test_losses = []
    # val_f1s = []
    # test_f1s = []
    # final_f1 = 0
    test_f1s = []
    for cnt in range(args.num_test):
        # best_val_loss = 10000
        # best_test_loss = 10000
        # best_train_loss = 10000
        # best_train_f1 = 0
        # best_val_f1 = 0
        # best_test_f1 = 0
        model = GTN(num_edge=A.shape[-1],
                        num_channels=num_channels,
                        w_in = node_features.shape[1],
                        w_out = node_dim,
                        num_class=num_classes,
                        num_layers=num_layers,
                        norm=norm)
        
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
        #Ws = []
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
            #train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
            #print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
            model.eval()
            # Valid
            with torch.no_grad():
                val_loss, y_valid,_ = model.forward(A, node_features, valid_node, valid_target)
                # val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
                #print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
                test_loss, y_test,W = model.forward(A, node_features, test_node, test_target)
                # test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
                # test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
                val_f1, test_f1 = evaluate_f1(y_valid, valid_target, y_test, test_target, average=args.average)
                #print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 (macro) = {:.3f}, best_test_f1 (macro) = {:.3f}'.
              format(time.time() - t0, training_time/epoch, best_val_f1, best_test_f1))
        test_f1s.append(best_test_f1)
    test_f1s = np.array(test_f1s)
    print('test macro-f1 (mean, std): ', test_f1s.mean(), test_f1s.std())
    test_f1s_2 = remove_edge_pts(test_f1s, pct=args.filter_pct)
    print('test macro-f1 (mean, std) after filter: ', test_f1s_2.mean(), test_f1s_2.std())
    return test_f1s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM', help='Dataset')
    parser.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    parser.add_argument('--epoch', type=int, default=500, help='Training Epochs')
    parser.add_argument('--patience', type=int, default=100, help='Training Epochs')
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
