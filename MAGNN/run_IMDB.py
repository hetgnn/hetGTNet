import time
import argparse
import os
import torch.nn.functional as F
import torch.sparse
import numpy as np
import dgl

from utils.pytorchtools import EarlyStopping
from utils.data import load_IMDB_data
from utils.tools import evaluate_results_nc, evaluate_f1, remove_edge_pts
from model import MAGNN_nc

def run_model_IMDB(args):
    try:
        os.mkdir(args.save_path)
    except:
        print('model folder exists!')
    feats_type, num_layers, hidden_dim = args.feats_type, args.layers, args.hidden_dim
    num_heads, attn_vec_dim, rnn_type = args.num_heads, args.attn_vec_dim, args.rnn_type
    num_epochs, patience, repeat = args.epoch, args.patience, args.repeat
    out_dim = 3
    dropout_rate = 0.5
    lr = 0.005
    weight_decay = 0.001
    etypes_lists = [[[0, 1], [2, 3]],
                    [[1, 0], [1, 2, 3, 0]],
                    [[3, 2], [3, 0, 1, 2]]]

    nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, train_val_test_idx = load_IMDB_data(args.data_path+'preprocessed/IMDB_processed')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1:
        in_dims = [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(1, len(features_list)):
            features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2:
        in_dims = [features.shape[0] for features in features_list]
        in_dims[0] = features_list[0].shape[1]
        for i in range(1, len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                   edge_metapath_indices_lists]
    labels = torch.LongTensor(labels).to(device)
    g_lists = []
    for nx_G_list in nx_G_lists:
        g_lists.append([])
        for nx_G in nx_G_list:
            g = dgl.DGLGraph(multigraph=True).to(device)
            g.add_nodes(nx_G.number_of_nodes())
            g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
            g_lists[-1].append(g)
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    # svm_macro_f1_lists = []
    # svm_micro_f1_lists = []
    # nmi_mean_list = []
    # nmi_std_list = []
    # ari_mean_list = []
    # ari_std_list = []
    # macro_f1_list = []
    f1s = []
    for i in range(repeat):
        net = MAGNN_nc(num_layers, [2, 2, 2], 4, etypes_lists, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim,
                       rnn_type, dropout_rate)
        # if i == 0:
        #     print('#Parameters:', sum(p.numel() for p in net.parameters()))
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        target_node_indices = np.where(type_mask == 0)[0]

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=args.save_path+'checkpoint_IMDB.pt')
        tic = time.time()
        t0 = time.time()
        training_time = 0
        for epoch in range(1, num_epochs + 1):
            t1 = time.time()
            # training forward
            net.train()
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            logp = F.log_softmax(logits, 1)
            train_loss = F.nll_loss(logp[train_idx], labels[train_idx])

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            training_time += time.time() - t1

            # validation forward
            net.eval()
            with torch.no_grad():
                logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
                f1 = evaluate_f1(labels[val_idx].cpu().numpy(), logits[val_idx].detach().cpu().numpy(), args.average)
                #logp = F.log_softmax(logits, 1)
                #val_loss = F.nll_loss(logp[val_idx], labels[val_idx])

            # early stopping
            #early_stopping(val_loss, net)
            early_stopping(-f1, net)
            if early_stopping.early_stop:
                #print('Early stopping!')
                break
            if epoch % args.log_step == 0:
                print('Epoch {}, time elapsed: {:.3f}, loss: {:.3f}, val f1: {:.3f} (best {:.3f})'.
                      format(epoch, time.time() - tic, train_loss.detach().item(), f1, early_stopping.best_score))
                tic = time.time()

        # testing
        net.load_state_dict(torch.load(args.save_path+'checkpoint_IMDB.pt'))
        net.eval()
        with torch.no_grad():
            logits, embeddings = net((g_lists, features_list, type_mask, edge_metapath_indices_lists), target_node_indices)
            f1 = evaluate_f1(labels[test_idx].cpu().numpy(), logits[test_idx].detach().cpu().numpy(), args.average)
        print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 ({}) = {:.3f}, test_f1 ({}) = {:.3f}'.
              format(time.time() - t0, training_time/epoch, args.average, early_stopping.best_score, args.average, f1))
        #     svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
        #         embeddings[test_idx].cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
        f1s.append(f1)
        # svm_macro_f1_lists.append(svm_macro_f1_list)
        # svm_micro_f1_lists.append(svm_micro_f1_list)
        # nmi_mean_list.append(nmi_mean)
        # nmi_std_list.append(nmi_std)
        # ari_mean_list.append(ari_mean)
        # ari_std_list.append(ari_std)

    # print out a summary of the evaluations
    f1s = np.array(f1s)
    print('----------------------------------------------------------------')
    print('test {}-f1 (mean, std): '.format(args.average), f1s.mean(), f1s.std())
    f1s_2 = remove_edge_pts(f1s, pct=args.filter_pct)
    print('test {}-f1 (mean, std) after filter: '.format(args.average), f1s_2.mean(), f1s_2.std())
    # svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2))
    # svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2))
    # nmi_mean_list = np.array(nmi_mean_list)
    # nmi_std_list = np.array(nmi_std_list)
    # ari_mean_list = np.array(ari_mean_list)
    # ari_std_list = np.array(ari_std_list)
    # print('----------------------------------------------------------------')
    # print('SVM tests summary')
    # print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
    #     macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
    #     zip(svm_macro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    # print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
    #     micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
    #     zip(svm_micro_f1_lists, [0.8, 0.6, 0.4, 0.2])]))
    # print('K-means tests summary')
    # print('NMI: {:.6f}~{:.6f}'.format(nmi_mean_list.mean(), nmi_std_list.mean()))
    # print('ARI: {:.6f}~{:.6f}'.format(ari_mean_list.mean(), ari_std_list.mean()))
    return f1s


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing')
    ap.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    ap.add_argument('--save_path', default='checkpoint/', help='folder path of saved model.')
    ap.add_argument('--feats_type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--layers', type=int, default=2, help='Number of layers. Default is 2.')
    ap.add_argument('--hidden_dim', type=int, default=8, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn_type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    ap.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.')
    ap.add_argument('-wd', '--weight_decay', type=float, default=0.001, help='weight decay in Adam optimizer.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
    ap.add_argument('--repeat', type=int, default=30, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    ap.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    ap.add_argument('--log_step', type=int, default=5, help='training log step.')

    args = ap.parse_args()
    f1s = run_model_IMDB(args)
