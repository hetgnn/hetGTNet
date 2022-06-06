import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_DBLP_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch, evaluate_f1, remove_edge_pts
from model import MAGNN_nc_mb

def run_model_DBLP(args):
    try:
        os.mkdir(args.save_path)
    except:
        print('model folder exists!')
    feats_type, hidden_dim = args.feats_type, args.hidden_dim
    num_heads, attn_vec_dim, rnn_type = args.num_heads, args.attn_vec_dim, args.rnn_type
    num_epochs, patience, repeat = args.epoch, args.patience, args.repeat
    batch_size, neighbor_samples = args.batch_size, args.samples
    
    out_dim = 4
    dropout_rate = args.dropout #0.5
    lr = args.learning_rate #0.005
    weight_decay = args.weight_decay #0.001
    # etype = {0: AP, 1: PA, 2: PT, 3:TP, 4:PV, 5:VP
    etypes_list = [[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]]
    
    adjlists, edge_metapath_indices_list, features_list, adjM, type_mask, labels, train_val_test_idx = load_DBLP_data(args.data_path+'preprocessed/DBLP_processed')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
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
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    # svm_macro_f1_lists = []
    # svm_micro_f1_lists = []
    # nmi_mean_list = []
    # nmi_std_list = []
    # ari_mean_list = []
    # ari_std_list = []
    # macro_f1_list = []
    f1s = []
    for i in range(repeat):
        net = MAGNN_nc_mb(3, 6, etypes_list, in_dims, hidden_dim, out_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        # if i == 0:
        #     print('#Parameters:', sum(p.numel() for p in net.parameters()))
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=False, save_path=args.save_path+'checkpoint_DBLP.pt')
        train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
        val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
        tic = time.time()
        t0 = time.time()
        training_time = 0
        for epoch in range(1, num_epochs + 1):
            t1 = time.time()
            # training
            net.train()
            for iteration in range(train_idx_generator.num_iterations()):
                # forward

                train_idx_batch = train_idx_generator.next()
                train_idx_batch.sort()
                train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
                    adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)

                logits, embeddings = net(
                    (train_g_list, features_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
                logp = F.log_softmax(logits, 1)
                train_loss = F.nll_loss(logp, labels[train_idx_batch])

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            
            training_time += time.time() - t1
            # validation
            net.eval()
            #val_logp = []
            val_logits = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                        adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
                    logits, embeddings = net(
                        (val_g_list, features_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
                    val_logits.append(logits)
                    #logp = F.log_softmax(logits, 1)
                    #val_logp.append(logp)
                #val_loss = F.nll_loss(torch.cat(val_logp, 0), labels[val_idx])
                val_logits = torch.cat(val_logits, 0)
                f1 = evaluate_f1(labels[val_idx].cpu().numpy(), val_logits.detach().cpu().numpy(), args.average)

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
        test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
        net.load_state_dict(torch.load(args.save_path+'checkpoint_DBLP.pt'))
        net.eval()
        test_logits = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(adjlists,
                                                                                             edge_metapath_indices_list,
                                                                                             test_idx_batch,
                                                                                             device, neighbor_samples)
                logits, embeddings = net((test_g_list, features_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
                #test_embeddings.append(embeddings)
                test_logits.append(logits)
            test_logits = torch.cat(test_logits, 0)
            #svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
            #    test_embeddings.cpu().numpy(), labels[test_idx].cpu().numpy(), num_classes=out_dim)
            f1 = evaluate_f1(labels[test_idx].cpu().numpy(), test_logits.detach().cpu().numpy(), args.average)
        print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 ({}) = {:.3f}, test_f1 ({}) = {:.3f}'.
              format(time.time() - t0, training_time/epoch, args.average, early_stopping.best_score, args.average, f1))
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
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--data_path', default='data/', help='folder path of saved preprocessed data.')
    ap.add_argument('--save_path', default='checkpoint/', help='folder path of saved model.')
    ap.add_argument('--feats_type', type=int, default=2,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2.')
    ap.add_argument('--hidden_dim', type=int, default=8, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    ap.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.')
    ap.add_argument('-wd', '--weight_decay', type=float, default=0.001, help='weight decay in Adam optimizer.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch_size', type=int, default=8, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    ap.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    ap.add_argument('--log_step', type=int, default=5, help='training log step.')

    args = ap.parse_args()
    f1s = run_model_DBLP(args)
