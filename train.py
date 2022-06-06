import numpy as np
import torch
import torch.nn as nn
from models import HAN, HGT, HetGTAN, HetGTCN, HetGCN, HetGAT, RGCN, HetGTAN_LW, HetGAT_NoSem, HetGTAN_NoSem, HetGTAN_mean, HetGTCN_mean, HetGTCN_LW, HetGAT_mean
from utils import remove_edge_pts, partition_edge_index_dict
#from collections import Counter
import time
from sklearn import metrics
import argparse
import pickle
import dgl
#import sys

def train(model, data, args):
    # it is used for HetGTAN or torch_geometric's HGT models
    device = args.device
    model.to(device)
    x_dict, edge_index_dict, y = data['x_dict'], data['edge_index_dict'], data['labels']
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

    for node_type in x_dict:
        x_dict[node_type] = x_dict[node_type].to(device)
    for edge_type in data['edge_index_dict']:
        edge_index_dict[edge_type] = edge_index_dict[edge_type].to(device)
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
        y_pred = model(x_dict, edge_index_dict, args.target_node_type)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dict, edge_index_dict, args.target_node_type)
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
    return test_f1, model 

def train_HetGTCN(model, data, args):
    device = args.device
    model.to(device)
    x_dict, y = data['x_dict'], data['labels']
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

    for node_type in x_dict:
        x_dict[node_type] = x_dict[node_type].to(device)
    #for edge_type in data['edge_index_dict']:
    #    edge_index_dict[edge_type] = edge_index_dict[edge_type].to(device)
    num_node_type_dict = {}
    for node_type in x_dict:
        num_node_type_dict[node_type] = x_dict[node_type].size(0)
    A1_dict, A2_dict = partition_edge_index_dict(data['edge_index_dict'], num_node_type_dict, args.device)
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
        y_pred = model(x_dict, A1_dict, A2_dict, args.target_node_type)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dict, A1_dict, A2_dict, args.target_node_type)
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
    return test_f1, model 

def train_HAN(model, data, args):
    # it is used for HAN models
    device = args.device
    model.to(device)
    g = [graph.to(device) for graph in data['g']]
    x, y = data['features'].to(device), data['labels'].to(device)
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

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
        y_pred = model(g, x)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(g, x)
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
    return test_f1, model

def train_HGT(model, G, data, args):
    # it is used for HGT model
    device = args.device
    model.to(device)
    G = G.to(device)
    y = data['labels'].to(device)
    train_mask, val_mask, test_mask = data['train_mask'], data['val_mask'], data['test_mask']

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
        y_pred = model(G, args.target_node_type)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(G, args.target_node_type)
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
    return test_f1, model

def train_RGCN(model, data, args):
    # it is used for modified RGCN models
    device = args.device
    model.to(device)
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
        edge_type1 = torch.zeros(A_pa.size(1), dtype=torch.int64)
        edge_type2 = torch.ones(A_ps.size(1), dtype=torch.int64)
        edge_type = torch.cat((edge_type1, edge_type2, edge_type1, edge_type2))
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
        edge_type1 = torch.zeros(A_ap.size(1), dtype=torch.int64)
        edge_type2 = torch.ones(A_pt.size(1), dtype=torch.int64)
        edge_type3 = 2 * torch.ones(A_pc.size(1), dtype=torch.int64)
        edge_type = torch.cat((edge_type1, edge_type2, edge_type3, edge_type1, edge_type2, edge_type3))
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
        edge_type1 = torch.zeros(A_md.size(1), dtype=torch.int64)
        edge_type2 = torch.ones(A_ma.size(1), dtype=torch.int64)
        edge_type = torch.cat((edge_type1, edge_type2, edge_type1, edge_type2))
        
    for node_type in x_dict:
        x_dict[node_type] = x_dict[node_type].to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
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
        y_pred = model(x_dict, node_type_order, edge_index, edge_type)
        loss = criterion(y_pred[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_time += time.time() - t1
        
        # evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dict, node_type_order, edge_index.to(device), edge_type.to(device))
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_f1

def main(args):
    # works for model HAN, HetGTCN, HetHTAN and HGT, which have the same format of input data
    if args.model == 'HAN':
        load_data_name = args.data + '_HAN.pkl'
    else:
        load_data_name = args.data + '.pkl'
    with open(args.data_path + load_data_name, 'rb') as fp:
        data = pickle.load(fp)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_out = data['labels'].max().item() + 1
    f1s = []
    for i in range(args.num_test):
        if args.model == 'HAN':
            model = HAN(len(data['g']), data['features'].size(1), args.n_hid//args.num_heads, n_out, [args.num_heads] * args.hop, args.dropout)
            # if i == 0:
            #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
            f1, _ = train_HAN(model, data, args)
        else:
            n_in_dict = {}
            for node_type in data['x_dict']:
                n_in_dict[node_type] = data['x_dict'][node_type].size(-1)
            if args.model == 'HetGTCN':
                model = HetGTCN(n_in_dict, args.n_hid, n_out, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train_HetGTCN(model, data, args)
            elif args.model == 'HetGTAN':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGTAN(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HGT':
                # build DGLGraph
                data_dict = {}
                etype_enum = 0
                for edge_type, edge_index in data['edge_index_dict'].items():
                    # rename the edge_types
                    #data_dict[(edge_type[0], str(etype_enum), edge_type[-1])] = (edge_index[0].numpy(), edge_index[1].numpy())
                    data_dict[(edge_type[0], str(etype_enum), edge_type[-1])] = (edge_index[0], edge_index[1])
                    etype_enum += 1
                num_nodes_dict = {}
                for node_type, x in data['x_dict'].items():
                    num_nodes_dict[node_type] = x.shape[0]
                G = dgl.heterograph(data_dict, num_nodes_dict)
                node_dict = {}
                edge_dict = {}
                n_inps = []
                for ntype in G.ntypes:
                    node_dict[ntype] = len(node_dict)
                    n_inps.append(data['x_dict'][ntype].shape[-1])
                for etype in G.etypes:
                    edge_dict[etype] = len(edge_dict)
                    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 
                # node feature
                for ntype in G.ntypes:
                    G.nodes[ntype].data['inp'] = data['x_dict'][ntype]
                model = HGT(G, node_dict, edge_dict, n_inps, args.n_hid, n_out, n_layers=args.hop, n_heads=args.num_heads, use_norm = True)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train_HGT(model, G, data, args)
            elif args.model == 'HetGCN':
                model = HetGCN(n_in_dict, args.n_hid, n_out, args.dropout, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train_HetGTCN(model, data, args)
            elif args.model == 'HetGAT':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGAT(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'RGCN':
                num_relations = (len(data['edge_index_dict']) + 1)//2
                model = RGCN(n_in_dict, args.n_hid, n_out, num_relations, args.num_bases, args.dropout, args.hop)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1 = train_RGCN(model, data, args)
            elif args.model == 'HetGAT_NoSem':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGAT_NoSem(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HetGAT_mean':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGAT_mean(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HetGTAN_NoSem':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGTAN_NoSem(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HetGTAN_LW':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGTAN_LW(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HetGTAN_mean':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGTAN_mean(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train(model, data, args)
            elif args.model == 'HetGTCN_mean':
                model = HetGTCN_mean(n_in_dict, args.n_hid, n_out, args.dropout, args.dropout2, args.hop)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train_HetGTCN(model, data, args)
            elif args.model == 'HetGTCN_LW':
                edge_types = list(data['edge_index_dict'].keys())
                model = HetGTCN_LW(n_in_dict, args.n_hid, n_out, edge_types, args.dropout, args.dropout2, args.hop, args.layer_wise)
                # if i == 0:
                #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
                f1, _ = train_HetGTCN(model, data, args)
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
    parser.add_argument('--data', default='ACM', help='Name of dataset.')
    parser.add_argument('--data_path', default='data/preprocessed/', help='folder path of saved preprocessed data.')
    parser.add_argument('--model', default='HetGTAN', help='Heterogeneous GNN model.')
    parser.add_argument('--target_node_type', default='paper', help='Target node type: paper for ACM data.')
    parser.add_argument('--n_hid', type=int, default=64, help='num of hidden features.')
    parser.add_argument('--num_heads', type=int, default=8, help='num heads for attention layer')
    parser.add_argument('--dropout', type=float, default=0.8, help='MLP1 dropout')
    parser.add_argument('--dropout2', type=float, default=0.2, help='Intermediate layer dropout')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00005, help='weight decay in Adam optimizer.')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience.')
    parser.add_argument('--num_iter', type=int, default=500, help='Max epochs to run.')
    parser.add_argument('--num_test', type=int, default=30, help='num of runs to test accuracy.')
    parser.add_argument('--hop', type=int, default=5, help='hop of GNN models.')
    parser.add_argument('--num_bases', type=int, default=5, help='num bases for RGCN model.')
    parser.add_argument('--filter_pct', type=float, default=0.1, 
                        help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.')
    parser.add_argument('--log_step', type=int, default=1000, help='training log step.')
    parser.add_argument('-lw', '--layer_wise', action='store_true', default=False, 
                        help="whether to share parameters for different layers")
    parser.add_argument('--average', default='macro', help='f1 average: can choose either macro or micro.')
    args = parser.parse_args()
    f1s = main(args)

