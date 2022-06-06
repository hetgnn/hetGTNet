import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import dgl
#from dgl.data.utils import download, get_download_dir, _get_dgl_url
#from pprint import pprint
import scipy
#from scipy import sparse
from scipy import io
import pickle
import scipy.io as sio
import networkx as nx
import pathlib
#import sys
import time
import os
from torch_geometric.datasets import DBLP, IMDB

def partition_edge_index_dict(edge_index_dict, num_node_type_dict, device):
    A1_dict, A2_dict = {}, {}
    for edge_type in edge_index_dict:
        edge_index = edge_index_dict[edge_type]
        node_type_s, node_type_t = edge_type[0], edge_type[-1]
        # build adj matrix based on edge_index
        A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_node_type_dict[node_type_s], num_node_type_dict[node_type_t]))
        A1, A2 = separate2(A)
        if node_type_s not in A1_dict:
            A1_dict[node_type_s], A2_dict[node_type_s] = {}, {}
        A1_dict[node_type_s][node_type_t] = A1.to(device)
        A2_dict[node_type_s][node_type_t] = A2.to(device)
    return A1_dict, A2_dict
        
def clean_A(A):
    s, t = A._indices().tolist()
    N = A.size(0)
    idx = []
    for i in range(len(s)):
        if s[i] == t[i]:
            idx.append(i)
    #print('self_loop # = ', len(idx))
    for i in idx[::-1]:
        del s[i]
        del t[i]
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    return A
    
def data_split(x, y, training_samples=20, val_samples=30):
    n_class = len(set(y.numpy()))
    #print('n_class = ', n_class)
    sel_samples = training_samples + val_samples
    sampling_strategy, train_samples = {}, {}
    for i in range(n_class):
        sampling_strategy[i] = sel_samples
        train_samples[i] = training_samples
    #sampling_strategy = {0:90, 1:90, 2:90, 3:90, 4:90, 5:90, 6:90}
    #train_samples = {0:20, 1:20, 2:20, 3:20, 4:20, 5:20, 6:20}
    rus1 = RandomUnderSampler(sampling_strategy=sampling_strategy)
    rus2 = RandomUnderSampler(sampling_strategy=train_samples)
    x_res, y_res = rus1.fit_resample(x.numpy(), y.numpy())
    test_indice = set(range(len(y)))
    selected_indice = rus1.sample_indices_
    test_indice.difference_update(set(selected_indice))
    test_indice = np.array(list(test_indice), dtype=np.int64)

    rus2.fit_resample(x_res, y_res)
    selected_indice2 = rus2.sample_indices_
    unselected_indice2 = set(range(len(y_res)))
    unselected_indice2.difference_update(set(selected_indice2))
    train_indice = selected_indice[selected_indice2]
    val_indice = selected_indice[list(unselected_indice2)]
    train_indice.sort()
    val_indice.sort()
    test_indice.sort()
    return train_indice, val_indice, test_indice

# D^-0.5 x A x D^-0.5
def norm_adj(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = torch.pow(degrees, -0.5)
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, torch.sparse.mm(A, D))

# D^-1 x A
def norm_adj2(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = 1/degrees
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, A)

def remove_edge_pts(accs, pct=0.1):
    accs = sorted(list(accs))
    N = len(accs)
    M = int(N * pct)
    accs = np.array(accs[M:N-M])
    return accs

def separate(A):
    A = norm_adj(A, self_loop=True)
    s, t = A._indices().tolist()
    N = A.size(0)
    values = A._values().tolist()
    value1 = [0] * N
    value2 = []
    s1, t1 = [], []
    for i in range(len(s)):
        if s[i] == t[i]:
            value1[s[i]] = values[i]
        else:
            s1.append(s[i])
            t1.append(t[i])
            value2.append(values[i])
    A1 = torch.sparse_coo_tensor([s1, t1], torch.tensor(value2, dtype=torch.float32), (N, N))
    A2 = torch.tensor(value1, dtype=torch.float32).unsqueeze(-1)
    return A1, A2

def separate2(A):
    degrees = torch.sparse.sum(A, dim=1).to_dense() + 1
    degrees = 1/degrees
    #degrees[torch.isinf(degrees)] = 0
    s, t = A._indices().tolist()
    N, M = A.size()
    values = A._values().tolist()
    value2 = []
    for i in range(len(s)):
        value2.append(values[i] * degrees[s[i]])
    A1 = torch.sparse_coo_tensor([s, t], torch.tensor(value2, dtype=torch.float32), (N, M))
    A2 = degrees.float().unsqueeze(-1)
    return A1, A2

def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])])
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(scipy.sparse.csr_matrix(adjM[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])]))
    return out_adjM.toarray()


# networkx.has_path may search too
def get_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:
        # consider only the edges relevant to the expected metapath
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])] = True
            temp[np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])] = True
            mask = np.logical_or(mask, temp)
        #A = (M * mask).astype(int)
        #A = scipy.sparse.csr_matrix(A)
        #partial_g_nx = nx.from_scipy_sparse_matrix(A)
        partial_g_nx = nx.from_numpy_matrix((M * mask).astype(int))

        # only need to consider the former half of the metapath
        # e.g., we only need to consider 0-1-2 for the metapath 0-1-2-1-0
        metapath_to_target = {}
        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (type_mask == metapath[(len(metapath) - 1) // 2]).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1)
                if target in single_source_paths:
                    has_path = True

                #if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [p for p in nx.all_shortest_paths(partial_g_nx, source, target) if
                                 len(p) == (len(metapath) + 1) // 2]
                    if len(shortests) > 0:
                        metapath_to_target[target] = metapath_to_target.get(target, []) + shortests
        metapath_neighbor_paris = {}
        for key, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_paris[(p1[0], p2[0])] = metapath_neighbor_paris.get((p1[0], p2[0]), []) + [
                        p1 + p2[-2::-1]]
        outs.append(metapath_neighbor_paris)
    return outs


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(metapath_neighbor_pairs.items())
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array

def preprocess_ACM(path = './data/ACM/ACM.mat', model_name='HAN', num_train=600, num_val=300, load_label_split=True):
    # target node is paper
    data = io.loadmat(path)
    A_ps = data['PvsL']     # paper-subject adjacency matrix
    A_pa = data['PvsA']     # paper-author adjacency matrix
    A_pt = data['PvsT']     # paper-term, bag of words
    A_pc = data['PvsC']     # paper-conference (for labels), 14 conferences (one-hot) for each paper
    
    # class 0 (data mining): KDD papers (0)
    # class 1 (database): SIGMOD and VLDB (1, 13)
    # class 2 communication): SIGCOMM and MOBICOMM (9, 10)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    
    pc_filter = A_pc[:, conf_ids] # filter out papers not in the above 5 conferences
    paper_idx = (pc_filter.sum(1) != 0).A1.nonzero()[0]
    A_ps = A_ps[paper_idx]
    A_pa = A_pa[paper_idx]
    A_pt = A_pt[paper_idx]
    A_pc = A_pc[paper_idx]
    
    # remove empty rows for A_pa and A_ps
    idx1 = (A_pa.sum(1) == 0).nonzero()[0]
    idx2 = (A_ps.sum(1) == 0).nonzero()[0]
    idx = set(range(A_pa.shape[0]))
    idx.difference_update(set(idx1))
    idx.difference_update(set(idx2))
    paper_idx = sorted(list(idx))
    A_ps = A_ps[paper_idx]
    A_pa = A_pa[paper_idx]
    A_pt = A_pt[paper_idx]
    A_pc = A_pc[paper_idx]
    
    # remove empty cols of A_ps, A_pa, A_pt, and reconstruct A_ps, A_pa, A_pt
    src, authors = A_pa.nonzero()
    author_dict = {}
    re_authors = []
    for author in authors:
        if author not in author_dict:
            author_dict[author] = len(author_dict) # + len(paper_idx)
        re_authors.append(author_dict[author])
    re_authors = np.array(re_authors)
    A_pa = scipy.sparse.csr_matrix((np.ones(len(src)), (src, re_authors)), shape=(len(paper_idx),len(author_dict)))
    
    src, subjects = A_ps.nonzero()
    subject_dict = {}
    re_subjects = []
    for subject in subjects:
        if subject not in subject_dict:
            subject_dict[subject] = len(subject_dict) # + len(paper_idx) + len(author_dic)
        re_subjects.append(subject_dict[subject])
    re_subjects = np.array(re_subjects)
    A_ps = scipy.sparse.csr_matrix((np.ones(len(src)), (src, re_subjects)), shape=(len(paper_idx),len(subject_dict)))
    
    src, terms = A_pt.nonzero()
    term_dict = {}
    re_terms = []
    for term in terms:
        if term not in term_dict:
            term_dict[term] = len(term_dict)
        re_terms.append(term_dict[term])
    re_terms = np.array(re_terms)
    A_pt = scipy.sparse.csr_matrix((np.ones(len(src)), (src, re_terms)), shape=(len(paper_idx),len(term_dict)))

    # generate target node features and corresponding labels
    features_paper = torch.FloatTensor(A_pt.toarray())
    pc_p, pc_c = A_pc.nonzero()
    labels = np.zeros(len(paper_idx), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)
    num_classes = 3
    # generate label splits, training samples: 600 (balanced), val samples: 300 (balanced), test samples: all remaining
    if load_label_split:
        with open('label_split/ACM_label_split', 'rb') as fp:
            train_mask, val_mask, test_mask = pickle.load(fp)
    else:
        train_mask, val_mask, test_mask = data_split(features_paper, labels, num_train//num_classes, num_val//num_classes)
        # save label split
        try:
            os.mkdir('label_split')
        except:
            print('folder created!')
        with open('label_split/ACM_label_split', 'wb') as fp:
            pickle.dump([train_mask, val_mask, test_mask], fp)
    
    if model_name == 'HAN':
        '''
        Prepare input data for HAN model (DGL version), which requires:
            1. g: list(dglGraph), each graph corresponds to one metapath
            2. node_features: Tensor of size N_paper x F
            3. labels: Tensor of size N_paper
            4. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        A_pap = A_pa.dot(A_pa.T)
        A_pap.data = np.ones(A_pap.data.shape)
        A_psp = A_ps.dot(A_ps.T)
        A_psp.data = np.ones(A_psp.data.shape)
        
        # create g
        author_g = dgl.from_scipy(A_pap)
        subject_g = dgl.from_scipy(A_psp)
        g = [author_g, subject_g]
        data = {'g': g, 'features': features_paper, 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'ACM_HAN.pkl', 'wb') as fp:
            pickle.dump(data, fp)
    
    elif model_name == 'GTN':
        '''
        Prepare input data for GTN model (https://github.com/seongjunyun/Graph_Transformer_Networks), which requires:
            1. node_features: Numpy array of size (N_paper + N_author + N_subject) x F,
                here concatenate nodes of all types [paper, author, subject] (in order) 
            2. edges: list of scipy.sparse adj matrices [PA, AP, PS, SP]
            3. labels: Numpy array of size N_paper x 2, each element is  np.array([label_idx, label])
            
            Note: the author feature is the mean of author's paper features.
        '''
        # obtain author, subject feature
        norm_factor = 1/np.array(A_pa.sum(axis=0), dtype=np.float32).flatten()
        A_ap_norm = scipy.sparse.diags(norm_factor).dot(A_pa.T)
        features_author = A_ap_norm.dot(A_pt)
        norm_factor = 1/np.array(A_ps.sum(axis=0), dtype=np.float32).flatten()
        A_sp_norm = scipy.sparse.diags(norm_factor).dot(A_ps.T)
        features_subject = A_sp_norm.dot(A_pt)
        features_author = features_author.astype(np.float32).toarray()
        features_subject = features_subject.astype(np.float32).toarray()
        
        # concatenate features
        node_features = np.vstack((features_paper.numpy(), features_author, features_subject))
        
        # edges: change A_pa and A_ps to the size of len(node_features) x len(node_features)
        N = node_features.shape[0]
        src, tgt = A_pa.nonzero()
        tgt = tgt + len(paper_idx)
        A_pa = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))

        src, tgt = A_ps.nonzero()
        tgt = tgt + len(paper_idx) + len(author_dict)
        A_ps = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))
        
        edges = [A_pa, A_pa.T, A_ps, A_ps.T]
        
        # labels
        train_label = np.vstack((train_mask, labels[train_mask].numpy())).T
        val_label = np.vstack((val_mask, labels[val_mask].numpy())).T
        test_label = np.vstack((test_mask, labels[test_mask].numpy())).T
        GTN_labels = [train_label, val_label, test_label]
        
        # save data
        save_folder = 'GTN/data/ACM/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'node_features.pkl', 'wb') as fp:
            pickle.dump(node_features, fp)
        with open(save_folder + 'edges.pkl', 'wb') as fp:
            pickle.dump(edges, fp)
        with open(save_folder + 'labels.pkl', 'wb') as fp:
            pickle.dump(GTN_labels, fp)
        #return node_features, edges, GTN_labels
    
    elif model_name == 'MAGNN':
        '''
        Prepare input data for MAGNN model (https://github.com/cynricfu/MAGNN), which requires:
            1. nx_G_list: list of <num_node_type> nx_Graphs, where the ith element corresponds to the nx_Graph
                with metapath starting and ending at nodes of type i, like metapath = [{0, 1, 0}, {0, 2, 0}] for node type 0.
            2. edge_metapath_indices_list: same shape of nx_G_list. It is corresponding node indices for each edge_metapath.
            3. features_list: list of scipy.sparse matrix: [features_paper, features_author, features_subject]
            4. adjM: combined one scipy.sparse matrix with shape (N_paper + N_author + N_subject) x (N_paper + N_author + N_subject)
            5. type_mask: 1D numpy array of size (N_paper + N_author + N_subject), where type 0: paper, type 1: author, type 2: subject
            6. labels: numpy of size N_paper
            7. train_val_test_idx; [val_idx, train_idx, test_idx], each is 1D numpy array
            
            Note: for fair comparison, the author and subject feature are obtained the same way as GTN: 
            mean of author's paper features, and mean of subject's paper features.
        '''
        # type_mask. 0: paper, 1: author, 2: subject
        N = len(paper_idx) + len(author_dict) + len(subject_dict)
        type_mask = np.zeros(N, dtype=int)
        type_mask[len(paper_idx):len(paper_idx) + len(author_dict)] = 1
        type_mask[len(paper_idx) + len(author_dict):] = 2
        
        # obtain feature list
        #features_author = (A_pa.T).dot(A_pt)
        #features_subject = (A_ps.T).dot(A_pt)
        norm_factor = 1/np.array(A_pa.sum(axis=0), dtype=np.float32).flatten()
        A_ap_norm = scipy.sparse.diags(norm_factor).dot(A_pa.T)
        features_author = A_ap_norm.dot(A_pt)
        norm_factor = 1/np.array(A_ps.sum(axis=0), dtype=np.float32).flatten()
        A_sp_norm = scipy.sparse.diags(norm_factor).dot(A_ps.T)
        features_subject = A_sp_norm.dot(A_pt)
        features_author = features_author.astype(np.float32).toarray()
        features_subject = features_subject.astype(np.float32).toarray()
        
        # obtain adjM
        src0, tgt0 = A_pa.nonzero()
        tgt0 = tgt0 + len(paper_idx)
        src1, tgt1 = A_ps.nonzero()
        tgt1 = tgt1 + len(paper_idx) + len(author_dict)
        src = np.hstack((src0, src1))
        tgt = np.hstack((tgt0, tgt1))
        src, tgt = np.hstack((src, tgt)), np.hstack((tgt, src)) # symmetric matrix needed
        adjM = scipy.sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (src, tgt)), shape=(N, N))
        adjM = adjM.toarray() # dense matrix required by MAGNN
        
        # obtain nx_G_list and edge_metapath_indices_list, here only considers metapath PAP, PSP
        expected_metapaths = [[(0,1,0), (0,2,0)]]
        num_ntypes = 3
        save_prefix = 'MAGNN/data/preprocessed/ACM_processed/'
        # create the directories if they do not exist
        for i in range(len(expected_metapaths)):
            pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
        for i in range(len(expected_metapaths)):
            t0 = time.time()
            # get metapath based neighbor pairs
            neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
            # construct and save metapath-based networks
            G_list = get_networkx_graph(neighbor_pairs, type_mask, i)
            print('time to generate metapath nodes = ', time.time() - t0)
    
        # save data
        # networkx graph (metapath specific)
        for G, metapath in zip(G_list, expected_metapaths[i]):
            nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
            # node indices of edge metapaths
        all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
        for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
            np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        # save nodes adjacency matrix
        scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
        # all nodes (authors, papers, subjects) features
        # currently only have features of authors, papers and terms
        features_paper = scipy.sparse.csr_matrix(features_paper.numpy())
        features_author = scipy.sparse.csr_matrix(features_author)
        features_subject = scipy.sparse.csr_matrix(features_subject)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_paper)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_author)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(2), features_subject)
        # save all nodes type labels
        np.save(save_prefix + 'node_types.npy', type_mask)
        # target labels
        np.save(save_prefix + 'labels.npy', labels.numpy())
        # save train/validation/test splits
        np.savez(save_prefix + 'train_val_test_idx.npz', val_idx=val_mask, train_idx=train_mask, test_idx=test_mask)
        
    elif model_name == 'DMGI':
        '''
        Prepare input data for DMGI model (https://github.com/pcy1302/DMGI), which requires:
            A dictionary containing the following keys
                train_idx: training index, val_idx: validation index, test_idx: test index. All having shape of 1 x num_train/val/test
                feature: feature matrix (float32 or float64), label: labels, 
            Relations (float32 or float64): 
                IMDB: MDM, MAM
                DBLP: PAP, PPP, PATAP
                ACM: PAP, PSP
        '''
        A_pap = A_pa.dot(A_pa.T)
        A_pap.data = np.ones(A_pap.data.shape, dtype=np.float32)
        A_psp = A_ps.dot(A_ps.T)
        A_psp.data = np.ones(A_psp.data.shape, dtype=np.float32)
        labels = labels.numpy()
        DMGI_labels = np.zeros((len(labels), labels.max()+1))
        DMGI_labels[np.arange(len(labels)), labels] = 1
        train_idx, val_idx, test_idx = train_mask[np.newaxis], val_mask[np.newaxis], test_mask[np.newaxis]
        data = {'label':DMGI_labels, 'feature':features_paper.numpy(), 'PAP':A_pap.toarray(), 'PSP':A_psp.toarray(), 
                'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}
        #data = {'label':labels, 'feature':features, 'PAP':PAP, 'PPP':PPrefP, 'PATAP':PATAP, 'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}
        # save data
        sio.savemat('DMGI/data/acm.mat', data)
    elif model_name in {'HGT', 'HetGTCN', 'HetGTAN'}:
        '''
        Prepare input data for HetGTCN, HetGTAN or HGT model (torch.geometric version: 
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HGTConv), which requires:
            1. x_dict (Dict[str, Tensor]) – A dictionary holding input node features for each individual node type
            2. edge_index_dict: A dictionary with key the edge type (tuple of source_node_type, relation, target_node_type, like 
                ('paper', 'to', 'author')), and value the edge connections, represented as a torch.LongTensor of shape [2, num_edges].
            #3. metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. 
            #    its node and edge types given by a list of strings and a list of string triplets, respectively. 
            #    For ACM, metadata = (['paper', 'author', 'subject'], 
            #    [('paper', 'to', 'author'), ('author', 'to', 'paper'), ('paper', 'to', 'subject'), ('subject', 'to', 'paper')])
            4. labels: Tensor of size N_paper
            5. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        norm_factor = 1/np.array(A_pa.sum(axis=0), dtype=np.float32).flatten()
        A_ap_norm = scipy.sparse.diags(norm_factor).dot(A_pa.T)
        features_author = A_ap_norm.dot(A_pt)
        norm_factor = 1/np.array(A_ps.sum(axis=0), dtype=np.float32).flatten()
        A_sp_norm = scipy.sparse.diags(norm_factor).dot(A_ps.T)
        features_subject = A_sp_norm.dot(A_pt)
        features_author = features_author.astype(np.float32).toarray()
        features_subject = features_subject.astype(np.float32).toarray()
        
        A_ap = A_pa.T
        A_sp = A_ps.T
        
        x_dict = {'paper': features_paper, 'author': torch.from_numpy(features_author), 'subject': torch.from_numpy(features_subject)}
        edge_index_dict = {('paper', 'to', 'author'): torch.from_numpy(np.array(A_pa.nonzero(), dtype=np.int64)), 
                           ('author', 'to', 'paper'): torch.from_numpy(np.array(A_ap.nonzero(), dtype=np.int64)), 
                           ('paper', 'to', 'subject'): torch.from_numpy(np.array(A_ps.nonzero(), dtype=np.int64)), 
                           ('subject', 'to', 'paper'): torch.from_numpy(np.array(A_sp.nonzero(), dtype=np.int64))}
        data = {'x_dict': x_dict, 'edge_index_dict': edge_index_dict, 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'ACM.pkl', 'wb') as fp:
            pickle.dump(data, fp)

def preprocess_DBLP(save_path = './data/torch_geometric/DBLP', model_name='HAN', num_train=800, num_val=400, load_label_split=True):
    # data comes from torch_geometric
    dataset = DBLP(save_path)
    data = dataset[0]
    # target node is author
    x_dict = data.x_dict
    # add one-hot feature for conference node, as done in MAGNN
    #x_dict['conference'] = torch.eye(20)
    # obtain adjacency matrix for different edge types
    edge_index_dict = data.edge_index_dict
    author_dim = x_dict['author'].shape[0]
    paper_dim = x_dict['paper'].shape[0]
    term_dim = x_dict['term'].shape[0]
    conference_dim = 20
    src, tgt = edge_index_dict[('author', 'to', 'paper')].tolist()
    A_ap = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(author_dim, paper_dim))
    
    src, tgt = edge_index_dict[('paper', 'to', 'term')].tolist()
    A_pt = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(paper_dim, term_dim))
    
    src, tgt = edge_index_dict[('paper', 'to', 'conference')].tolist()
    A_pc = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(paper_dim, conference_dim))
    
    labels = data['author'].y
    num_classes = labels.max().item() + 1
    # generate label splits, training samples: 600 (balanced), val samples: 300 (balanced), test samples: all remaining
    if load_label_split:
        with open('label_split/DBLP_label_split', 'rb') as fp:
            train_mask, val_mask, test_mask = pickle.load(fp)
    else:
        train_mask, val_mask, test_mask = data_split(x_dict['author'], labels, num_train//num_classes, num_val//num_classes)
        # save label split
        try:
            os.mkdir('label_split')
        except:
            print('folder created!')
        with open('label_split/DBLP_label_split', 'wb') as fp:
            pickle.dump([train_mask, val_mask, test_mask], fp)
    
    # obtain feature list. the paper and conference features are the same as used in GTN, except using row normalization
    features_author = x_dict['author'].numpy()
        
    norm_factor = 1/np.array(A_ap.sum(axis=0), dtype=np.float32).flatten()
    A_pa_norm = scipy.sparse.diags(norm_factor).dot(A_ap.T)
    features_paper = A_pa_norm.dot(features_author).astype(np.float32)
    x_dict['paper'] = torch.from_numpy(features_paper)
        
    #term features comes from MAGNN
    features_term = x_dict['term'].numpy()
    
    norm_factor = 1/np.array(A_pc.sum(axis=0), dtype=np.float32).flatten()
    A_cp_norm = scipy.sparse.diags(norm_factor).dot(A_pc.T)
    features_conference = A_cp_norm.dot(features_paper).astype(np.float32)
    x_dict['conference'] = torch.from_numpy(features_conference)
        
    if model_name == 'HAN':
        '''
        Prepare input data for HAN model (DGL version), which requires:
            1. g: list(dglGraph), each graph corresponds to one metapath
            2. node_features: Tensor of size N_author x F
            3. labels: Tensor of size N_paper
            4. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        # build APA, APTPA and APCPA metapaths
        A_apa = A_ap.dot(A_ap.T)
        A_apa.data = np.ones(A_apa.data.shape, dtype=np.float32)
        A_aptpa = A_ap.dot(A_pt).dot(A_pt.T).dot(A_ap.T)
        A_aptpa.data = np.ones(A_aptpa.data.shape, dtype=np.float32)
        A_apcpa = A_ap.dot(A_pc).dot(A_pc.T).dot(A_ap.T)
        A_apcpa.data = np.ones(A_apcpa.data.shape, dtype=np.float32)
        
        # create g
        g0 = dgl.from_scipy(A_apa)
        g1 = dgl.from_scipy(A_aptpa)
        g2 = dgl.from_scipy(A_apcpa)
        g = [g0, g1, g2]
        data = {'g': g, 'features': x_dict['author'], 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'DBLP_HAN.pkl', 'wb') as fp:
            pickle.dump(data, fp)
        #return g, features_paper, labels, train_mask, val_mask, test_mask
    
    elif model_name == 'GTN':
        '''
        Prepare input data for GTN model (https://github.com/seongjunyun/Graph_Transformer_Networks), which requires:
            1. node_features: Numpy array of size (N_author + N_paper + N_term + N_conference) x F,
                here concatenate nodes of all types [paper, author, subject] (in order) 
            2. edges: list of scipy.sparse adj matrices [PA, AP, PS, SP]
            3. labels: Numpy array of size N_paper x 2, each element is  np.array([label_idx, label])
            
            Note: the author feature is the mean of author's paper features.
        '''
        # obtain author, subject feature
        # features_author = x_dict['author'].numpy()
        
        # norm_factor = 1/np.array(A_ap.sum(axis=0), dtype=np.float32).flatten()
        # A_pa_norm = scipy.sparse.diags(norm_factor).dot(A_ap.T)
        # features_paper = A_pa_norm.dot(features_author)
        
        # norm_factor = 1/np.array(A_pc.sum(axis=0), dtype=np.float32).flatten()
        # A_cp_norm = scipy.sparse.diags(norm_factor).dot(A_pc.T)
        # features_conference = A_cp_norm.dot(features_paper)
        
        # concatenate features
        node_features = np.vstack((features_author, features_paper, features_conference))
        
        # edges: [PA, AP, PC, CP]
        N = node_features.shape[0]
        src, tgt = A_ap.nonzero()
        tgt = tgt + author_dim
        A_ap = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))

        src, tgt = A_pc.nonzero()
        src = src + author_dim
        tgt = tgt + author_dim + paper_dim
        A_pc = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))
        
        edges = [A_ap.T, A_ap, A_pc, A_pc.T]
        
        # labels
        train_label = np.vstack((train_mask, labels[train_mask].numpy())).T
        val_label = np.vstack((val_mask, labels[val_mask].numpy())).T
        test_label = np.vstack((test_mask, labels[test_mask].numpy())).T
        GTN_labels = [train_label, val_label, test_label]
        
        # save data
        save_folder = 'GTN/data/DBLP/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'node_features.pkl', 'wb') as fp:
            pickle.dump(node_features, fp)
        with open(save_folder + 'edges.pkl', 'wb') as fp:
            pickle.dump(edges, fp)
        with open(save_folder + 'labels.pkl', 'wb') as fp:
            pickle.dump(GTN_labels, fp)
        #return node_features, edges, GTN_labels
    
    elif model_name == 'MAGNN':
        '''
        Prepare input data for MAGNN model (https://github.com/cynricfu/MAGNN), which requires:
            1. nx_G_list: list of <num_node_type> nx_Graphs, where the ith element corresponds to the nx_Graph
                with metapath starting and ending at nodes of type i, like metapath = [{0, 1, 0}, {0, 2, 0}] for node type 0.
            2. edge_metapath_indices_list: same shape of nx_G_list. It is corresponding node indices for each edge_metapath.
            3. features_list: list of scipy.sparse matrix: [features_paper, features_author, features_subject]
            4. adjM: combined one scipy.sparse matrix with shape (N_paper + N_author + N_subject) x (N_paper + N_author + N_subject)
            5. type_mask: 1D numpy array of size (N_paper + N_author + N_subject), where type 0: paper, type 1: author, type 2: subject
            6. labels: numpy of size N_paper
            7. train_val_test_idx; [val_idx, train_idx, test_idx], each is 1D numpy array
            
            Note: for fair comparison, the author and subject feature are obtained the same way as GTN: 
            mean of author's paper features, and mean of subject's paper features.
        '''
        # Here we only update the label_split as preprocessing is out of time and out of memory, you can still try running the preprocessing 
        # by deleting the return line below
        save_prefix = 'MAGNN/data/preprocessed/DBLP_processed/'
        # target labels
        np.save(save_prefix + 'labels.npy', labels.numpy())
        return
        
        # type_mask. 0: author, 1:paper, 2: term, 3: conference
        N = author_dim + paper_dim + term_dim + conference_dim
        type_mask = np.zeros(N, dtype=int)
        type_mask[author_dim:author_dim + paper_dim] = 1
        type_mask[author_dim + paper_dim:author_dim + paper_dim + term_dim] = 2
        type_mask[author_dim + paper_dim + term_dim:] = 3
        
        # obtain adjM
        src0, tgt0 = A_ap.nonzero()
        tgt0 = tgt0 + author_dim
        
        src1, tgt1 = A_pt.nonzero()
        src1 = src1 + author_dim
        tgt1 = tgt1 + author_dim + paper_dim
        
        src2, tgt2 = A_pc.nonzero()
        src2 = src2 + author_dim
        tgt2 = tgt2 + author_dim + paper_dim + term_dim
        
        src = np.hstack((src0, src1, src2))
        tgt = np.hstack((tgt0, tgt1, tgt2))
        src, tgt = np.hstack((src, tgt)), np.hstack((tgt, src)) # symmetric matrix needed
        adjM = scipy.sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (src, tgt)), shape=(N, N))
        adjM = adjM.toarray() # dense matrix required by MAGNN
        
        # obtain nx_G_list and edge_metapath_indices_list
        expected_metapaths = [[(0, 1, 0), (0, 1, 2, 1, 0), (0, 1, 3, 1, 0)],
                              [(1, 0, 1), (1, 2, 1), (1, 3, 1)],
                              [(2, 1, 2), (2, 1, 0, 1, 2), (2, 1, 3, 1, 2)],
                              [(3, 1, 3), (3, 1, 0, 1, 3), (3, 1, 2, 1, 3)]]
        num_ntypes = 4
        save_prefix = 'MAGNN/data/preprocessed/DBLP_processed/'
        # create the directories if they do not exist
        for i in range(1):
            pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
        for i in range(1):
            t0 = time.time()
            # get metapath based neighbor pairs
            neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
            # construct and save metapath-based networks
            G_list = get_networkx_graph(neighbor_pairs, type_mask, i)
            print('time to generate metapath nodes = ', time.time() - t0)
    
        # save data
        # networkx graph (metapath specific)
        for G, metapath in zip(G_list, expected_metapaths[i]):
            nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
            # node indices of edge metapaths
        all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
        for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
            np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        # save nodes adjacency matrix
        scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
        # all nodes (authors, papers, subjects) features
        # currently only have features of authors, papers and terms
        features_author = scipy.sparse.csr_matrix(features_author)
        features_paper = scipy.sparse.csr_matrix(features_paper)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_author)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_paper)
        np.save(save_prefix + 'features_{}.npy'.format(2), features_term)
        np.save(save_prefix + 'features_{}.npy'.format(3), features_conference)
        
        # save all nodes type labels
        np.save(save_prefix + 'node_types.npy', type_mask)
        # target labels
        np.save(save_prefix + 'labels.npy', labels.numpy())
        # save train/validation/test splits
        np.savez(save_prefix + 'train_val_test_idx.npz', val_idx=val_mask, train_idx=train_mask, test_idx=test_mask)
        
    elif model_name == 'DMGI':
        '''
        Prepare input data for DMGI model (https://github.com/pcy1302/DMGI), which requires:
            A dictionary containing the following keys
                train_idx: training index, val_idx: validation index, test_idx: test index. All have shape of 1 x num_train/val/test
                feature: feature matrix (float32 or float64), label: labels, 
            Relations (float32 or float64): 
                IMDB: MDM, MAM
                DBLP: APA, APTPA, APCPA
                ACM: PAP, PSP
        '''
        # build APA, APTPA and APCPA metapaths
        A_apa = A_ap.dot(A_ap.T)
        A_apa.data = np.ones(A_apa.data.shape, dtype=np.float32)
        A_aptpa = A_ap.dot(A_pt).dot(A_pt.T).dot(A_ap.T)
        A_aptpa.data = np.ones(A_aptpa.data.shape, dtype=np.float32)
        A_apcpa = A_ap.dot(A_pc).dot(A_pc.T).dot(A_ap.T)
        A_apcpa.data = np.ones(A_apcpa.data.shape, dtype=np.float32)
        
        labels = labels.numpy()
        DMGI_labels = np.zeros((len(labels), labels.max()+1))
        DMGI_labels[np.arange(len(labels)), labels] = 1
        train_idx, val_idx, test_idx = train_mask[np.newaxis], val_mask[np.newaxis], test_mask[np.newaxis]
        data = {'label':DMGI_labels, 'feature':features_author, 'APA':A_apa.toarray(), 'APTPA':A_aptpa.toarray(), 'APCPA':A_apcpa.toarray(),
                'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}
        #data = {'label':labels, 'feature':features, 'PAP':PAP, 'PPP':PPrefP, 'PATAP':PATAP, 'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}
        # save data
        # save data, pickle
        with open('DMGI/data/dblp.pkl', 'wb') as fp:
            pickle.dump(data, fp)
    elif model_name in {'HGT', 'HetGTCN', 'HetGTAN'}:
        '''
        Prepare input data for HetGTCN, HetGTAN or HGT model (torch.geometric version: 
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HGTConv), which requires:
            1. x_dict (Dict[str, Tensor]) – A dictionary holding input node features for each individual node type
            2. edge_index_dict: A dictionary with key the edge type (tuple of source_node_type, relation, target_node_type, like 
                ('paper', 'to', 'author')), and value the edge connections, represented as a torch.LongTensor of shape [2, num_edges].
            #3. metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. 
            #    its node and edge types given by a list of strings and a list of string triplets, respectively. 
            #    For ACM, metadata = (['paper', 'author', 'subject'], 
            #    [('paper', 'to', 'author'), ('author', 'to', 'paper'), ('paper', 'to', 'subject'), ('subject', 'to', 'paper')])
            4. labels: Tensor of size N_paper
            5. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        data = {'x_dict': x_dict, 'edge_index_dict': edge_index_dict, 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'DBLP.pkl', 'wb') as fp:
            pickle.dump(data, fp)

def preprocess_IMDB(save_path = './data/torch_geometric/IMDB', model_name='HAN', num_train=300, num_val=300, load_label_split=True):
    # data comes form torch_geometric, same data as used in MAGNN
    dataset = IMDB(save_path)
    data = dataset[0]
    # target node is author
    x_dict = data.x_dict
    # obtain adjacency matrix for different edge types
    edge_index_dict = data.edge_index_dict
    movie_dim = x_dict['movie'].shape[0]
    director_dim = x_dict['director'].shape[0]
    actor_dim = x_dict['actor'].shape[0]
    src, tgt = edge_index_dict[('movie', 'to', 'director')].tolist()
    A_md = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(movie_dim, director_dim))
    
    src, tgt = edge_index_dict[('movie', 'to', 'actor')].tolist()
    A_ma = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(movie_dim, actor_dim))
    
    labels = data['movie'].y
    num_classes = labels.max().item() + 1
    # generate label splits, training samples: 300 (balanced), val samples: 300 (balanced), test samples: all remaining
    if load_label_split:
        with open('label_split/IMDB_label_split', 'rb') as fp:
            train_mask, val_mask, test_mask = pickle.load(fp)
    else:
        train_mask, val_mask, test_mask = data_split(x_dict['movie'], labels, num_train//num_classes, num_val//num_classes)
        # save label split
        try:
            os.mkdir('label_split')
        except:
            print('folder created!')
        with open('label_split/IMDB_label_split', 'wb') as fp:
            pickle.dump([train_mask, val_mask, test_mask], fp)
    
    if model_name == 'HAN':
        '''
        Prepare input data for HAN model (DGL version), which requires:
            1. g: list(dglGraph), each graph corresponds to one metapath
            2. node_features: Tensor of size N_author x F
            3. labels: Tensor of size N_paper
            4. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        # build MDM and MAM metapaths
        A_mdm = A_md.dot(A_md.T)
        A_mdm.data = np.ones(A_mdm.data.shape, dtype=np.float32)
        A_mam = A_ma.dot(A_ma.T)
        A_mam.data = np.ones(A_mam.data.shape, dtype=np.float32)
        
        # create g
        g0 = dgl.from_scipy(A_mdm)
        g1 = dgl.from_scipy(A_mam)
        g = [g0, g1]
        data = {'g': g, 'features': x_dict['movie'], 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'IMDB_HAN.pkl', 'wb') as fp:
            pickle.dump(data, fp)
        #return g, features_paper, labels, train_mask, val_mask, test_mask
    
    elif model_name == 'GTN':
        '''
        Prepare input data for GTN model (https://github.com/seongjunyun/Graph_Transformer_Networks), which requires:
            1. node_features: Numpy array of size (N_author + N_paper + N_term + N_conference) x F,
                here concatenate nodes of all types [paper, author, subject] (in order) 
            2. edges: list of scipy.sparse adj matrices [PA, AP, PS, SP]
            3. labels: Numpy array of size N_paper x 2, each element is  np.array([label_idx, label])
            
            Note: the author feature is the mean of author's paper features.
        '''
        # concatenated node features
        node_features = np.vstack((x_dict['movie'].numpy(), x_dict['director'].numpy(), x_dict['actor'].numpy()))
        
        # edges: [A_md, A_dm, A_ma, A_am]
        N = node_features.shape[0]
        src, tgt = A_md.nonzero()
        tgt = tgt + movie_dim
        A_md = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))

        src, tgt = A_ma.nonzero()
        tgt = tgt + movie_dim + director_dim
        A_ma = scipy.sparse.csr_matrix((np.ones(len(src)), (src, tgt)), shape=(N, N))
        
        edges = [A_md, A_md.T, A_ma, A_ma.T]
        
        # labels
        train_label = np.vstack((train_mask, labels[train_mask].numpy())).T
        val_label = np.vstack((val_mask, labels[val_mask].numpy())).T
        test_label = np.vstack((test_mask, labels[test_mask].numpy())).T
        GTN_labels = [train_label, val_label, test_label]
        
        # save data
        save_folder = 'GTN/data/IMDB/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'node_features.pkl', 'wb') as fp:
            pickle.dump(node_features, fp)
        with open(save_folder + 'edges.pkl', 'wb') as fp:
            pickle.dump(edges, fp)
        with open(save_folder + 'labels.pkl', 'wb') as fp:
            pickle.dump(GTN_labels, fp)
    
    elif model_name == 'MAGNN':
        '''
        Prepare input data for MAGNN model (https://github.com/cynricfu/MAGNN), which requires:
            1. nx_G_list: list of <num_node_type> nx_Graphs, where the ith element corresponds to the nx_Graph
                with metapath starting and ending at nodes of type i, like metapath = [{0, 1, 0}, {0, 2, 0}] for node type 0.
            2. edge_metapath_indices_list: same shape of nx_G_list. It is corresponding node indices for each edge_metapath.
            3. features_list: list of scipy.sparse matrix: [features_paper, features_author, features_subject]
            4. adjM: combined one scipy.sparse matrix with shape (N_paper + N_author + N_subject) x (N_paper + N_author + N_subject)
            5. type_mask: 1D numpy array of size (N_paper + N_author + N_subject), where type 0: paper, type 1: author, type 2: subject
            6. labels: numpy of size N_paper
            7. train_val_test_idx; [val_idx, train_idx, test_idx], each is 1D numpy array
            
            Note: for fair comparison, the author and subject feature are obtained the same way as GTN: 
            mean of author's paper features, and mean of subject's paper features.
        '''
        # type_mask. 0: movie, 1: director, 2: actor
        N = movie_dim + director_dim + actor_dim
        type_mask = np.zeros(N, dtype=int)
        type_mask[movie_dim:movie_dim + director_dim] = 1
        type_mask[movie_dim + director_dim:] = 2
        
        # obtain adjM
        src0, tgt0 = A_md.nonzero()
        tgt0 = tgt0 + movie_dim
        src1, tgt1 = A_ma.nonzero()
        tgt1 = tgt1 + movie_dim + director_dim
        src = np.hstack((src0, src1))
        tgt = np.hstack((tgt0, tgt1))
        src, tgt = np.hstack((src, tgt)), np.hstack((tgt, src)) # symmetric matrix needed
        adjM = scipy.sparse.csr_matrix((np.ones(len(src), dtype=np.int8), (src, tgt)), shape=(N, N))
        adjM = adjM.toarray() # dense matrix required by MAGNN
        
        # obtain nx_G_list and edge_metapath_indices_list, here only considers metapath MDM, MAM
        expected_metapaths = [[(0,1,0), (0,2,0)]]
        num_ntypes = 3
        save_prefix = 'MAGNN/data/preprocessed/IMDB_processed/'
        # create the directories if they do not exist
        for i in range(len(expected_metapaths)):
            pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)
        for i in range(len(expected_metapaths)):
            t0 = time.time()
            # get metapath based neighbor pairs
            neighbor_pairs = get_metapath_neighbor_pairs(adjM, type_mask, expected_metapaths[i])
            # construct and save metapath-based networks
            G_list = get_networkx_graph(neighbor_pairs, type_mask, i)
            print('time to generate metapath nodes = ', time.time() - t0)
    
        # save data
        # networkx graph (metapath specific)
        for G, metapath in zip(G_list, expected_metapaths[i]):
            nx.write_adjlist(G, save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist')
            # node indices of edge metapaths
        all_edge_metapath_idx_array = get_edge_metapath_idx_array(neighbor_pairs)
        for metapath, edge_metapath_idx_array in zip(expected_metapaths[i], all_edge_metapath_idx_array):
            np.save(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.npy', edge_metapath_idx_array)

        # save nodes adjacency matrix
        scipy.sparse.save_npz(save_prefix + 'adjM.npz', scipy.sparse.csr_matrix(adjM))
        # all nodes (authors, papers, subjects) features
        # currently only have features of authors, papers and terms
        features_movie = scipy.sparse.csr_matrix(x_dict['movie'].numpy())
        features_director = scipy.sparse.csr_matrix(x_dict['director'].numpy())
        features_actor = scipy.sparse.csr_matrix(x_dict['actor'].numpy())
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(0), features_movie)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(1), features_director)
        scipy.sparse.save_npz(save_prefix + 'features_{}.npz'.format(2), features_actor)
        # save all nodes type labels
        np.save(save_prefix + 'node_types.npy', type_mask)
        # target labels
        np.save(save_prefix + 'labels.npy', labels.numpy())
        # save train/validation/test splits
        np.savez(save_prefix + 'train_val_test_idx.npz', val_idx=val_mask, train_idx=train_mask, test_idx=test_mask)
        
    elif model_name == 'DMGI':
        '''
        Prepare input data for DMGI model (https://github.com/pcy1302/DMGI), which requires:
            A dictionary containing the following keys
                train_idx: training index, val_idx: validation index, test_idx: test index. All have shape of 1 x num_train/val/test
                feature: feature matrix (float32 or float64), label: labels, 
            Relations (float32 or float64): 
                IMDB: MDM, MAM
                DBLP: APA, APTPA, APCPA
                ACM: PAP, PSP
        '''
        A_mdm = A_md.dot(A_md.T)
        A_mdm.data = np.ones(A_mdm.data.shape, dtype=np.float32)
        A_mam = A_ma.dot(A_ma.T)
        A_mam.data = np.ones(A_mam.data.shape, dtype=np.float32)
        labels = labels.numpy()
        DMGI_labels = np.zeros((len(labels), labels.max()+1))
        DMGI_labels[np.arange(len(labels)), labels] = 1
        train_idx, val_idx, test_idx = train_mask[np.newaxis], val_mask[np.newaxis], test_mask[np.newaxis]
        data = {'label':DMGI_labels, 'feature':x_dict['movie'].numpy(), 'MDM':A_mdm.toarray(), 'MAM':A_mam.toarray(), 
                'train_idx':train_idx, 'val_idx':val_idx, 'test_idx':test_idx}
        # save data, pickle
        with open('DMGI/data/imdb.pkl', 'wb') as fp:
            pickle.dump(data, fp)
    elif model_name in {'HGT', 'HetGTCN', 'HetGTAN'}:
        '''
        Prepare input data for HetGTCN, HetGTAN or HGT model (torch.geometric version: 
            https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.HGTConv), which requires:
            1. x_dict (Dict[str, Tensor]) – A dictionary holding input node features for each individual node type
            2. edge_index_dict: A dictionary with key the edge type (tuple of source_node_type, relation, target_node_type, like 
                ('paper', 'to', 'author')), and value the edge connections, represented as a torch.LongTensor of shape [2, num_edges].
            #3. metadata (Tuple[List[str], List[Tuple[str, str, str]]]) – The metadata of the heterogeneous graph, i.e. 
            #    its node and edge types given by a list of strings and a list of string triplets, respectively. 
            #    For ACM, metadata = (['paper', 'author', 'subject'], 
            #    [('paper', 'to', 'author'), ('author', 'to', 'paper'), ('paper', 'to', 'subject'), ('subject', 'to', 'paper')])
            4. labels: Tensor of size N_paper
            5. (label split) train_mask, val_mask, test_mask: Tensor
        '''
        data = {'x_dict': x_dict, 'edge_index_dict': edge_index_dict, 'labels': labels, 'train_mask': torch.from_numpy(train_mask), 
                'val_mask': torch.from_numpy(val_mask), 'test_mask': torch.from_numpy(test_mask)}
        save_folder = 'data/preprocessed/'
        try:
            os.mkdir(save_folder)
        except:
            print('folder created!')
        with open(save_folder + 'IMDB.pkl', 'wb') as fp:
            pickle.dump(data, fp)