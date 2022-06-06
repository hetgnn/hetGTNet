from utils import preprocess_ACM, preprocess_IMDB, preprocess_DBLP
import time
import argparse
'''
All preprocessed ACM data will be generated from the original ACM.mat file, located at './data/ACM/ACM.mat'

All preprocessed IMDB data will be generated from IMDB data in torch_geometric.datasets, 
where original torch_geometric IMDB data will be saved at './data/torch_geometric/IMDB' (default)

All preprocessed DBLP data will be generated from DBLP data in torch_geometric.datasets, 
where original torch_geometric DBLP data will be saved at './data/torch_geometric/DBLP' (default)

1. Preprocessed data for HAN
    The preprocessed data for HAN model will be saved at: './data/preprocessed/{data_name}_HAN.pkl', where data_name = ACM, IMDB and DBLP
    The saved data for HAN is a dictionary with keys: 'g', 'features', 'labels', 'train_mask', 'val_mask', 'test_mask', where
        g: list(dglGraph), each graph corresponds to one metapath
        features: target node features, a tensor of size N x F
        labels: target node labels, a tensor of size N
        train_mask: train indices, a tensor of size N_train  
        val_mask: val indices, a tensor of size N_val
        test_mask: test indices, a tensor of size N_test

2. Preprocessed data for GTN
    The preprocessed data for GTN model will be saved in the folder './GTN/data/{data_name}/', where data_name = ACM, IMDB and DBLP.
    Inside each folder, there generated 3 file (node_features.pkl, edges.pkl and labels.pkl) which can be directly used by the original GTN code 
    The format of node_features.pkl, edges.pkl and labels.pkl are:
        node_features: Numpy array of size N x F, here GTN concatenate nodes of all types (in order) 
        edges: list of scipy.sparse adj matrices for the whole concatenated graph
        labels: Numpy array of size N_target x 2, each element is  np.array([label_idx, label])
    Details of order of nodes concatenation (target node will always be the first), as well as edges are:
        ACM: node concat order of [paper, author, subject], edges = [PA, AP, PS, SP]
        IMDB: node concat order of [movie, director, actor], edges = [MD, DM, MA, AM]
        DBLP: node concat order of [author, paper, conference], edges = [PA, AP, PC, CP]
        
3. Preprocessed data for MAGNN
    The preprocessed data for MAGNN model will be saved in the folder './MAGNN/data/preprocessed/{data_name}_processed/', where data_name = ACM, IMDB and DBLP
    Inside each folder, there generated some files (metapath instance folder, adjM.npz, features_.npz, labels.npy, node_types.npy, 
        train_val_test_idx.npz) which can be directly used by the original MAGNN code
    Note that it is out of time and out of memory to preprocess the DBLP dataset, therefore, we use the orignal preprocessed data from MAGNN, 
        except that we use a different label split (this label split is the same for all the models).

4. Preprocessed data for DMGI
    The preprocessed data for DMGI model will be saved at: './DMGI/data/acm.mat', './DMGI/data/imdb.pkl' and './DMGI/data/dblp.pkl', 
        which can be used directly by orignal DMGI code.
    The saved data for DMGI is a dictionary with keys: 'label', 'feature', 'metapath_name', 'train_idx', 'val_idx', 'test_idx', where
        label: target node labels, a tensor of size N
        feature: target node features, a tensor of size N x F
        metapaths used: 
                IMDB: MDM, MAM
                DBLP: PAP, PPP, PATAP
                ACM: PAP, PSP
        train_idx: train indices, a tensor of size N_train  
        val_idx: val indices, a tensor of size N_val
        test_idx: test indices, a tensor of size N_test

5. Preprocessed data for HetGTCN, HetGTAN, HGT, HetGCN and HetGAT  
    The preprocessed data have the same standard format, located at: './data/preprocessed/{data_name}.pkl', where data_name = ACM, IMDB and DBLP
    The saved data is a dictionary with keys: 'x_dict', 'edge_index_dict', 'labels', 'train_mask', 'val_mask', 'test_mask', where
        x_dict (Dict[str, Tensor]) – A dictionary holding input node features for each individual node type
        edge_index_dict: A dictionary with key the edge type (tuple of source_node_type, relation, target_node_type, like 
                ('paper', 'to', 'author')), and value the edge connection indices, represented as a torch.LongTensor of shape [2, num_edges].
        labels: Tensor of size N_paper
        train_mask: train indices, a tensor of size N_train  
        val_mask: val indices, a tensor of size N_val
        test_mask: test indices, a tensor of size N_test

Note that the label split was generated once (located in './label_split/ folder'), and will then be used by all models.
Note that SimpleHGN will use the preprocessed data from GTN, 
homogeneous GCN and GAT will use the preprocessed data from HetGTCN with slight modification (see source code).
'''
def main(args):
    # begin generating all preprocessed data for ACM dataset (assuming label split was generated).
    # if there is no label split, then set load_label_split=False once for any one model, then set load_label_split=True for all other models
    # MAGNN takes very long time for data 
    models = ['HAN', 'GTN', 'DMGI', 'HetGTAN']
    if not args.skip_MAGNN:
        models.append('MAGNN')
        
    # ACM preprocessed data for all models 
    ACM_raw_path = './data/ACM/ACM.mat'
    for model_name in models:
        tic = time.time()
        preprocess_ACM(path=ACM_raw_path, model_name=model_name, load_label_split=True)
        print('ACM preprocessing time for {} is {:.3f}'.format(model_name, time.time() - tic))

    # IMDB preprocessed data for all models 
    IMDB_raw_path = './data/torch_geometric/IMDB'
    for model_name in models:
        tic = time.time()
        preprocess_IMDB(save_path=IMDB_raw_path, model_name=model_name, load_label_split=True)
        print('IMDB preprocessing time for {} is {:.3f}'.format(model_name, time.time() - tic))

    # DBLP preprocessed data for all models 
    DBLP_raw_path = './data/torch_geometric/DBLP'
    # note that MAGNN preprocessing is out of time and out of memory for DBLP dataset
    # please use the original preprocessed data from MAGNN, and only update the label split (labels.npy)
    for model_name in models:
        tic = time.time()
        preprocess_DBLP(save_path=DBLP_raw_path, model_name=model_name, load_label_split=True)
        print('DBLP preprocessing time for {} is {:.3f}'.format(model_name, time.time() - tic))

if __name__ == "__main__":
    """
        Heterogeneous GNN Model Hyperparameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_MAGNN', action='store_true', default=False, help="whether to skip preprocessing MAGNN data")
    args = parser.parse_args()
    f1s = main(args)