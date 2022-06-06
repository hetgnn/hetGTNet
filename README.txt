# Heterogeneous Graph Tree Neural Networks
Source code for Heterogeneous Graph Tree Convolution Network (HetGTCN) and Heterogeneous Graph Tree Attention Network (HetGTAN).

# Datasets for test
ACM, IMDB and DBLP

# Steps to run
0. Place the ACM raw data (ACM.mat from https://github.com/Jhy1993/HAN/tree/master/data/acm) into './data/ACM' folder, if ACM data is not available.
1. Data preprocessing for all test models
	run the command: python gen_preprocessed_data.py
	Note: The above command will generate corrsponding input data for different models. The preprocessing time for all models are fast except MAGNN, which will take ~1 hour for ACM and IMDB 	datasets, and cause out-of-memory (OOM) and out-of-time (OOT) for DBLP dataset. So please copy the preprocessed DBLP data from original source (https://github.com/cynricfu/MAGNN) and place it inside './MAGNN/data/preprocessed/DBLP_processed', then run python gen_preprocessed_data.py. If you don't want to test MAGNN model and don't want to generate preprocessed data for MAGNN, you can run python gen_preprocessed_data.py --skip_MAGNN

2. Start to run different models including HetGTCN and HetGTAN models.
	The test results for all models except SimpleHGN are recorded in the Jupyter files in './result' folder, you can simply open each Jupyter file and run it directly without any extra work. The model settings are clearly shown in each Jupyter notebook. The Jupyter file for HetGTCN and HetGTAN at hop=5, with f1-macro metric is named as f1-macro_HetGTCN_hop5.ipynb and f1-macro_HetGTAN_hop5.ipynb. The Jupyter file is usually formatted as {metric}_{model}_{optional hop#}.ipynb, for instance:
	f1-macro_RGCN_hop10.ipynb: test of RGCN model at hop 10 with f1-macro metric
	f1-macro_HAN_hop1.ipynb: test of HAN model with 1 layer (each layer includes different predefined metapaths), using f1-macro metric
	f1-micro_HetGAT_hop2.ipynb: test of HetGAT model at hop 2 with f1-macro metric
	f1_DMGI.ipynb: test of DMGI model with predefined metapaths (both f1-macro and f1-micro metrics included)
	f1-macro_MAGNN.ipynb: test of MAGNN model with f1-macro metric
	f1-macro_HGT.ipynb: test of HGT model with f1-macro metric
	
	You can also use the following command to run a test without Jupyter
		python train.py
    		with the optional input arguments:
			'--data', default='ACM', help='Name of dataset, can be ACM, IMDB and DBLP'
			'--data_path', default='data/preprocessed/', help='folder path of saved preprocessed data.'
			'--model', default='HetGTAN', help='Heterogeneous GNN model, choices of: 
				HAN, HetGTCN, HetGTAN, HGT, HetGCN, HetGAT, RGCN, HetGTAN_NoSem, HetGTAN_LW, HetGTAN_mean, HetGTCN_mean, HetGTCN_LW'
			'--target_node_type', default='paper', help='Target node type: paper for ACM data.'
			'--n_hid', type=int, default=64, help='num of hidden features.'
			'--num_heads', type=int, default=8, help='num heads for attention layer'
			'--dropout', type=float, default=0.8, help='Initial layer dropout, or dropout for HetGCN'
			'--dropout2', type=float, default=0.2, help='Intermediate layer dropout for HetGTAN or HetGTCN, or attn dropout for HetGAT'
			'-lr', '--learning_rate', type=float, default=0.005, help='Learning rate.'
			'-wd', '--weight_decay', type=float, default=0.00005, help='weight decay in Adam optimizer.'
			'--patience', type=int, default=100, help='Early stopping patience.'
			'--num_iter', type=int, default=500, help='Max epochs to run.'
			'--num_test', type=int, default=30, help='num of runs to test accuracy.'
			'--hop', type=int, default=5, help='hop or #layers of GNN models.'
			'--num_bases', type=int, default=5, help='num bases for RGCN model.'
			'--filter_pct', type=float, default=0.1, help='remove the top and bottom filer_pct points before obtaining statistics of test accuracy.'
			'--log_step', type=int, default=1000, help='training log step.'
			'-lw', '--layer_wise', action='store_true', default=False, help="whether to share parameters for different layers"
			'--average', default='macro', help='f1 average: can choose either macro or micro.'


	Examples: 
		1. Run HetGTAN with 5 hop for ACM dataset with f1-macro metric with 30 runs: python train.py -lw
		2. Run HetGTAN with 5 hop for IMDB dataset with f1-macro metric with 30 runs: python train.py --data IMDB --target_node_type movie -lw
		3. Run HetGTCN with 5 hop for DBLP dataset with f1-macro metric with 10 runs: python train.py --model HetGTCN --data DBLP --target_node_type author --dropout 0.8 --dropout2 0.6 -wd 1e-5 --num_test 10 -lw
		
	The running results for SimpleHGN model are recorded in Jupyter files inside './SimpleHGN' folder. For instance, f1-macro_SimpleHGN_hop2.ipynb shows the test result of SimpleHGN at hop = 2. You can simply open the Jupyter files and run it to reproduce the results. You can also run the command for SimpleHGN using python train_simpleHGN.py. If you want to run command for MAGNN, simply go to './MAGNN' folder and run python main.py. Same for GTN model (main_sparse.py in './GTN' folder) and DMGI model (main.py in './DMGI' folder)

Running hardware: RTX 3060 12GB.

## Tools used by the source code
1. Pytorch
2. Pytorch Geometric
3. DGL
4. Scikit-Learn
5. cogdl (for SimpleHGN test)
6. Imblearn 0.8.0 (only used for label split)
