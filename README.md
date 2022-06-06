# Heterogeneous Graph Tree Neural Networks
This repository contains the codes and data to replicate the experimental results in the manuscript Heterogeneous Graph Tree Networks.

## Datasets for test
ACM, IMDB, and DBLP

## Models
1. HetGTCN
2. HetGTAN
3. HetGCN
4. HetGAT
5. [GCN](https://arxiv.org/pdf/1609.02907.pdf%EF%BC%89)
6. [GAT](https://arxiv.org/pdf/1710.10903.pdf)
7. [RGCN](https://arxiv.org/pdf/1703.06103.pdf?ref=https://githubhelp.com)
8. [HAN](https://arxiv.org/pdf/1903.07293.pdf?ref=https://githubhelp.com)
9. [MAGNN](https://arxiv.org/pdf/2002.01680.pdf)
10. [GTN](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)
11. [HGT](https://dl.acm.org/doi/pdf/10.1145/3366423.3380027)
12. [SimpleHGN](https://arxiv.org/pdf/2112.14936.pdf)
13. [DMGI](https://ojs.aaai.org/index.php/AAAI/article/view/5985)


## Running hardware
RTX 3060 12GB

## Tools
1. Pytorch
2. Pytorch Geometric
3. DGL
4. Scikit-Learn
5. cogdl (used for implementation of SimpleHGN)
6. Imblearn 0.8.0 (only used for label split)

## Steps to run
1. Place the ACM raw data (`ACM.mat` from <https://github.com/Jhy1993/HAN/tree/master/data/acm>) into `./data/ACM` folder, if ACM data is not available.
2. Data preprocessing for all test models.
   - Run the command: `python gen_preprocessed_data.py`
   Note: The above command will generate corrsponding input data for different models. The preprocessing time for all models are fast except MAGNN, which takes about one hour for the ACM and IMDB datasets, and causes out-of-memory (OOM) and out-of-time (OOT) issue for DBLP dataset. So we use the preprocessed DBLP dataset from the original source (<https://github.com/cynricfu/MAGNN>) and place it inside the folder `./MAGNN/data/preprocessed/DBLP_processed`, then run `python gen_preprocessed_data.py`. If you don't want to test MAGNN model and don't want to generate preprocessed data for MAGNN, you can run `python gen_preprocessed_data.py --skip_MAGNN`.
3. Run baseline models and the proposed HetGTCN and HetGTAN models.
   - The test results of all models except for SimpleHGN are recorded in the Jupyter files located in the folder `./result`. The file name is formatted as `{metric}_{model}_{optional hop#}.ipynb`. For example, the Macro-F1 scores of **HetGTCN** and **HetGTAN** with 5 model layers are named as `f1-macro_HetGTCN_hop5.ipynb` and `f1-macro_HetGTAN_hop5.ipynb`, respectively. You can simply open the a Jupyter file in the results folder and run it directly without any extra work. The model settings are clearly described in each Jupyter notebook.
   - You can also use the following command to run a test without using Jupyter Notebook.
       - `python train.py` with the optional input arguments:
          - `--data`, default = `ACM`, help = `Name of dataset, can be ACM, IMDB and DBLP`
          - `--data_path`, default = `data/preprocessed/`, help = `folder path of saved preprocessed data`
          - `--model`, default = `HetGTAN`, help = `Heterogeneous GNN model, choices of: HAN, HetGTCN, HetGTAN, HGT, HetGCN, HetGAT, RGCN, HetGTAN_NoSem, HetGTAN_LW, HetGTAN_mean, HetGTCN_mean, HetGTCN_LW`
          - `--target_node_type`, default = `paper`, help = `Target node type: paper for ACM data`
          - `--n_hid`, type = `int`, default = `64`, help = `num of hidden features`
          - `--num_heads`, type = `int`, default = `8`, help = `num heads for attention layer`
          - `--dropout`, type = `float`, default = `0.8`, help = `Initial layer dropout, or dropout for HetGCN`
          - `--dropout2`, type = `float`, default = `0.2`, help = `Intermediate layer dropout for HetGTAN or HetGTCN, or attention dropout for HetGAT`
          - `-lr`, `--learning_rate`, type = `float`, default = `0.005`, help = `Learning rate`
          - `-wd`, `--weight_decay`, type = `float`, default = `0.00005`, help = `weight decay in Adam optimizer`
          - `--patience`, type = `int`, default = `100`, help = `Early stopping patience`
          - `--num_iter`, type = `int`, default = `500`, help = `Max epochs to run`
          - `--num_test`, type = `int`, default = `30`, help = 'num of runs to test accuracy`
          - `--hop`, type = `int`, default = `5`, help = `hop or #layers of GNN models`
          - `--num_bases`, type = `int`, default = `5`, help = `num bases for RGCN model`
          - `--filter_pct`, type = `float`, default= `0.1`, help = `remove the top and bottom filer_pct points before obtaining statistics of test accuracy`
          - `--log_step`, type = `int`, default = `1000`, help = `training log step`
          - `-lw`, `--layer_wise`, action = `store_true`, default = `False`, help = `whether to share parameters for different layers`
          - `--average`, default = `macro`, help = `f1 average: can choose either macro or micro`
   - Examples:
       1. Use the following command to run a five-layer HetGTAN on the ACM dataset with F1-macro metric 30 times: `python train.py -lw`
       2. Use the following command to run a five-layer HetGTAN on the IMDB dataset with F1-macro metric 30 times: `python train.py --data IMDB --target_node_type movie -lw`
       3. Use the following command to run a five-layer HetGTCN on the DBLP dataset with F1-macro metric 10 times: `python train.py --model HetGTCN --data DBLP --target_node_type author --dropout 0.8 --dropout2 0.6 -wd 1e-5 --num_test 10 -lw`
   - The results for the SimpleHGN model are recorded as Jupyter files located in the folder `./SimpleHGN`. For instance, `f1-macro_SimpleHGN_hop2.ipynb` contains the test result of a two-layer SimpleHGN model. You can simply open the Jupyter files and run it to reproduce our results. You can also run the following command `python train_simpleHGN.py` for the same purpose. 
   - If you want to run command for MAGNN, simply go to the folder `./MAGNN` and run `python main.py`. 
   - If you want to run command for GTN, simply go to the folder `./GTN` and run `python main_sparse.py`. 
   - If you want to run command for DMGI, simple go to the folder `./DMGI`, and run `python main.py`.
      
