import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, RGCNConv
from layers import SemanticAttention, HANLayer, HGTLayer, Learnable_Weight
from torch_scatter import scatter

class HetGTCN(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=5, layerwise=True):
        super(HetGTCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not self.layerwise:
            self.semantic_attention = nn.ModuleDict()
            for node_type in n_in_dict:
                self.semantic_attention[node_type] = SemanticAttention(n_hid)
        else:
            self.semantic_attention = nn.ModuleList()
            for _ in range(hop):
                sem_att = nn.ModuleDict()
                for node_type in n_in_dict:
                    sem_att[node_type] = SemanticAttention(n_hid)
                self.semantic_attention.append(sem_att)
        # self.reset_parameters()
    def forward(self, x_dict_input, A1_dict, A2_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        A1_dict: dict of dict of sparse adj matrix (no diag), e.g. A1_dict[node_type] = {node_type1: a1, node_type2: a2}
        a1 is the sparse adj matrix for edge type (node_type, node_type1)
        A2_dict: dict of dict of diag matrix tensor, e.g. A2_dict[node_type] = {node_type1: d1, node_type2: d2}
        d1 is the diag matrix tensor for node type, with shape Nc x 1
        '''
        x_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
        h_dict = x_dict.copy()
        for i in range(self.hop):
            if not self.layerwise:
                sem_attn = self.semantic_attention
            else:
                sem_attn = self.semantic_attention[i]
            # update node feature by type
            for node_type, x in x_dict.items():
                out = []
                for node_type2 in A1_dict[node_type]:
                    A1, A2 = A1_dict[node_type][node_type2], A2_dict[node_type][node_type2]
                    h = torch.sparse.mm(A1, h_dict[node_type2]) + A2 * x
                    out.append(h)
                # semantic attention
                out = torch.stack(out, dim=1)
                h = sem_attn[node_type](out)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

class HetGTAN(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=5, layerwise=True):
        super(HetGTAN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not self.layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
            self.semantic_attention = nn.ModuleDict()
            for node_type in n_in_dict:
                self.semantic_attention[node_type] = SemanticAttention(n_hid)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
            self.semantic_attention = nn.ModuleList()
            for _ in range(hop):
                sem_att = nn.ModuleDict()
                for node_type in n_in_dict:
                    sem_att[node_type] = SemanticAttention(n_hid)
                self.semantic_attention.append(sem_att)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        semantic_embeddings = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
            semantic_embeddings[node_type] = []
        h_dict = x_dict.copy()
        
        for i in range(self.hop):
            if not self.layerwise:
                attn1 = self.attn1
                attn2 = self.attn2
                sem_attn = self.semantic_attention
            else:
                attn1 = self.attn1[i]
                attn2 = self.attn2[i]
                sem_attn = self.semantic_attention[i]
            for node_type in semantic_embeddings:
                semantic_embeddings[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], h_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                semantic_embeddings[node_type_s].append(h)
            # semantic attention
            for node_type in semantic_embeddings:
                h = sem_attn[node_type](torch.stack(semantic_embeddings[node_type], dim=1))
                h = self.elu(h)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)
                

class HAN(nn.Module):
    '''
    HAN model from DGL: https://github.com/dmlc/dgl/blob/master/examples/pytorch/han/model.py
    '''
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        return self.predict(h)

class HGT(nn.Module):
    '''
    HGT model from DGL: https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py
    '''
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp[t], n_hid))
        # for ntype in node_dict:
        #     n_id = node_dict[ntype]
        #     self.adapt_ws[n_id][n_id] = nn.Linear(n_inp[ntype], n_hid)
        # self.adapt_ws = nn.ModuleList(self.adapt_ws)
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])
    
class HetGCN(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, dropout=0.5, hop=2, layerwise=True):
        super(HetGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fcs = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(hop)])
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not self.layerwise:
            self.semantic_attention = nn.ModuleDict()
            for node_type in n_in_dict:
                self.semantic_attention[node_type] = SemanticAttention(n_hid)
        else:
            self.semantic_attention = nn.ModuleList()
            for _ in range(hop):
                sem_att = nn.ModuleDict()
                for node_type in n_in_dict:
                    sem_att[node_type] = SemanticAttention(n_hid)
                self.semantic_attention.append(sem_att)
        #self.reset_parameters()
    def forward(self, x_dict_input, A1_dict, A2_dict, target_node_type):
        x_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
        
        for i in range(self.hop):
            if not self.layerwise:
                sem_attn = self.semantic_attention
            else:
                sem_attn = self.semantic_attention[i]
                
            # linear transformation
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.fcs[i](x)
            # update node feature by type
            for node_type, x in x_dict.items():
                out = []
                for node_type2 in A1_dict[node_type]:
                    A1, A2 = A1_dict[node_type][node_type2], A2_dict[node_type][node_type2]
                    h = torch.sparse.mm(A1, x_dict[node_type2]) + A2 * x
                    out.append(h)
                # semantic attention
                out = torch.stack(out, dim=1)
                h = sem_attn[node_type](out)
                h = self.relu(h)
                h = self.dropout(h)
                x_dict[node_type] = h
        h = self.fc2(x_dict[target_node_type])
        return h
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        for i in range(len(self.fcs)):
            nn.init.xavier_uniform_(self.fcs[i].weight, gain=1)
        
class HetGAT(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=2, layerwise=True):
        super(HetGAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fcs = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(hop)])
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not self.layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
            self.semantic_attention = nn.ModuleDict()
            for node_type in n_in_dict:
                self.semantic_attention[node_type] = SemanticAttention(n_hid)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
            self.semantic_attention = nn.ModuleList()
            for _ in range(hop):
                sem_att = nn.ModuleDict()
                for node_type in n_in_dict:
                    sem_att[node_type] = SemanticAttention(n_hid)
                self.semantic_attention.append(sem_att)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        semantic_embeddings = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
            semantic_embeddings[node_type] = []
        
        for i in range(self.hop):
            if not self.layerwise:
                attn1 = self.attn1
                attn2 = self.attn2
                sem_attn = self.semantic_attention
            else:
                attn1 = self.attn1[i]
                attn2 = self.attn2[i]
                sem_attn = self.semantic_attention[i]
            # linear transformation
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.fcs[i](x)
            for node_type in semantic_embeddings:
                semantic_embeddings[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], x_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                semantic_embeddings[node_type_s].append(h)
            # semantic attention
            for node_type in semantic_embeddings:
                h = sem_attn[node_type](torch.stack(semantic_embeddings[node_type], dim=1))
                h = self.elu(h)
                h = self.dropout2(h)
                x_dict[node_type] = h
        h = self.fc2(x_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)

class RGCN(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, num_relations, num_bases, dropout=0.5, hop=10):
        super(RGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.layers = nn.ModuleList()
        for i in range(hop - 1):
            self.layers.append(RGCNConv(n_hid, n_hid, num_relations, num_bases))
        self.layers.append(RGCNConv(n_hid, n_out, num_relations, num_bases))
    def forward(self, x_dict_input, node_type_order, edge_index, edge_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        node_type_order: list of node types that consist of the one big het. graph
        edge_index: shape of 2 x E
        edge_type: edge type ids, shape of E
        '''
        x = []
        for node_type in node_type_order:
            x1 = x_dict_input[node_type]
            x1 = self.fc1[node_type](x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)
            x.append(x1)
        x = torch.cat(x)

        for i in range(self.hop - 1):
            x = self.layers[i](x, edge_index, edge_type)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index, edge_type)
        return x

# below are other models for different tests    
class HetGTAN_LW(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=5, layerwise=True):
        super(HetGTAN_LW, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.layerwise = layerwise
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
            self.LW = Learnable_Weight(len(edge_types))
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
            self.LW = nn.ModuleList()
            for _ in range(hop):
                self.LW.append(Learnable_Weight(len(edge_types)))
        self.edge_id = {}
        for edge_type in edge_types:
            e_type = '_'.join(edge_type)
            self.edge_id[e_type] = len(self.edge_id)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        edge_aggr = {}
        n_id = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
            edge_aggr[node_type] = []
            n_id[node_type] = []
        h_dict = x_dict.copy()
        for i in range(self.hop):
            if not self.layerwise:
                attn1, attn2 = self.attn1, self.attn2
                LW = self.LW
            else:
                attn1, attn2 = self.attn1[i], self.attn2[i]
                LW = self.LW[i]
            for node_type in edge_aggr:
                edge_aggr[node_type] = []
                n_id[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], h_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                edge_aggr[node_type_s].append(h)
                n_id[node_type_s].append(self.edge_id[e_type])
                    
            # semantic attention
            for node_type in edge_aggr:
                h = LW(torch.stack(edge_aggr[node_type], dim=1), n_id[node_type])
                h = self.elu(h)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)

class HetGTCN_mean(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, dropout=0.5, dropout2=0.5, hop=5):
        super(HetGTCN_mean, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        # self.reset_parameters()
    def forward(self, x_dict_input, A1_dict, A2_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        A1_dict: dict of dict of sparse adj matrix (no diag), e.g. A1_dict[node_type] = {node_type1: a1, node_type2: a2}
        a1 is the sparse adj matrix for edge type (node_type, node_type1)
        A2_dict: dict of dict of diag matrix tensor, e.g. A2_dict[node_type] = {node_type1: d1, node_type2: d2}
        d1 is the diag matrix tensor for node type, with shape Nc x 1
        '''
        x_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
        h_dict = x_dict.copy()
        for i in range(self.hop):
            # update node feature by type
            for node_type, x in x_dict.items():
                out = []
                for node_type2 in A1_dict[node_type]:
                    A1, A2 = A1_dict[node_type][node_type2], A2_dict[node_type][node_type2]
                    h = torch.sparse.mm(A1, h_dict[node_type2]) + A2 * x
                    out.append(h)
                h = torch.mean(torch.stack(out, dim=0), dim=0)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

class HetGTCN_LW(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, dropout2=0.5, hop=5, layerwise=True):
        super(HetGTCN_LW, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not self.layerwise:
            self.LW = Learnable_Weight(len(edge_types))
        else:
            self.LW = nn.ModuleList()
            for _ in range(hop):
                self.LW.append(Learnable_Weight(len(edge_types)))
        self.edge_id = {}
        for edge_type in edge_types:
            e_type = '_'.join(edge_type)
            self.edge_id[e_type] = len(self.edge_id)
        # self.reset_parameters()
    def forward(self, x_dict_input, A1_dict, A2_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        A1_dict: dict of dict of sparse adj matrix (no diag), e.g. A1_dict[node_type] = {node_type1: a1, node_type2: a2}
        a1 is the sparse adj matrix for edge type (node_type, node_type1)
        A2_dict: dict of dict of diag matrix tensor, e.g. A2_dict[node_type] = {node_type1: d1, node_type2: d2}
        d1 is the diag matrix tensor for node type, with shape Nc x 1
        '''
        x_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
        h_dict = x_dict.copy()
        for i in range(self.hop):
            if not self.layerwise:
                LW = self.LW
            else:
                LW = self.LW[i]
            # update node feature by type
            for node_type, x in x_dict.items():
                out = []
                n_id = []
                for node_type2 in A1_dict[node_type]:
                    A1, A2 = A1_dict[node_type][node_type2], A2_dict[node_type][node_type2]
                    h = torch.sparse.mm(A1, h_dict[node_type2]) + A2 * x
                    out.append(h)
                    e_type = node_type + '_to_' + node_type2
                    n_id.append(self.edge_id[e_type])
                # semantic attention
                out = torch.stack(out, dim=1)
                h = LW(out, n_id)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        
class HetGTAN_mean(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=5, layerwise=True):
        super(HetGTAN_mean, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.layerwise = layerwise
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
        self.edge_id = {}
        for edge_type in edge_types:
            e_type = '_'.join(edge_type)
            self.edge_id[e_type] = len(self.edge_id)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        edge_aggr = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
            edge_aggr[node_type] = []
        h_dict = x_dict.copy()
        for i in range(self.hop):
            if not self.layerwise:
                attn1, attn2 = self.attn1, self.attn2
            else:
                attn1, attn2 = self.attn1[i], self.attn2[i]
            for node_type in edge_aggr:
                edge_aggr[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], h_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                edge_aggr[node_type_s].append(h)
                    
            # semantic attention
            for node_type in edge_aggr:
                h = torch.mean(torch.stack(edge_aggr[node_type], dim=0), dim=0)
                h = self.elu(h)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)

class HetGTAN_NoSem(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=5, layerwise=True):
        super(HetGTAN_NoSem, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.layerwise = layerwise
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        edge_aggr = {}
        div_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x
            edge_aggr[node_type] = []
            #div_dict[node_type] = []
        h_dict = x_dict.copy()
        for i in range(self.hop):
            if not self.layerwise:
                attn1, attn2 = self.attn1, self.attn2
            else:
                attn1, attn2 = self.attn1[i], self.attn2[i]
            for node_type in edge_aggr:
                edge_aggr[node_type] = []
                div_dict[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], h_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                # h = h/div
                edge_aggr[node_type_s].append(h)
                div_dict[node_type_s].append(div)
                    
            # semantic attention
            for node_type in edge_aggr:
                #h = torch.mean(torch.stack(edge_aggr[node_type], dim=0), dim=0)
                h = torch.stack(edge_aggr[node_type], dim=0)
                h = torch.sum(h, dim=0)
                div = torch.stack(div_dict[node_type], dim=0)
                div = torch.sum(div, dim=0)
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
                h_dict[node_type] = h
        h = self.fc2(h_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)
            
class GCN(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, dropout=0.5, hop=2):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hop = hop
        self.relu = nn.ReLU()
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fcs = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(hop - 1)])
        self.fc2 = nn.Linear(n_hid, n_out)
        #self.reset_parameters()
    def forward(self, x_dict_input, g, node_type_order):
        A = g['A']
        x = []
        for node_type in node_type_order:
            x1 = x_dict_input[node_type]
            x1 = self.fc1[node_type](x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)
            x.append(x1)
        x = torch.cat(x)
        for i in range(self.hop - 1):
            x = self.fcs[i](x)
            x = torch.sparse.mm(A, x)
            x = self.relu(x)
            x = self.dropout(x)
        # last layer
        x = self.fc2(x)
        x = torch.sparse.mm(A, x)
        return x
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        for i in range(len(self.fcs)):
            nn.init.xavier_uniform_(self.fcs[i].weight, gain=1)

class GAT(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, dropout=0.5, attn_dropout=0.5, hop=2, num_heads=1, num_out_heads=1, alpha=0.2):
        super(GAT, self).__init__()
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.fc1 = nn.ModuleDict()
        # initial layer
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
            
        self.layers = nn.ModuleList()
        n_in = n_hid
        for i in range(hop - 1):
            self.layers.append(GATConv(n_in, n_hid, num_heads, concat=True, negative_slope=alpha, dropout=attn_dropout))
            n_in = n_hid * num_heads
        self.layers.append(GATConv(n_in, n_out, num_out_heads, concat=False, negative_slope=alpha, dropout=attn_dropout))
        # self.reset_parameters()
    def forward(self, x_dict_input, g, node_type_order):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        '''
        edge_index = g['edge_index']
        x = []
        for node_type in node_type_order:
            x1 = x_dict_input[node_type]
            x1 = self.fc1[node_type](x1)
            x1 = self.relu(x1)
            x1 = self.dropout(x1)
            x.append(x1)
        x = torch.cat(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x
    def reset_parameters(self):
        for node_type in self.fc1:
            nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))

class HetGAT_NoSem(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=2, layerwise=True):
        super(HetGAT_NoSem, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fcs = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(hop)])
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        edge_aggr = {}
        div_dict = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x

        for i in range(self.hop):
            if not self.layerwise:
                attn1, attn2 = self.attn1, self.attn2
            else:
                attn1, attn2 = self.attn1[i], self.attn2[i]
            # linear transformation
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.fcs[i](x)
                edge_aggr[node_type] = []
                div_dict[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], x_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                #h = h/div
                edge_aggr[node_type_s].append(h)
                div_dict[node_type_s].append(div)
            # aggregate over all edge types
            for node_type in edge_aggr:
                h = torch.stack(edge_aggr[node_type], dim=0)
                h = torch.sum(h, dim=0)
                div = torch.stack(div_dict[node_type], dim=0)
                div = torch.sum(div, dim=0)
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
                x_dict[node_type] = h
        h = self.fc2(x_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)

class HetGAT_mean(nn.Module):
    def __init__(self, n_in_dict, n_hid, n_out, edge_types, dropout=0.5, attn_dropout=0.5, hop=2, layerwise=True):
        super(HetGAT_mean, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.n_hid = n_hid
        self.hop = hop
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.edge_types = edge_types
        self.fc1 = nn.ModuleDict()
        for node_type in n_in_dict:
            self.fc1[node_type] = nn.Linear(n_in_dict[node_type], n_hid)
        self.fcs = nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(hop)])
        self.fc2 = nn.Linear(n_hid, n_out)
        self.layerwise = layerwise
        if not layerwise:
            self.attn1, self.attn2 = nn.ModuleDict(), nn.ModuleDict()
            for edge_type in edge_types:
                self.attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
        else:
            self.attn1, self.attn2 = nn.ModuleList(), nn.ModuleList()
            for _ in range(hop):
                attn1, attn2 = nn.ModuleDict(), nn.ModuleDict()
                for edge_type in edge_types:
                    attn1['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                    attn2['_'.join(edge_type)] = nn.Linear(n_hid, 1, bias=False)
                self.attn1.append(attn1)
                self.attn2.append(attn2)
        self.reset_parameters()
    def forward(self, x_dict_input, edge_index_dict, target_node_type):
        '''
        x: dict of tensor, x_dict[node_type]: feature_tensor
        edge_index_dict: dict of edge_index with shape of 2 x E
        '''
        x_dict = {}
        edge_aggr = {}
        for node_type, x in x_dict_input.items():
            x = self.dropout(x)
            x = self.fc1[node_type](x)
            x = self.relu(x)
            x = self.dropout(x)
            x_dict[node_type] = x

        for i in range(self.hop):
            if not self.layerwise:
                attn1, attn2 = self.attn1, self.attn2
            else:
                attn1, attn2 = self.attn1[i], self.attn2[i]
            # linear transformation
            for node_type, x in x_dict.items():
                x_dict[node_type] = self.fcs[i](x)
                edge_aggr[node_type] = []
            for edge_type in self.edge_types:
                s, t = edge_index_dict[edge_type]
                node_type_s, node_type_t = edge_type[0], edge_type[-1]
                x, h = x_dict[node_type_s], x_dict[node_type_t]
                N = x.size(0)
                e_type = '_'.join(edge_type)
                x1 = attn1[e_type](x)
                h1 = attn2[e_type](h)
                w2 = x1 + attn2[e_type](x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                edge_aggr[node_type_s].append(h)
            # aggregate over all edge types
            for node_type in edge_aggr:
                h = torch.mean(torch.stack(edge_aggr[node_type], dim=0), dim=0)
                h = self.elu(h)
                h = self.dropout2(h)
                x_dict[node_type] = h
        h = self.fc2(x_dict[target_node_type])
        return h
    def reset_parameters(self):
        #for node_type in self.fc1:
        #    nn.init.xavier_uniform_(self.fc1[node_type].weight, gain=nn.init.calculate_gain('relu'))
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if not self.layerwise:
            for edge_type in self.attn1:
                nn.init.zeros_(self.attn1[edge_type].weight)
                nn.init.zeros_(self.attn2[edge_type].weight)
        else:
            for i in range(self.hop):
                for edge_type in self.attn1[i]:
                    nn.init.zeros_(self.attn1[i][edge_type].weight)
                    nn.init.zeros_(self.attn2[i][edge_type].weight)
                    