import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
from embedder import embedder
from layers import GCN, Discriminator, Attention
import numpy as np
np.random.seed(0)
from evaluate import evaluate
from models import LogReg
import pickle as pkl
import time

def remove_edge_pts(accs, pct=0.1):
    accs = sorted(list(accs))
    N = len(accs)
    M = int(N * pct)
    accs = np.array(accs[M:N-M])
    return accs

class DMGI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        macro_f1s, micro_f1s = [], []
        for i in range(self.args.num_test):
            model = modeler(self.args).to(self.args.device)
            # if i == 0:
            #     print('#Parameters:', sum(p.numel() for p in model.parameters()))
            optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
            cnt_wait = 0; best = 1e9
            b_xent = nn.BCEWithLogitsLoss()
            xent = nn.CrossEntropyLoss()
            t0 = time.time()
            training_time = 0
            for epoch in range(1, self.args.nb_epochs + 1):
                t1 = time.time()
                xent_loss = None
                model.train()
                optimiser.zero_grad()
                idx = np.random.permutation(self.args.nb_nodes)
                
                shuf = [feature[:, idx, :] for feature in features]
                shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]
                
                lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
                lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
                lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

                result = model(features, adj, shuf, self.args.sparse, None, None, None)
                logits = result['logits']

                for view_idx, logit in enumerate(logits):
                    if xent_loss is None:
                        xent_loss = b_xent(logit, lbl)
                    else:
                        xent_loss += b_xent(logit, lbl)

                loss = xent_loss

                reg_loss = result['reg_loss']
                loss += self.args.reg_coef * reg_loss

                if self.args.isSemi:
                    sup = result['semi']
                    semi_loss = xent(sup[self.idx_train], self.train_lbls)
                    loss += self.args.sup_coef * semi_loss

                if loss < best:
                    best = loss
                    cnt_wait = 0
                    torch.save(model.state_dict(), self.args.save_path + 'best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
                else:
                    cnt_wait += 1

                if cnt_wait == self.args.patience:
                    break

                loss.backward()
                optimiser.step()
                training_time += time.time() - t1


            model.load_state_dict(torch.load(self.args.save_path + 'best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths)))

            # Evaluation
            model.eval()
            best_val_f1_macro, test_f1_macro, best_val_f1_micro, test_f1_micro = evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args)
            print('total time = {:.3f}, train time/epoch = {:.5f}, best_val_f1 (macro) = {:.3f}, test_f1 (macro) = {:.3f}, best_val_f1 (micro) = {:.3f}, test_f1 (micro) = {:.3f}'.
                  format(time.time() - t0, training_time/epoch, best_val_f1_macro, test_f1_macro, best_val_f1_micro, test_f1_micro))
            macro_f1s.append(test_f1_macro)
            micro_f1s.append(test_f1_micro)
        macro_f1s = np.array(macro_f1s)
        micro_f1s = np.array(micro_f1s)
        print('macro_f1s: \n', macro_f1s)
        print('micro_f1s: \n', micro_f1s)
        print("\t[Classification] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})"
              .format(np.mean(macro_f1s), np.std(macro_f1s), np.mean(micro_f1s), np.std(micro_f1s)))
        macro_f1s_2 = remove_edge_pts(macro_f1s)
        micro_f1s_2 = remove_edge_pts(micro_f1s)
        print("\t[Classification after filter] Macro-F1: {:.4f} ({:.4f}) | Micro-F1: {:.4f} ({:.4f})"
              .format(np.mean(macro_f1s_2), np.std(macro_f1s_2), np.mean(micro_f1s_2), np.std(micro_f1s_2)))
        np.savetxt('f1-macro_DMGI.txt', macro_f1s)
        np.savetxt('f1-micro_DMGI.txt', micro_f1s)
            

class modeler(nn.Module):
    def __init__(self, args):
        super(modeler, self).__init__()
        self.args = args
        self.gcn = nn.ModuleList([GCN(args.ft_size, args.hid_units, args.activation, args.drop_prob, args.isBias) for _ in range(args.nb_graphs)])

        self.disc = Discriminator(args.hid_units)
        self.H = nn.Parameter(torch.FloatTensor(1, args.nb_nodes, args.hid_units))
        self.readout_func = self.args.readout_func
        if args.isAttn:
            self.attn = nn.ModuleList([Attention(args) for _ in range(args.nheads)])

        if args.isSemi:
            self.logistic = LogReg(args.hid_units, args.nb_classes).to(args.device)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.H)

    def forward(self, feature, adj, shuf, sparse, msk, samp_bias1, samp_bias2):
        h_1_all = []; h_2_all = []; c_all = []; logits = []
        result = {}

        for i in range(self.args.nb_graphs):
            h_1 = self.gcn[i](feature[i], adj[i], sparse)

            # how to readout positive summary vector
            c = self.readout_func(h_1)
            c = self.args.readout_act_func(c)  # equation 9
            h_2 = self.gcn[i](shuf[i], adj[i], sparse)
            logit = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

            h_1_all.append(h_1)
            h_2_all.append(h_2)
            c_all.append(c)
            logits.append(logit)

        result['logits'] = logits

        # Attention or not
        if self.args.isAttn:
            h_1_all_lst = []; h_2_all_lst = []; c_all_lst = []

            for h_idx in range(self.args.nheads):
                h_1_all_, h_2_all_, c_all_ = self.attn[h_idx](h_1_all, h_2_all, c_all)
                h_1_all_lst.append(h_1_all_); h_2_all_lst.append(h_2_all_); c_all_lst.append(c_all_)

            h_1_all = torch.mean(torch.cat(h_1_all_lst, 0), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all_lst, 0), 0).unsqueeze(0)

        else:
            h_1_all = torch.mean(torch.cat(h_1_all), 0).unsqueeze(0)
            h_2_all = torch.mean(torch.cat(h_2_all), 0).unsqueeze(0)


        # consensus regularizer
        pos_reg_loss = ((self.H - h_1_all) ** 2).sum()
        neg_reg_loss = ((self.H - h_2_all) ** 2).sum()
        reg_loss = pos_reg_loss - neg_reg_loss
        result['reg_loss'] = reg_loss

        # semi-supervised module
        if self.args.isSemi:
            semi = self.logistic(self.H).squeeze(0)
            result['semi'] = semi



        return result