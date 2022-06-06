import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models import LogReg
import torch.nn as nn
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise
import time

def evaluate(embeds, idx_train, idx_val, idx_test, labels, args, isTest=True):
    hid_units = embeds.shape[2]
    nb_classes = labels.shape[2]
    xent = nn.CrossEntropyLoss()
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)
    
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    log.to(args.device)

    best_val_f1_macro = 0
    test_f1_macro = 0
    for epoch in range(1, args.num_iter + 1):
        # train
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

        # val
        logits = log(val_embs)
        preds = torch.argmax(logits, dim=1)

        Val_f1 = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            
        # test
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)

        Test_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            
        if Val_f1 > best_val_f1_macro:
            best_val_f1_macro = Val_f1
            test_f1_macro = Test_f1
            count = 0
        elif count >= args.patience2:
            break
        else:
            count += 1
    
    best_val_f1_micro = 0
    test_f1_micro = 0
    for epoch in range(1, args.num_iter + 1):
        # train
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

        # val
        logits = log(val_embs)
        preds = torch.argmax(logits, dim=1)

        Val_f1 = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')
            
        # test
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)

        Test_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
            
        if Val_f1 > best_val_f1_micro:
            best_val_f1_micro = Val_f1
            test_f1_micro = Test_f1
            count = 0
        elif count >= args.patience2:
            break
        else:
            count += 1
            
    # test_embs = np.array(test_embs.cpu())
    # test_lbls = np.array(test_lbls.cpu())

    # run_kmeans(test_embs, test_lbls, nb_classes)
    # run_similarity_search(test_embs, test_lbls)
    return best_val_f1_macro, test_f1_macro, best_val_f1_micro, test_f1_micro

def run_similarity_search(test_embs, test_lbls):
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4)))

    st = ','.join(st)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(st))


def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)

        s1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)

    s1 = sum(NMI_list) / len(NMI_list)

    print('\t[Clustering] NMI: {:.4f}'.format(s1))