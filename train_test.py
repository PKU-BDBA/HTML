""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import HTML
import random

cuda = True if torch.cuda.is_available() else False

def seed_it(seed):
    random.seed(seed) 
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(40)

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def prepare_trte_data(data_folder,uni=True,dual=True,triple=True):
    num_view = 1
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view+1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), dtype=np.float64,delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), dtype=np.float64,delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    if uni:
        for i in range(len(data_tensor_list)):
            data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
            data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()),0))
            data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    
    if dual and num_view>=2:
        for i in range(len(data_tensor_list)):
            for j in range(i+1,len(data_tensor_list)):
                data_train_list.append(torch.cat([data_tensor_list[i][idx_dict["tr"]].clone(),data_tensor_list[j][idx_dict["tr"]].clone()],1))
                data_test_list.append(torch.cat([data_tensor_list[i][idx_dict["te"]].clone(),data_tensor_list[j][idx_dict["te"]].clone()],1))
                data_all_list.append(torch.cat((torch.cat([data_tensor_list[i][idx_dict["tr"]].clone(),data_tensor_list[j][idx_dict["tr"]].clone()],1),
                                        torch.cat([data_tensor_list[i][idx_dict["te"]].clone(),data_tensor_list[j][idx_dict["te"]].clone()],1)),0))

    if triple and num_view>=3:
        data_train_list.append(torch.cat([data_tensor_list[0][idx_dict["tr"]].clone(),data_tensor_list[1][idx_dict["tr"]].clone(),data_tensor_list[2][idx_dict["tr"]].clone()],1))
        data_test_list.append(torch.cat([data_tensor_list[0][idx_dict["te"]].clone(),data_tensor_list[1][idx_dict["te"]].clone(),data_tensor_list[2][idx_dict["te"]].clone()],1))
        data_all_list.append(torch.cat((torch.cat([data_tensor_list[0][idx_dict["tr"]].clone(),data_tensor_list[1][idx_dict["tr"]].clone(),data_tensor_list[2][idx_dict["tr"]].clone()],1),
                                        torch.cat([data_tensor_list[0][idx_dict["te"]].clone(),data_tensor_list[1][idx_dict["te"]].clone(),data_tensor_list[2][idx_dict["te"]].clone()],1)),0))

    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _, uncertainty = model(data_list, label)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit,uncertainty = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob,uncertainty

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)
    
def computeAUROC(dataGT, dataPRED, classCount=5):
    outAUROC = []
        
    datanpGT = dataGT
    datanpPRED = dataPRED
    dataIndex=torch.argmax(dataGT,dim=1)

    for i in range(classCount):
        if i in dataIndex:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
    return outAUROC


def train(data_folder, modelpath, testonly,uni,dual,triple):
    test_inverval = 1
    if 'BRCA' in data_folder:
        hidden_dim = [1000]
        num_epoch = 2500
        lr = 2e-4
        num_class = 5
    elif 'ROSMAP' in data_folder:
        hidden_dim = [500]
        num_epoch = 1500
        lr = 1e-4
        num_class = 2
    elif 'LGG' in data_folder:
        hidden_dim = [500]
        num_epoch = 1500
        lr = 1e-4
        num_class = 2
    elif 'KIPAN' in data_folder:
        hidden_dim = [500]
        num_epoch = 500
        lr = 1e-4
        num_class = 3

    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder,uni,dual,triple)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]
    model = HTML(dim_list, hidden_dim, num_class, dropout=0.5)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    
    labels=torch.tensor([])
    for i in labels_trte[trte_idx["te"]]:
        _=[0]*(max(labels_trte[trte_idx["te"]])+1)
        _[i]=1
        labels=torch.cat([labels,torch.tensor([_])],dim=0)
    
    if testonly:
        load_checkpoint(model, os.path.join(modelpath, data_folder[11:],"checkpoint.pt"))
        te_prob,uncertainty = test_epoch(data_test_list, model)
        if num_class == 2:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
            print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])*100))
            print("Test Uncertainty:{:.5f}".format(np.mean(uncertainty)*100))
        else:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')*100))
            print("Test average AUC: {:.5f}".format(np.mean(computeAUROC(labels, te_prob,num_class))*100))
            print("Test Uncertainty:{:.5f}".format(np.mean(uncertainty)*100))
    else:
        print("\nTraining...")
        best_result={"acc":0}
        for epoch in range(num_epoch+1):
            train_epoch(data_tr_list, labels_tr_tensor, model, optimizer)
            scheduler.step()
            if epoch % test_inverval == 0:
                te_prob,uncertainty = test_epoch(data_test_list, model)
                print("\nTrain: Epoch {:d}".format(epoch))
                if num_class == 2:
                    print("Train ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
                    print("Train F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
                    print("Train AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])*100))
                    print("Train Uncertainty:{:.5f}".format(np.mean(uncertainty)*100))
                else:
                    print("Train ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100))
                    print("Train F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')*100))
                    print("Train average AUC: {:.5f}".format(np.mean(computeAUROC(labels, te_prob,num_class))*100))
                    print("Train Uncertainty:{:.5f}".format(np.mean(uncertainty)*100))
                if accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100>=best_result["acc"]:
                    best_result["acc"]=accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))*100
                    best_result["f1-macro"]=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')*100
                    best_result["f1-weighted"]=f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')*100
                    best_result["uncertainty"]=np.mean(uncertainty)*100
                    save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder[11:]))
        print(best_result)
        return best_result