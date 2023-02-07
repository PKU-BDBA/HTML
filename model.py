""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class HTMLModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class+1
        self.dropout_rate = dropout

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.ModalityConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.ModalityClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], self.classes) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        FeatureInfo, feature, Prediction, MCC = dict(), dict(), dict(), dict()
        for view in range(self.views):
            
            data_list[view]=data_list[view].squeeze(0)
            FeatureInfo[view] = torch.sigmoid(self.dropout(
                self.FeatureInforEncoder[view](data_list[view])))
            feature[view] = data_list[view] * FeatureInfo[view]


            # Encoder
            feature[view] = self.FeatureEncoder[view](feature[view])
            # feature[view] = self.relu(feature[view])
            feature[view] = self.dropout(feature[view])

            # Classifier
            Prediction[view] = self.dropout(self.ModalityClassifierLayer[view](feature[view]).squeeze(0))
                
            # Modality Confidence Calculation
            MCC[view] = torch.sigmoid(self.ModalityConfidenceLayer[view](feature[view]))

        Predictions = torch.ones(Prediction[0].shape).cuda()
        MMLoss = 0

        for view in range(self.views):
            # l_0 regularization for high sparsity
            MMLoss = MMLoss+torch.mean(FeatureInfo[view])
            # add up the corresponding probability of each element
            Predictions = torch.add(Predictions, F.softmax(Prediction[view], dim=1) * MCC[view])
            # add the uncertainty probability
            Predictions[:, -1] = Predictions[:, -1]+(torch.sub(F.softmax(Prediction[view], dim=1)[:, -1], torch.mul(
                F.softmax(Prediction[view], dim=1)[:, -1], F.softmax(Prediction[view], dim=1)[:, -1]))*2)*MCC[view].squeeze(1)
    
        MMlogit = F.softmax(Predictions[:, :-1], dim=1)
        # l_cls+l_uncertain
        uncertainty = torch.div(Predictions[:, -1], torch.max(Predictions, dim=1)[0])
        if infer:
            return MMlogit, uncertainty.cpu().detach().numpy()
        MMLoss = MMLoss+torch.mean(criterion(Predictions[:, :-1], label))
        MMLoss = MMLoss+torch.mean(uncertainty)
        return MMLoss, MMlogit, uncertainty.cpu().detach().numpy()

    
    def infer(self, data_list):
        MMlogit,uncertainty = self.forward(data_list, infer=True)
        return MMlogit,uncertainty