import random
import pandas as pd
import time
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.backends.cudnn as cudnn
from torch_geometric.nn import GCN2Conv
from torch_geometric.utils import dropout_adj

from sklearn import metrics

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
EPOCH = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './data/luad/'
data = pd.read_csv(path + "single_cancer_luad.csv", sep=",").values[:, :3]
data = torch.Tensor(data).to(device)
data1 = torch.load("./data/CPDB_data.pkl")  # get edge_index
data_str = torch.load("./data/str_fearures.pkl")
print(device)
data1 = data1.to(device)


def load_label_single(path):
    label = np.loadtxt(path + "label_file.txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos.txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)

    return Y, label_pos, label_neg


def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    # pos_label：Positive sample index
    # neg_label：Negative sample index
    # l：number of genes
    # l1：Number of positive samples
    # l2：number of negative samples
    # i：number of folds
    pos_test = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_test))
    neg_test = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_test))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_test)):
        indexs2[pos_test[j]] = True
    for j in range(len(neg_test)):
        indexs2[neg_test[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    te_mask = torch.from_numpy(np.array(indexs2))

    return tr_mask, te_mask


def train(mask, Y):
    model.train()
    optimizer.zero_grad()
    pred, pred1 = model()
    loss = 0.8 * F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) + \
           0.2 * F.binary_cross_entropy_with_logits(pred1[mask], Y[mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(mask, Y):
    model.eval()
    x, x1 = model()
    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    pred1 = torch.sigmoid(x1[mask]).cpu().detach().numpy()
    pred2 = 0.6 * pred + 0.4 * pred1
    Yn = Y[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred2)
    area = metrics.auc(recall, precision)

    return metrics.roc_auc_score(Yn, pred2), area


class Net(torch.nn.Module):
    def __init__(self, num_feature=3, num_label=1, hidden_channels_2=64, num_layers=2, alpha=0.9, theta=0.2,
                 dropout=0.1):
        super(Net, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_feature, hidden_channels_2))
        self.lins.append(Linear(hidden_channels_2, num_label))
        self.convs = torch.nn.ModuleList()
        self.convs1 = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels_2, alpha, theta, layer + 1,  # - 0.1*layer
                         shared_weights=False, add_self_loops=False, normalize=True))
            self.convs1.append(
                GCN2Conv(hidden_channels_2, alpha, theta, layer + 1,
                         shared_weights=False, add_self_loops=False, normalize=True))
        self.dropout = dropout
        self.lin1 = Linear(num_feature, hidden_channels_2)
        self.lin11 = Linear(num_feature + 13, hidden_channels_2)
        self.lin2 = Linear(hidden_channels_2, 1)

    def forward(self):
        edge_index, _ = dropout_adj(data1.edge_index, p=0.2,
                                    force_undirected=True,
                                    num_nodes=data.shape[0],
                                    training=self.training)
        x0 = F.dropout(data, self.dropout, training=self.training)
        x00 = F.dropout(data_str, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x0).relu()
        x1 = x_1 = self.lin11(x00).relu()
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = x.relu()
        for i, conv1 in enumerate(self.convs1):
            x1 = F.dropout(x1, self.dropout, training=self.training)
            x1 = conv1(x1, x_1, edge_index)
            x1 = x1.relu()
        z = F.dropout(x, self.dropout, training=self.training)
        z1 = F.dropout(x1, self.dropout, training=self.training)
        z = self.lins[1](z)
        z1 = self.lin2(z1)
        return z, z1


time_start = time.time()
# ten five-fold cross-validations
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    label, label_pos, label_neg = load_label_single(path)
    random.shuffle(label_pos)
    random.shuffle(label_neg)
    l = len(label)
    l1 = int(len(label_pos) / 5)
    l2 = int(len(label_neg) / 5)
    for cv_run in range(5):
        print('第', i + 1, '次5折的第', cv_run + 1, '折：')
        tr_mask, te_mask = sample_division_single(label_pos, label_neg, l, l1, l2, cv_run)
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, EPOCH):
            if epoch % 100 == 0:
                print('epoch:', epoch)
            train(tr_mask, label)
        AUC[i][cv_run], AUPR[i][cv_run] = test(te_mask, label)
        print(AUC[i][cv_run], '\n', AUPR[i][cv_run])

    print(time.time() - time_start)

print(AUC.mean())
print(AUC.var())
print(AUPR.mean())
print(AUPR.var())
