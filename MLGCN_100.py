import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv
from sklearn import metrics

EPOCH = 970

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data = torch.load("data/datadata_PathNet.pkl")
data_str = torch.load('./data/str_fearures_PathNet_p=0.5_q=0.5_58.pkl')
data = data.to(device)


@torch.no_grad()
def test(data, mask):
    model.eval()
    x, x1 = model()
    pred = torch.sigmoid(x[mask])
    pred1 = torch.sigmoid(x1[mask])
    pred2 = 0.8 * pred + 0.2 * pred1
    precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred2.cpu().detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred2.cpu().detach().numpy()), area


class Net(torch.nn.Module):
    def __init__(self, num_feature=data.x.shape[1], num_label=1, hidden_channels_2=64, num_layers=4, alpha=0.4,
                 theta=0.9, dropout=0.6):
        super(Net, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(num_feature, hidden_channels_2))
        self.lins.append(Linear(hidden_channels_2, num_label))
        self.convs = torch.nn.ModuleList()
        self.convs1 = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels_2, alpha, theta, layer + 1,
                         shared_weights=False, add_self_loops=False, normalize=True))
            self.convs1.append(
                GCN2Conv(hidden_channels_2, alpha, theta, layer + 1,
                         shared_weights=False, add_self_loops=False, normalize=True))
        self.dropout = dropout
        self.lin1 = Linear(num_feature, hidden_channels_2)
        self.lin11 = Linear(58, 64)
        self.lin2 = Linear(64, 1)
        self.lin3 = Linear(64, 1)

    def forward(self):
        edge_index = data.edge_index
        x0 = F.dropout(data.x, self.dropout, training=self.training)
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
        z1 = self.lin3(z1)
        return z, z1


pred_all = np.zeros((data.num_nodes, 1))
for i in np.arange(0, 100):
    # Creating model
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-6)

    # Training model
    for epoch in range(1, EPOCH + 1):
        if epoch % 100 == 0:
            print('epoch:', epoch)
        model.train()
        optimizer.zero_grad()
        pred, pred1 = model()
        loss = 0.2 * F.binary_cross_entropy_with_logits(pred, data.y.view(-1, 1)) \
               + 0.8 * F.binary_cross_entropy_with_logits(pred1, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
    print('Training model for %d times.' % (i + 1))
    # Predicting via trained model
    pred, pred1 = model()
    pred2 = 0.8 * pred + 0.2 * pred1
    pred_all = pred2.cpu().detach().numpy() + pred_all

pred_all = pred_all / 100
pre_res = pd.DataFrame(pred_all, columns=['score'], index=data.node_names)
pre_res.sort_values(by=['score'], inplace=True, ascending=False)
# Save the final ranking list of predicted driver genes
pre_res.to_csv(path_or_buf='./data/predicted_socres_PathNet.txt', sep='\t', index=True, header=True)
