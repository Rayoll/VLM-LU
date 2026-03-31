import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl

class GAT(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for layer in range(self.num_layers):
            if layer == 0:
                self.layers.append(GATConv(in_channels,hid_channels,num_heads=3))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))
            else:
                self.layers.append(GATConv(hid_channels,hid_channels,num_heads=3))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))

        self.fc = nn.Linear(hid_channels, out_channels)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, g, h):
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            h = h.mean(1)
            if len(h) != 1:
                h = self.batch_norms[i](h)
            h = F.relu(h)
        h = self.drop(h)
        h = self.fc(h)
        with g.local_scope():
            g.ndata['h'] = h
            h = dgl.mean_nodes(g,'h')

        return h
