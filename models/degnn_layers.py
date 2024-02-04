import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, features):
        """
        Graph Convolutional Layer forward function
        """
        if features.data.is_sparse:
            support = torch.spmm(features, self.weight)
        else:
            support = torch.mm(features, self.weight)
        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = adj @ support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_model(nn.Module):
    """
    2-layer GCN used as the downstream network.
    """
    def __init__(self, in_feats, n_hidden, out_dims, dropout):
        super(GCN_model, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(in_feats, n_hidden))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(GraphConvolution(n_hidden, out_dims))

    def forward(self, adj, features):
        h = features
        for layer in self.layers:
            if isinstance(layer, nn.Dropout) or isinstance(layer, nn.ReLU):
                h = layer(h)
            else:
                h = layer(adj, h)
        return h


class GCN_1layer(nn.Module):
    """
    1-layer GCN used as the encoder of experts
    """
    def __init__(self, in_feats, out_dims, dropout):
        super(GCN_1layer, self).__init__()
        self.layers = nn.ModuleList()
        self.gcn = GraphConvolution(in_feats, out_dims)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, adj, features):
        x = self.gcn(adj, features)
        x = self.act(x)
        return x


class Reconstruct_adj(torch.nn.Module):
    """
    Reconstruction process of the edge expert.
    """
    def __init__(self, n_nodes, n_edges, k_ratio, device='cpu'):
        super(Reconstruct_adj, self).__init__()
        self.n_nodes = n_nodes
        self.device = device
        self.eye = torch.eye(n_nodes).to(device)
        self.ones = torch.ones((n_nodes, n_nodes)).to(device)
        self.zeros = torch.zeros((n_nodes, n_nodes)).to(device)
        self.k = int(n_edges * k_ratio)
        print(f"k: {self.k}")

    
    def forward(self, adj, H):
        normalized_z = F.normalize(H, dim=1)
        sim_mat = torch.mm(normalized_z, normalized_z.t()) * (self.ones - self.eye)

        # remove
        values, indices = torch.topk((sim_mat - adj * 100 + self.eye * 500).view(-1), k=self.k, largest=False)
        delete_mask = torch.ones((self.n_nodes, self.n_nodes)).to(self.device)
        delete_mask.view(-1)[indices] = 0
        adj_removed = adj * delete_mask

        # add
        values, indices = torch.topk((sim_mat - adj_removed * 100 - self.eye * 100).view(-1), k=self.k) # no overlap
        add_mask = torch.zeros((self.n_nodes, self.n_nodes)).to(self.device)
        add_mask.view(-1)[indices[0 <= values]] = 1
        add_mat = F.threshold(sim_mat * add_mask, 0, 0)
        adj_new = adj_removed + add_mat
        return adj_new


class BCEExeprtLoss(nn.Module):
    """
    Binary cross-entropy loss for the expert.
    """
    def __init__(self, n_nodes, device):
        super(BCEExeprtLoss, self).__init__()
        self.lbl_pos = torch.ones(n_nodes*3).to(device)
        self.lbl_neg = torch.zeros(n_nodes).to(device)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, logits_aa, logits_ta, logits_ata, logits_neg):
        logits_pos = torch.squeeze(torch.cat((logits_aa, logits_ta, logits_ata), dim=0))
        logits_neg = torch.squeeze(logits_neg)
        loss = self.criterion(logits_pos, self.lbl_pos) + self.criterion(logits_neg, self.lbl_neg)
        return loss


class Discriminator_innerprod(nn.Module):
    """
    Discriminator defined by inner product function.
    """
    def __init__(self):
        super(Discriminator_innerprod, self).__init__()

    def forward(self, H, Haa, Hta, Hata, Hneg):
        logits_aa = torch.sum(torch.mul(H, Haa), dim=1, keepdim=True)
        logits_ta = torch.sum(torch.mul(H, Hta), dim=1, keepdim=True)
        logits_ata = torch.sum(torch.mul(H, Hata), dim=1, keepdim=True)
        logits_neg = torch.sum(torch.mul(H, Hneg), dim=1, keepdim=True)
        return logits_aa, logits_ta, logits_ata, logits_neg