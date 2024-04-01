"""multiple transformaiton and multiple propagation"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
from copy import deepcopy
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from graphslim import utils


class APPNP1(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4, alpha=0.1,
                 activation="relu", with_relu=True, with_bias=True, with_bn=False, device=None):

        super(APPNP1, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass
        self.alpha = alpha
        activation_functions = {
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'relu': F.relu,
            'linear': lambda x: x,
            'softplus': F.softplus,
            'leakyrelu': F.leaky_relu,
            'relu6': F.relu6,
            'elu': F.elu
        }
        self.activation = activation_functions.get(activation)

        if with_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(nn.BatchNorm1d(nhid))

        self.layers = nn.ModuleList([])
        # self.layers.append(MyLinear(nfeat, nclass))
        self.layers.append(MyLinear(nfeat, nhid))
        # self.layers.append(MyLinear(nhid, nhid))
        self.layers.append(MyLinear(nhid, nclass))

        # if nlayers == 1:
        #     self.layers.append(nn.Linear(nfeat, nclass))
        # else:
        #     self.layers.append(nn.Linear(nfeat, nhid))
        #     for i in range(nlayers-2):
        #         self.layers.append(nn.Linear(nhid, nhid))
        #     self.layers.append(nn.Linear(nhid, nclass))

        self.nlayers = nlayers
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None
        self.sparse_dropout = SparseDropout(dprob=0)

    def forward(self, x, adj):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        h = x
        # here nlayers means K
        for i in range(self.nlayers):
            # adj_drop = self.sparse_dropout(adj, training=self.training)
            adj_drop = adj
            x = torch.spmm(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h


        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        h = x
        for ix, (adj, _, size) in enumerate(adjs):
            # x_target = x[: size[1]]
            # x = self.layers[ix]((x, x_target), edge_index)
            # adj = adj.to(self.device)
            # adj_drop = F.dropout(adj, p=self.dropout)
            adj_drop = adj
            h = h[: size[1]]
            x = torch_sparse.matmul(adj_drop, x)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        for ix, (adj) in enumerate(adjs):
            # x_target = x[: size[1]]
            # x = self.layers[ix]((x, x_target), edge_index)
            # adj = adj.to(self.device)
            x = torch_sparse.matmul(adj, x)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


    def initialize(self):
        """Initialize parameters of GCN.
        """
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, val=False, **kwargs):
        '''data: full data class'''
        if initialize:
            self.initialize()

        # features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from utils import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels


        if val:
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val(labels, data, train_iters, verbose)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data.labels_val).to(self.device)

        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = self.loss(output, labels)
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full_norm)
                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data.idx_val], labels_val)
                    acc_val = utils.accuracy(output[data.idx_val], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, data):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict(data.feat_full, data.adj_full)
        # output = self.output
        idx_test = data.idx_test
        labels_test = torch.LongTensor(data.labels_test).cuda()
        loss_test = F.nll_loss(output[idx_test], labels_test)
        acc_test = utils.accuracy(output[idx_test], labels_test)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)



class MyLinear(Module):
    """Simple Linear layer, modified from https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor([out_features]))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = support
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        self.kprob=1-dprob

    def forward(self, x, training):
        if training:
            mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
            rc=x._indices()[:,mask]
            val=x._values()[mask]*(1.0/self.kprob)
            return torch.sparse.FloatTensor(rc, val, x.size())
        else:
            return x
