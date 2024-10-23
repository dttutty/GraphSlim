from copy import deepcopy

import torch.nn as nn
import torch.optim as optim

from graphslim.utils import *


class BaseGNN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, args, mode):
        super(BaseGNN, self).__init__()


        self.args = args
        self.device = args.device
        self.with_bn = args.with_bn
        self.with_relu = True
        self.with_bias = True
        self.nlayers = args.nlayers
        self.ntrans = args.ntrans
        self.multi_label = args.multi_label
        self.metric = args.metric
        # self.metric = accuracy if args.metric == 'accuracy' else f1_macro

        # 设置根据 mode 的不同调整的参数
        self.loss = F.nll_loss if mode == 'attack' else None
        self.dropout = 0 if mode == 'eval' else args.dropout
        self.weight_decay = 5e-4 if mode == 'eval' else args.weight_decay
    
        self.lr = args.lr
        self.alpha = args.alpha
        self.output = self.best_model = self.best_output = self.adj_norm = self.features = self.float_label = None

        self.layers = nn.ModuleList([])
    
    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, adj, output_layer_features=False):

        if isinstance(adj, list):
            for i, layer in enumerate(self.layers):
                x = layer(x, adj[i])
                if i != self.nlayers - 1:
                    x = self.bns[i](x) if self.with_bn else x
                    x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)
        else:
            feat_list = []
            for ix, layer in enumerate(self.layers):
                x = layer(x, adj)
                if ix != self.nlayers - 1:
                    x = self.bns[ix](x) if self.with_bn else x
                    if self.with_relu:
                        x = F.relu(x)
                    x = F.dropout(x, self.dropout, training=self.training)
                if output_layer_features and ix < self.nlayers:
                    feat_list.append(x.reshape(-1, x.shape[-1]))

        x = x.view(-1, x.shape[-1])
        if self.multi_label:
            return torch.sigmoid(x)
        if output_layer_features:
            return feat_list, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)

    def fit_with_val(self, data, train_iters=600, verbose=False,
                     normadj=True, setting='trans', reduced=False, final_output=False, best_val=None, **kwargs):
        args=self.args

        self.initialize()
        # data for training
        adj, features, labels, labels_val = self._prepare_data(data, setting, reduced)
        adj = self._process_adj_tensor(args, adj)

        if self.loss is None:
            if args.method == 'geom' and args.soft_label:
                self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            elif data.nclass == 1:
                self.loss = torch.nn.BCELoss()
            elif len(labels.shape) == 2:
                if args.eval_loss=='MSE':
                    self.loss = torch.nn.MSELoss()
                elif args.eval_loss=='KLD':
                    self.loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
                else:
                    raise NotImplementedError
                self.weight_decay = args.eval_wd
            else:
                labels = to_tensor(label=labels, device=self.device)
                self.loss = F.nll_loss
        else:
            labels = to_tensor(label=labels, device=self.device)


        if verbose:
            print('=== training ===')

        if best_val is None:
            best_acc_val = 0
        else:
            best_acc_val = best_val
        if setting == 'ind':
            feat_full, adj_full = data.feat_val, data.adj_val
        else:
            feat_full, adj_full = data.feat_full, data.adj_full
        feat_full, adj_full = to_tensor(feat_full, adj_full, device=self.device)
        if normadj:
            adj_full = normalize_adj_tensor(adj_full, sparse=is_sparse_tensor(adj_full))

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train()
        for i in range(train_iters):
            if i == train_iters // 2 and self.lr > 0.001:
                optimizer = optim.Adam(self.parameters(), lr=self.lr * 0.1, weight_decay=self.weight_decay)

            optimizer.zero_grad()
            output = self.forward(features, adj)
            loss_train = self.loss(output if output.shape[0] == labels.shape[0] else output[data.idx_train], labels)

            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                acc_train = accuracy(output if output.shape[0] == labels.shape[0] else output[data.idx_train], labels)
                print('Epoch {}, training acc: {}'.format(i, acc_train))

            with torch.no_grad():
                self.eval()
                output = self.forward(feat_full, adj_full)

                acc_val = self.metric(output if output.shape[0] == labels_val.shape[0] else output[data.idx_val],
                                   labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    # self.output = output
                    weights = deepcopy(self.state_dict())
        if final_output:
            return
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        try:
            self.load_state_dict(weights)
        except:
            pass
        return best_acc_val.item()

    def _process_adj_tensor(self, args, adj):
        if self.__class__.__name__ == 'GAT':
            if len(adj.shape) == 3:
                adj = [normalize_adj_tensor(a.to_sparse(), sparse=True) for a in adj]
            else:
                adj = normalize_adj_tensor(adj.to_sparse() if not is_sparse_tensor(adj) else adj, sparse=True)

        # SparseTensor synthetic graph only used in graphsage, msgc and simgc
        elif self.__class__.__name__ == 'GraphSage' and args.method == 'msgc':
            pass
        elif args.method == 'simgc':
            adj = normalize_adj_tensor(adj, sparse=True)
        else:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))
        return adj

    def _prepare_data(self, data, setting, reduced):
        if reduced:
            return to_tensor(data.adj_syn, data.feat_syn, data.labels_syn, label2=data.labels_val, device=self.device)
        elif setting == 'trans':
            return to_tensor(data.adj_full, data.feat_full, label=data.labels_train, label2=data.labels_val, device=self.device)
        else:
            return to_tensor(data.adj_train, data.feat_train, label=data.labels_train, label2=data.labels_val, device=self.device)

    @torch.no_grad()
    def test(self, data, setting='trans', verbose=False):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        idx_test = data.idx_test
        labels_test = torch.LongTensor(data.labels_test).to(self.device)
        # whether condensed or not, use the raw graph to test

        if setting == 'ind':
            output = self.predict(data.feat_test, data.adj_test)
            loss_test = F.nll_loss(output, labels_test)
            acc_test = self.metric(output, labels_test)
        else:
            output = self.predict(data.feat_full, data.adj_full)
            loss_test = F.nll_loss(output[idx_test], labels_test)
            acc_test = self.metric(output[idx_test], labels_test)

        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    @torch.no_grad()
    def predict(self, features=None, adj=None, normadj=True, output_layer_features=False):

        self.eval()
        features, adj = to_tensor(features, adj, device=self.device)
        if normadj:
            adj = normalize_adj_tensor(adj, sparse=is_sparse_tensor(adj))

        return self.forward(features, adj, output_layer_features=output_layer_features)
