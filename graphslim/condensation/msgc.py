from torch import nn
from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.condensation.utils import match_loss
from graphslim.dataset.utils import save_reduced
from graphslim.models import *
from graphslim.utils import *


class MSGC(GCondBase):

    def __init__(self, setting, data, args, **kwargs):
        super(MSGC, self).__init__(setting, data, args, **kwargs)
        x_channels = data.feat_train.shape[1]
        edge_hidden_channels = 256
        self.n_syn = self.nnodes_syn
        self.y_syn = to_tensor(label=data.labels_syn, device=args.device)
        self.x_syn = nn.Parameter(torch.empty(self.n_syn, x_channels).to(args.device))
        self.batch_size = args.batch_adj
        self.n_classes = data.nclass
        self.device = args.device

        self.adj_mlp = nn.Sequential(
            nn.Linear(x_channels * 2, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, edge_hidden_channels),
            nn.BatchNorm1d(edge_hidden_channels),
            nn.ReLU(),
            nn.Linear(edge_hidden_channels, 1)
        ).to(args.device)
        # -------------------------------------------------------------------------
        # self.reset_adj_batch()

    def reduce(self, data, verbose=True):

        args = self.args
        features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)

        adj = normalize_adj_tensor(adj, sparse=True)
        y_syn = self.y_syn.repeat(self.batch_size)
        assert args.condense_model != 'GAT'
        basic_model = eval(args.condense_model)(self.feat_syn.shape[1], args.hidden, data.nclass, args).to(self.device)

        self.reset_adj_batch()
        feat_init = self.init_feat()
        self.x_syn.data.copy_(feat_init)

        optimizer_x = torch.optim.Adam(self.x_parameters(), lr=args.lr_feat)
        optimizer_adj = torch.optim.Adam(self.adj_parameters(), lr=args.lr_adj)
        best_val = 0
        args.window = args.patience = 20
        losses = FixLenList(args.window)
        x_syns = FixLenList(args.window)
        adj_t_syns = FixLenList(args.window)
        optimizer_basic_model = torch.optim.Adam(basic_model.parameters(), lr=args.lr)

        for it in trange(args.epochs):
            basic_model.initialize()
            loss_avg = 0
            for step_syn in range(args.outer_loop):
                basic_model = self.check_bn(basic_model)
                basic_model.eval()  # fix basic_model while optimizing graphsyner
                ######################graph optimization#####################################
                adj_t_syn = self.get_adj_t_syn()
                x_syn = self.x_syn
                loss = 0.
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                        c, adj, args)
                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = basic_model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_reals = torch.autograd.grad(loss_real, basic_model.parameters())
                    gw_reals = list((_.detach().clone() for _ in gw_reals))
                    # ------------------------------------------------------------------
                    output_syn = basic_model(x_syn, adj_t_syn)
                    # ind = self.syn_class_indices[c]
                    loss_syn = F.nll_loss(output_syn[y_syn == c], y_syn[y_syn == c])
                    gw_syns = torch.autograd.grad(loss_syn, basic_model.parameters(), create_graph=True)
                    # ------------------------------------------------------------------
                    coeff = self.num_class_dict[c] / self.n_syn
                    ml = match_loss(gw_syns, gw_reals, args, args.device)
                    loss += coeff * ml
                loss_avg += loss.item()
                optimizer_x.zero_grad()
                optimizer_adj.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    optimizer_adj.step()
                else:
                    optimizer_x.step()

                x_syn = x_syn.detach()
                adj_t_syn = self.get_adj_t_syn().detach()
                #################################################
                for i in range(args.inner_loop):
                    optimizer_basic_model.zero_grad()
                    logits = basic_model(x_syn, adj_t_syn)
                    inner_loss = F.nll_loss(logits, y_syn)

                    inner_loss.backward()
                    optimizer_basic_model.step()
            loss_avg /= (data.nclass * 20)
            losses.append(loss_avg)
            x_syns.append(x_syn)
            adj_t_syns.append(adj_t_syn)
            if it + 1 in args.checkpoints:
                loss_window = sum(losses.data) / len(losses.data)
                best_x_syn = sum(x_syns.data) / len(x_syns.data)
                best_adj_t_syn = sum(adj_t_syns.data) / len(adj_t_syns.data)
                data.feat_syn, data.adj_syn, data.labels_syn = best_x_syn, best_adj_t_syn, y_syn
                best_val = self.intermediate_evaluation(best_val, loss_window)

        return data

    def x_parameters(self):
        return [self.x_syn]

    def adj_parameters(self):
        return self.adj_mlp.parameters()

    def reset_adj_batch(self):
        rows = []
        cols = []
        batch = []
        for i in range(self.batch_size):
            n_neighbor = torch.zeros(self.n_syn, self.n_classes, device=self.device)
            index = torch.arange(self.n_syn, device=self.device)
            row = []
            col = []
            for row_id in range(self.n_syn):
                for c in range(self.n_classes):
                    c_mask = self.y_syn == c
                    c_mask[row_id] = False
                    if c_mask.sum() == 0 or n_neighbor[row_id][c] > 1:
                        continue
                    link_coef = n_neighbor[c_mask, self.y_syn[row_id]]
                    selected = link_coef.argmin()
                    candidates_mask = link_coef == link_coef[selected]
                    if candidates_mask.sum() == 1:
                        col_id = index[c_mask][selected].item()
                    else:
                        candidates = index[c_mask][candidates_mask]
                        col_id = candidates[torch.randperm(candidates.shape[0], device=self.device)[0]].item()
                    n_neighbor[row_id, c] += 1
                    n_neighbor[col_id, self.y_syn[row_id]] += 1
                    row.append(row_id)
                    row.append(col_id)
                    col.append(col_id)
                    col.append(row_id)
            rows.append(torch.LongTensor(row))
            cols.append(torch.LongTensor(col))
            batch.append(torch.LongTensor([i] * len(row)))

        self.rows = torch.cat(rows).to(self.device)
        self.cols = torch.cat(cols).to(self.device)
        self.batch = torch.cat(batch).to(self.device)

    def get_adj_t_syn(self):
        adj = torch.zeros(size=(self.batch_size, self.n_syn, self.n_syn), device=self.device)
        adj[self.batch, self.rows, self.cols] = torch.sigmoid(self.adj_mlp(
            torch.cat([self.x_syn[self.rows], self.x_syn[self.cols]], dim=1)).flatten())
        adj = (torch.transpose(adj, 1, 2) + adj) / 2

        adj += torch.eye(self.n_syn, device=self.device).view(1, self.n_syn, self.n_syn)
        deg = adj.sum(2)
        deg_inv = deg.pow(-1 / 2)
        deg_inv[torch.isinf(deg_inv)] = 0.
        adj = adj * deg_inv.view(self.batch_size, -1, 1)
        adj = adj * deg_inv.view(self.batch_size, 1, -1)
        return adj

    def __repr__(self):
        n_edge = self.rows.shape[0] / self.batch_size / 2
        return f'nodes:{self.x_syn.shape}\nnumber of nodes in each class:{self.n_each_y.tolist()}\nn_edge:{n_edge}'


class FixLenList:
    def __init__(self, lenth):
        self.lenth = lenth
        self.data = []

    def append(self, element):
        self.data.append(element)
        if len(self.data) > self.lenth:
            del self.data[0]
