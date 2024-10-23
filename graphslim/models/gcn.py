import torch.nn as nn

from graphslim.models.base import BaseGNN
from graphslim.models.layers import GraphConvolution
from graphslim.utils import *
from torch_geometric.nn import GCNConv

class GCN(BaseGNN):
    def __init__(self, nfeat, nhid, nclass, args, mode='train'):
        super(GCN, self).__init__(nfeat, nhid, nclass, args, mode)

        # 如果只有一层
        if self.nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass))
        else:
            self._build_layers(nfeat, nhid, nclass)

    def _build_layers(self, nfeat, nhid, nclass):
        """构建多层GCN，支持批归一化"""
        if self.with_bn:
            self.bns = torch.nn.ModuleList()

        # 第一层
        self.layers.append(GraphConvolution(nfeat, nhid))
        if self.with_bn:
            self.bns.append(nn.BatchNorm1d(nhid))

        # 中间隐藏层
        for _ in range(self.nlayers - 2):
            self.layers.append(GraphConvolution(nhid, nhid))
            if self.with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))

        # 最后一层
        self.layers.append(GraphConvolution(nhid, nclass))
