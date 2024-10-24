import networkit as nk
import numpy as np
import torch
from graphslim.sparsification.edge_sparsification_base import EdgeSparsifier
from torch_geometric.utils.convert import from_networkit


class RandomEdge(EdgeSparsifier):
    def __init__(self, setting, data, args, **kwargs):
        super(RandomEdge, self).__init__(setting, data, args, **kwargs)

    def edge_cutter(self, G):
        args = self.args
        randomEdgeSparsifier = nk.sparsification.RandomEdgeSparsifier()


        # 赵雷的修正：获取原始图的边数量
        original_edge_count = G.numberOfEdges()
        # 赵雷的修正：根据 reduction_rate 计算要保留的边数量
        target_edge_count = int(original_edge_count * args.reduction_rate)
        # 赵雷的修正
        randomGraph = randomEdgeSparsifier.getSparsifiedGraph(G, args.reduction_rate)


        # randomGraph = randomEdgeSparsifier.getSparsifiedGraphOfSize(G, args.reduction_rate)#原代码错误地方
        if args.verbose:
            nk.overview(randomGraph)
        edge_index, edge_attr = from_networkit(randomGraph)

        return edge_index, edge_attr
