import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class PGE(nn.Module):
    """
    PGE (Parametrized Graph Encoder) is a neural network module designed to generate an adjacency matrix for a graph based on node features.
    Attributes:
        layers (nn.ModuleList): List of linear layers used in the network.
        bns (torch.nn.ModuleList): List of batch normalization layers.
        edge_index (np.ndarray): Array representing all possible edges in the graph.
        nnodes (int): Number of nodes in the graph.
        device (torch.device): Device on which the model is run.
        cnt (int): Counter for tracking operations.
        args (Namespace): Arguments containing dataset and reduction rate information.
    Methods:
        __init__(nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
            Initializes the PGE model with the given parameters.
        forward(x, inference=False):
            Forward pass of the model. Generates the adjacency matrix based on input node features.
        inference(x):
            Generates the adjacency matrix in inference mode without gradient computation.
        reset_parameters():
            Resets the parameters of the model's layers.
    """
    
    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        """
        Initializes the PGE model.
        Args:
            nfeat (int): Number of input features.
            nnodes (int): Number of nodes in the graph.
            nhid (int, optional): Number of hidden units. Default is 128.
            nlayers (int, optional): Number of layers in the model. Default is 3.
            device (torch.device, optional): Device on which to run the model. Default is None.
            args (argparse.Namespace, optional): Additional arguments, including dataset and reduction rate. Default is None.
        Attributes:
            layers (torch.nn.ModuleList): List of linear layers.
            bns (torch.nn.ModuleList): List of batch normalization layers.
            edge_index (numpy.ndarray): Array of edge indices.
            nnodes (int): Number of nodes in the graph.
            device (torch.device): Device on which to run the model.
            cnt (int): Counter initialized to 0.
            args (argparse.Namespace): Additional arguments.
        """
        
        super(PGE, self).__init__()
        # 设置nhid参数，取决于数据集类型和reduction_rate
        dataset_configs = {
            'ogbn-arxiv': {'nhid': 256, 'nlayers': 3},
            'arxiv': {'nhid': 256, 'nlayers': 3},
            'flickr': {'nhid': 256, 'nlayers': 3},
            'reddit': {'nhid': 128 if args.reduction_rate == 0.01 else 256, 'nlayers': 3}
        }
        config = dataset_configs.get(args.dataset, {'nhid': nhid, 'nlayers': nlayers})
        nhid = config['nhid']
        nlayers = config['nlayers']
    
        # 初始化线性层和批量归一化层
        self.layers = nn.ModuleList([nn.Linear(nfeat * 2, nhid)] + 
                                [nn.Linear(nhid, nhid) for _ in range(nlayers - 2)] +
                                [nn.Linear(nhid, 1)])
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(nhid) for _ in range(nlayers - 1)])


        # edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        
        self.edge_index = self.generate_edge_index(nnodes)
        self.nnodes = nnodes
        self.device = device
        self.cnt = 0
        self.args = args
        
        self.reset_parameters()
    
    def generate_edge_index(self, nnodes):
        row = torch.arange(nnodes).repeat(nnodes)
        col = torch.arange(nnodes).unsqueeze(1).repeat(1, nnodes).flatten()
        return torch.stack([row, col], dim=0)

    def forward(self, x, inference=False):
        """
        Perform a forward pass to compute the adjacency matrix.
        Args:
            x (torch.Tensor): Input node features.
            inference (bool, optional): Flag to indicate if the model is in inference mode. Defaults to False.
        Returns:
            torch.Tensor: The computed adjacency matrix.
        """
        
        edge_index = self.edge_index

        # 根据条件决定是否进行分块处理
        if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = [self._process_edges(x, edge_index, idx) for idx in splits]
            edge_embed = torch.cat(edge_embed)
        else:
            edge_embed = self._process_edges(x, edge_index)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)
        adj = torch.sigmoid((adj + adj.T) / 2)
        adj = adj - torch.diag(torch.diag(adj, 0))

        return adj

    def _process_edges(self, x, edge_index, idx=None):
        """ Helper function to process edges """
        if idx is not None:
            edge_pair = torch.cat([x[edge_index[0][idx]], x[edge_index[1][idx]]], axis=1)
        else:
            edge_pair = torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=1)


        for ix, (layer, bn) in enumerate(zip(self.layers[:-1], self.bns)):
            edge_pair = F.relu(bn(layer(edge_pair)))
        edge_pair = self.layers[-1](edge_pair)  # Last layer without batch norm and relu

        return edge_pair


    @torch.no_grad()
    def inference(self, x):
        """
        Perform inference to generate a synthetic adjacency matrix.
        Args:
            x (torch.Tensor): Input tensor for the model.
        Returns:
            torch.Tensor: The synthetic adjacency matrix generated by the model.
        """
        
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        This method iterates over all the modules in the model and resets the parameters
        of any `nn.Linear` or `nn.BatchNorm1d` layers to their initial values.
        The `weight_reset` function is applied to each module in the model using the 
        `self.apply` method.
        """
        
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

        self.apply(weight_reset)
