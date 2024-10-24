import json
import os.path as osp
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon, Flickr, Reddit2
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from dgl.data import FraudDataset
import shutil
import gdown

from graphslim.dataset.convertor import edge_index_to_c_s_r, csr2ei, from_dgl
from graphslim.dataset.utils import split_dataset
from graphslim.utils import index_to_mask, to_tensor


def get_dataset(name='cora', args=None, load_path='../../data'):
    path = osp.join(load_path)
    
    # Create a dictionary that maps standard names to normalized names
    standard_names = ['flickr', 'reddit', 'dblp', 'cora_ml', 'physics', 'cs', 'cora', 'citeseer', 'pubmed', 'photo',
                      'computers', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-arxiv', 'yelp', 'amazon']
    normalized_names = [name.replace('-', '').replace('_', '') for name in standard_names]
    name_dict = dict(zip(normalized_names, standard_names))

    # Normalize the name input
    normalized_name = name.replace('-', '').replace('_', '')


    if normalized_name not in name_dict:
        raise ValueError("Dataset name not recognized.")
    
    name = name_dict[normalized_name]  # 转换为标准名称
    dataset = load_dataset_by_name(name, path)
    
    try:
        data = dataset[0]
    except KeyError:
        data = dataset

    # 添加分割信息
    data = split_dataset(data, args.split)

    # 进行TransAndInd处理
    data = TransductiveAndInductive(data, name, args.pre_norm)

    # 设置类数量
    try:
        data.nclass = dataset.num_classes
    except AttributeError:
        data.nclass = data.num_classes

    # 输出节点信息
    print_node_info(data)
    return data


def load_dataset_by_name(name, path):
    """根据数据集名称加载不同的数据集"""
    if name == 'flickr':
        return Flickr(root=osp.join(path, 'flickr'))
    elif name == 'reddit':
        return Reddit2(root=osp.join(path, 'reddit'))
    elif name in ['dblp', 'cora_ml', 'cora_full', 'citeseer_full']:
        return CitationFull(root=path, name=name)
    elif name in ['physics', 'cs']:
        return Coauthor(root=path, name=name)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Planetoid(root=path, name=name)
    elif name in ['photo', 'computers']:
        return Amazon(root=path, name=name)
    elif name == 'ogbn-arxiv':
        dataset = DataGraphSAINT(root=path, dataset=name)
        dataset.num_classes = 40
        return dataset
    elif name in ['ogbn-products', 'ogbn-proteins', 'ogbn-papers100M']:
        return PygNodePropPredDataset(name, root=path)
    elif name in ['yelp', 'amazon']:
        # dataset = pickle.load(open(f'{path}/{args.dataset}.dat', 'rb'))
        # dataset.num_classes = 2
        dataset = FraudDataset(name, raw_dir=path)
        return from_dgl(dataset[0], name=name, hetero=False)  # dgl2pyg
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def print_node_info(data):
    """打印训练、验证、测试集的节点数量信息"""
    print("train nodes num:", sum(data.train_mask).item())
    print("val nodes num:", sum(data.val_mask).item())
    print("test nodes num:", sum(data.test_mask).item())
    print("total nodes num:", data.x.shape[0])

class TransductiveAndInductive:
    """
    A class to handle data transformation and indexing for graph-based datasets.

    Attributes:
    -----------
    class_dict : dict or None
        Dictionary to store class-wise training data indices.
    samplers : list or None
        List of NeighborSampler objects for each class.
    class_dict2 : dict or None
        Dictionary to store class-wise indices for sampling during training.
    sparse_adj : SparseTensor or None
        Sparse adjacency matrix.
    adj_full : csr_matrix or None
        Full adjacency matrix.
    feat_full : torch.Tensor or None
        Full feature matrix.
    labels_full : torch.Tensor or None
        Full labels.
    num_nodes : int
        Number of nodes in the dataset.
    train_mask : torch.Tensor
        Mask for training data.
    val_mask : torch.Tensor
        Mask for validation data.
    test_mask : torch.Tensor
        Mask for test data.
    edge_index : torch.Tensor
        Edge indices.
    idx_train : torch.Tensor
        Indices for training data.
    idx_val : torch.Tensor
        Indices for validation data.
    idx_test : torch.Tensor
        Indices for test data.
    adj_train : csr_matrix
        Adjacency matrix for training data.
    adj_val : csr_matrix
        Adjacency matrix for validation data.
    adj_test : csr_matrix
        Adjacency matrix for test data.
    labels_train : torch.Tensor
        Labels for training data.
    labels_val : torch.Tensor
        Labels for validation data.
    labels_test : torch.Tensor
        Labels for test data.
    feat_train : torch.Tensor
        Features for training data.
    feat_val : torch.Tensor
        Features for validation data.
    feat_test : torch.Tensor
        Features for test data.

    Methods:
    --------
    __init__(self, data, dataset, norm=True):
        Initializes the TransAndInd object with data and dataset information.

    to(self, device):
        Moves data to the specified device.

    pyg_saint(self, data):
        Processes data in PyG or SAINT format.

    retrieve_class(self, c, num=256):
        Retrieves a specified number of samples from a given class.

    retrieve_class_sampler(self, c, adj, args, num=256):
        Retrieves a sampler for a given class.

    reset(self):
        Resets the samplers and class_dict2 attributes.
    """

    def __init__(self, data, dataset, norm=True):
        self.class_dict = None  # sample the training data per class when initializing synthetic graph
        self.samplers = None
        self.class_dict2 = None  # sample from the same class when training
        self.sparse_adj = None
        self.adj_full = None
        self.feat_full = None
        self.labels_full = None
        self.num_nodes = data.num_nodes
        self.train_mask, self.val_mask, self.test_mask = data.train_mask, data.val_mask, data.test_mask
        self.pyg_saint(data)
        if dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            self.edge_index = to_undirected(self.edge_index, self.num_nodes)
            feat_train = self.x[data.idx_train]
            scaler = StandardScaler()
            scaler.fit(feat_train)
            self.feat_full = scaler.transform(self.x)
            self.feat_full = torch.from_numpy(self.feat_full).float()
        if norm and dataset in ['cora', 'citeseer', 'pubmed']:
            self.feat_full = F.normalize(self.feat_full, p=1, dim=1)
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        # self.nclass = max(self.labels_full).item() + 1

        self.adj_train = self.adj_full[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj_full[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj_full[np.ix_(self.idx_test, self.idx_test)]

        self.labels_train = self.labels_full[self.idx_train]
        self.labels_val = self.labels_full[self.idx_val]
        self.labels_test = self.labels_full[self.idx_test]

        self.feat_train = self.feat_full[self.idx_train]
        self.feat_val = self.feat_full[self.idx_val]
        self.feat_test = self.feat_full[self.idx_test]

    def to(self, device):
        """
        Transfers specified attributes of the dataset to the given device.
        This method moves the following attributes to the specified device:
        'feat_full', 'labels_full', 'x', 'y', 'edge_index', 'feat_train', 
        'feat_val', 'feat_test'. It checks if each attribute exists before 
        attempting to move it to avoid errors due to uninitialized data.
        Args:
            device (torch.device): The device to which the attributes should be moved.
        Returns:
            self: The dataset object with its attributes moved to the specified device.
        """
        
        # 将所有需要迁移的属性放入一个列表中，避免重复代码
        attrs_to_move = [
            'feat_full', 'labels_full', 'x', 'y', 'edge_index',
            'feat_train', 'feat_val', 'feat_test'
        ]
        
        # attrs_to_move.extend(['labels_train', 'labels_val', 'labels_test'])

        for attr in attrs_to_move:
            # 检查属性是否存在，防止某些数据未初始化而引发错误
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))

        return self

    def pyg_saint(self, data):
        """
        Processes the input data and sets various attributes based on the format of the data.
        Parameters:
        data (object): The input data object. It can be in two formats:
            - PyG format: The data object should have attributes 'x', 'y', and 'edge_index'.
            - SAINT format: The data object should have attributes 'feat_full', 'labels_full', and 'adj_full'.
        Returns:
        object: The input data object.
        Attributes Set:
        - self.x: Node features (PyG format) or full features (SAINT format).
        - self.y: Node labels (PyG format) or full labels (SAINT format).
        - self.feat_full: Full node features.
        - self.labels_full: Full node labels.
        - self.adj_full: Full adjacency matrix.
        - self.edge_index: Edge index in COO format.
        - self.sparse_adj: Sparse adjacency matrix in SparseTensor format.
        """
        # Function implementation
        
        # reference type
        # pyg format use x,y,edge_index
        if hasattr(data, 'x'):
            # PyG 格式的数据处理
            self.process_pyg_format(data)
        elif hasattr(data, 'feat_full'):
            # SAINT 格式的数据处理
            self.process_saint_format(data)
        
        return data

    def process_pyg_format(self, data):
        """处理 PyG 格式的数据"""
        self.x = self.feat_full = data.x
        self.y = self.labels_full = data.y
        self.adj_full = edge_index_to_c_s_r(data.edge_index, data.x.shape[0])
        self.edge_index = data.edge_index
        self.sparse_adj = SparseTensor.from_edge_index(data.edge_index)

    def process_saint_format(self, data):
        """处理 SAINT 格式的数据"""
        self.adj_full = data.adj_full
        self.feat_full = self.x = data.feat_full
        self.labels_full = self.y = data.labels_full
        self.edge_index = csr2ei(data.adj_full)
        self.sparse_adj = SparseTensor.from_edge_index(self.edge_index)
    
    def retrieve_class(self, c, num=256):
        # change the initialization strategy here
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, args, num=256):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if args.setting == 'trans':
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                sizes = [15, 8]
            else:
                sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(adj,
                                                     node_idx=node_idx,
                                                     sizes=sizes, batch_size=num,
                                                     num_workers=8, return_e_id=False,
                                                     num_nodes=adj.size(0),
                                                     shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch.astype(np.int64))
        return out

    def reset(self):
        """
        Resets the synthetic data and samplers to None.

        This method sets the following attributes to None:
        - samplers: The samplers used for data loading.
        - class_dict2: A dictionary mapping classes.
        - labels_syn: Synthetic labels.
        - feat_syn: Synthetic features.
        - adj_syn: Synthetic adjacency matrix.
        """
        self.samplers = self.class_dict2 = self.labels_syn = self.feat_syn = self.adj_syn = None



class LargeDataLoader(nn.Module):
    def __init__(self, name='Flickr', split='train', batch_size=200, split_method='kmeans'):
        super(LargeDataLoader, self).__init__()
        path = osp.join('../../data')
        self.split_method = split_method

        
        # Load dataset based on the name
        if name.lower() == 'ogbn-arxiv':
            dataset = DataGraphSAINT(root=path, dataset=name)
            dataset.num_classes = 40
            data = dataset[0]
            features = to_tensor(data.feat_full)
            labels = data.labels_full
            edge_index = csr2ei(data.adj_full)
            self.split_idx = torch.tensor(data.idx_train)
            self.n_classes = dataset.num_classes
        else:
            dataset_cls = {'flickr': Flickr, 'reddit': Reddit2}.get(name.lower())
            if dataset_cls is None:
                raise ValueError(f"Unsupported dataset name: {name}")

            dataset = dataset_cls(root=osp.join(path, name))
            data = dataset[0]
            features = data.x
            labels = data.y
            edge_index = data.edge_index
            mask = f'{split}_mask'
            self.split_idx = torch.where(data[mask])[0]
            self.n_classes = dataset.num_classes
        
        # Common attributes
        self.n, self.dim = features.shape
        self.n_split = len(self.split_idx)
        self.k = int(round(self.n_split / batch_size))

        # Create adjacency matrix
        self.Adj = self.create_adjacency_matrix(edge_index, self.n)

        # Normalize features
        features = self.normalize_data(features)

        # Apply GCF if needed, in the original code, this is enabled for ogbn-arxiv only
        # features = self.GCF(self.Adj, features, k=1)

        self.split_feat = features[self.split_idx]
        self.split_label = labels[self.split_idx]

        # Masked adjacency for non-ogbn-arxiv datasets
        if name.lower() != 'ogbn-arxiv':
            self.Adj_mask = self.create_masked_adjacency(self.Adj, self.split_idx)
            # Optionally apply GCF on masked adjacency
            # self.split_feat = self.GCF(self.Adj_mask, self.split_feat, k=2)
            

    
    def create_adjacency_matrix(self, edge_index, num_nodes):
        values = torch.ones(edge_index.shape[1])
        Adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
        identity = torch.eye(num_nodes)
        return Adj + identity

    def create_masked_adjacency(self, Adj, split_idx):
        n_split = len(split_idx)
        optor_index = torch.stack([split_idx, torch.arange(n_split)])
        optor = torch.sparse_coo_tensor(optor_index, torch.ones(n_split), (self.n, n_split))
        return torch.sparse.mm(torch.sparse.mm(optor.t(), Adj), optor)


    
    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0)  
        std = data.std(dim=0)  
        std[std == 0] = 1  
        normalized_data = (data - mean) / std
        return normalized_data


    def graph_conv_filter(self, adj, x, k=2):
        """
        Graph Convolutional Filter (GCF) for propagating features through a graph.
        Parameters:
        adj (torch.sparse.FloatTensor): The adjacency matrix of the graph in sparse format.
        x (torch.Tensor): The feature matrix of the nodes.
        k (int, optional): The number of propagation steps. Default is 2.
        Returns:
        torch.Tensor: The propagated feature matrix after k steps.
        """
        
        # Get the number of nodes
        n = adj.shape[0]

        # Add self-loops to adjacency matrix
        adj = self.add_self_loops(adj, n)

        # Compute the degree matrix D^-1/2 for normalization
        D_inv_sqrt = torch.pow(torch.sparse.sum(adj, dim=1).to_dense(), -0.5)
        D_inv_sqrt = torch.diag(D_inv_sqrt)

        # Apply D^-1/2 * A * D^-1/2 normalization
        filter = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj), D_inv_sqrt)

        # Propagate features through the graph for k steps
        for _ in range(k):
            x = torch.sparse.mm(filter, x)

        return x

    def add_self_loops(self, adj, n):
        """
        Adds self-loops to the adjacency matrix.
        Parameters:
            adj: torch.Tensor, adjacency matrix
            n: int, number of nodes
        Returns:
            torch.Tensor, adjacency matrix with self-loops
        """
        ind = torch.arange(n).repeat(2, 1)
        identity = torch.sparse_coo_tensor(ind, torch.ones(n), (n, n))
        return adj + identity
    

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n

    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """
        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.k.item(), n_init=10)
            kmeans.fit(self.split_feat.numpy())
            self.batch_labels = kmeans.predict(self.split_feat.numpy())

    def getitem(self, idx):
        """
        对于给定的 idx 输出对应的 node_features, labels, sub Ajacency matrix
        """
        # idx   = [idx]
        n_idx = len(idx)
        idx_raw = self.split_idx[idx]
        feat = self.split_feat[idx]
        label = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1, n_idx), torch.tensor(range(n_idx)).reshape(1, n_idx)), dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n, n_idx])
        optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        idx = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i = self.getitem(idx)
        return batch_i


class DataGraphSAINT:
    '''datasets used in GraphSAINT paper'''

    def __init__(self, root, dataset, **kwargs):
        dataset = dataset.replace('-', '_')
        
        dataset_str = root + '/' + dataset + '/raw/'
        
        if not osp.exists(dataset_str):
            os.makedirs(dataset_str)
            print('Downloading dataset')
            url = 'https://drive.google.com/drive/folders/1VDobXR5KqKoov6WhYXFMwH4rN0FMnVOa'  # Change this to your actual file ID
            downloaded_folder=gdown.download_folder(url=url, output=dataset_str, quiet=False)
            if downloaded_folder and osp.isdir(downloaded_folder):
                for filename in os.listdir(downloaded_folder):
                    file_path = osp.join(downloaded_folder, filename)
                    shutil.move(file_path, dataset_str)

                shutil.rmtree(downloaded_folder)

        if dataset == 'ogbn_arxiv':
            dataset_str = dataset_str +'ogbn-arxiv/'
            self.adj_full = sp.load_npz(dataset_str  +'adj_full.npz')
            self.adj_full = self.adj_full + self.adj_full.T
            self.adj_full[self.adj_full > 1] = 1

        self.num_nodes = self.adj_full.shape[0]

        role = json.load(open(dataset_str + 'role.json', 'r'))
        self.idx_train = role['tr']
        self.idx_test = role['te']
        self.idx_val = role['va']
        self.train_mask = index_to_mask(self.idx_train, self.num_nodes)
        self.test_mask = index_to_mask(self.idx_test, self.num_nodes)
        self.val_mask = index_to_mask(self.idx_val, self.num_nodes)

        self.feat_full = np.load(dataset_str + 'feats.npy')
        # ---- normalize feat ----

        class_map = json.load(open(dataset_str + 'class_map.json', 'r'))
        self.labels_full = to_tensor(label=self.process_labels(class_map))

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.num_nodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int64)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def get(self, idx):
        return self

    def __getitem__(self, idx):
        return self.get(idx)
