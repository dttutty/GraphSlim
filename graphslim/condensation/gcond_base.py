from collections import Counter

import torch.nn as nn

from graphslim.coarsening import *
from graphslim.condensation.utils import *
from graphslim.models import *
from graphslim.sparsification import *
from graphslim.utils import *
from graphslim.dataset.utils import save_reduced


class GCondBase:
    """
    A base class for graph condition generation and training.

    Parameters
    ----------
    setting : str
        The setting for the graph condensation process.
    data : object
        The data object containing the dataset.
    args : Namespace
        Arguments and hyperparameters for the model and training process.
    **kwargs : keyword arguments
        Additional arguments for initialization.
    """
    
    METHODS_WITHOUT_LABELS = {'msgc'}
    METHODS_WITHOUT_PGE = {'sgdd', 'gcsntk', 'msgc'}

    def __init__(self, setting, data, args, **kwargs):
        """
        Initializes a GCondBase instance.

        Parameters
        ----------
        setting : str
            The type of experimental setting.
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.
        args : Namespace
            Arguments object containing hyperparameters for training and model.
        **kwargs : keyword arguments
            Additional optional parameters.
        """
        self.data = data
        self.args = args
        self.device = args.device
        self.setting = setting

        # Initialize synthetic labels if method requires it
        if args.method not in self.METHODS_WITHOUT_LABELS:
            self.labels_syn = self.data.labels_syn = self.create_synthetic_labels(data)
            n = self.nnodes_syn = self.data.labels_syn.shape[0]
        else:
            n = self.nnodes_syn = int(data.feat_train.shape[0] * args.reduction_rate)
        
        # Set the feature dimension
        self.d = d = data.feat_train.shape[1]
        
        # Log reduced sizes
        print(f'target reduced size:{int(data.feat_train.shape[0] * args.reduction_rate)}')
        print(f'actual reduced size:{n}')

        # Initialize synthetic features as a learnable parameter
        self.feat_syn = nn.Parameter(torch.empty(n, d).to(self.device))
        
        # Initialize PGE and optimizers if required by the method
        if args.method not in self.METHODS_WITHOUT_PGE:
            self.pge = PGE(nfeat=self.d, nnodes=self.nnodes_syn, device=self.device, args=args).to(self.device)
            self.adj_syn = None
            self._initialize_optimizers()
            

    def _initialize_optimizers(self):
        """Initializes optimizers for synthetic features and PGE."""
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=self.args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=self.args.lr_adj)
        self._log(f'adj_syn: {(self.nnodes_syn, self.nnodes_syn)}, feat_syn: {self.feat_syn.shape}')

    def _log(self, message):
        """Helper function to log messages."""
        if self.args.verbose:
            print(message)
            
    def reset_parameters(self, init_strategy='random_normal'):
        """
        Resets the parameters of the model.
        
        Parameters
        ----------
        init_strategy : str, optional
            The initialization strategy for synthetic features ('random_normal', 'xavier', 'kaiming', etc.).
        """
        # Initialize synthetic features based on the chosen strategy
        if init_strategy == 'xavier':
            nn.init.xavier_uniform_(self.feat_syn.data)
        elif init_strategy == 'kaiming':
            nn.init.kaiming_uniform_(self.feat_syn.data)
        else:
            # Default to normal random initialization
            self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

        # Reset PGE parameters if PGE exists
        if hasattr(self, 'pge') and self.pge is not None:
            self.pge.reset_parameters()


    def create_synthetic_labels(self, data):
        """
        Generates synthetic labels to match the target number of samples.

        Parameters
        ----------
        data : object
            The graph data object, which includes features, adjacency matrix, labels, etc.

        Returns
        -------
        np.ndarray
            A numpy array of synthetic labels.
        """
        counter = Counter(data.labels_train.tolist())
        n = len(data.labels_train)
        sorted_counter = sorted(counter.items(), key=lambda x: x[1])

        # Calculate total number of synthetic samples and initialize storage
        total_syn_samples = int(n * self.args.reduction_rate)
        labels_syn = []
        self.syn_class_indices = {}
        num_class_dict = {}
        
        
        sum_ = 0
        for ix, (class_label, num) in enumerate(sorted_counter):
            # For all but the last class
            if ix < len(sorted_counter) - 1:
                num_syn = max(int(num * self.args.reduction_rate), 1)
            else:
                # Last class takes the remaining samples
                num_syn = max(total_syn_samples - len(labels_syn), 1)
            # 使用独立函数更新索引和样本数
            self._update_class_info(class_label, num_syn, labels_syn, num_class_dict)
        
        # Update the data object
        self.data.num_class_dict = self.num_class_dict = num_class_dict
        self._log(f'Synthetic class distribution: {num_class_dict}')
        
        return np.array(labels_syn)


    def _update_class_info(self, class_label, num_syn, labels_syn, num_class_dict):
        """
        更新每个类的索引范围和类别样本数。

        Parameters
        ----------
        class_label : int
            类别标签。
        num_syn : int
            类别的合成样本数。
        labels_syn : list
            存储合成标签的列表。
        num_class_dict : dict
            存储每个类别对应样本数的字典。
        """
        self.syn_class_indices[class_label] = [len(labels_syn), len(labels_syn) + num_syn]
        labels_syn.extend([class_label] * num_syn)
        num_class_dict[class_label] = num_syn

    
    def initialize_synthetic_features(self, include_adjacency=False, keep_init=False):
        """
        Initializes synthetic features and (optionally) adjacency matrix.

        Parameters
        ----------
        with_adj : bool, optional
            Whether to initialize the adjacency matrix (default is False).

        Returns
        -------
        tuple
            A tuple containing the synthetic features and (optionally) the adjacency matrix.
        """
        args = self.args
        agent_classes = {
            'clustering': ClusterAgg if args.agg else Cluster,
            'averaging': Average,
            'kcenter': KCenter,
            'herding': Herding,
            'cent_p': CentP,
            'cent_d': CentD
        }
        agent = agent_classes.get(args.init, Random)(setting=args.setting, data=self.data, args=args)

        if keep_init:
            save_path = f'{args.save_path}/reduced_graph/{args.init}'
            if include_adjacency and os.path.exists(f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt'):
                feat_syn = torch.load(
                        f'{save_path}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
                return feat_syn, adj_syn
            if os.path.exists(f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt'):
                feat_syn = torch.load(
                        f'{save_path}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt', map_location=args.device)
                return feat_syn
        temp = args.method
        args.method = args.init
        reduced_data = agent.reduce(self.data, verbose=True, save=True)
        args.method = temp
        if include_adjacency:
            return reduced_data.feat_syn, reduced_data.adj_syn
        else:
            return reduced_data.feat_syn
            #return reduced_data.feat_syn_0, reduced_data.feat_syn_1,reduced_data.feat_syn_2



    def train_class(self, model, adj, features, labels, labels_syn, args, soft=True):
        """
        Trains the model and computes the loss.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.
        adj : torch.Tensor
            The adjacency matrix.
        features : torch.Tensor
            The feature matrix.
        labels : torch.Tensor
            The actual labels.
        labels_syn : torch.Tensor
            The synthetic labels.
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        data = self.data
        feat_syn = self.feat_syn
        adj_syn = self.adj_syn
        loss = torch.tensor(0.0, device=self.device)

        if not soft:
            loss_fn = F.nll_loss
            # Convert labels to class indices if they are one-hot encoded
            if labels.dim() > 1:
                hard_labels = torch.argmax(labels, dim=-1)
            else:
                hard_labels = labels.long()
            if labels_syn.dim() > 1:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
            else:
                hard_labels_syn = labels_syn.long()
        else:
            loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
            # Convert labels to one-hot encoding if they are class indices
            if labels.dim() == 1:
                hard_labels = labels
                soft_labels = F.one_hot(labels, num_classes=data.nclass).float()
            if labels_syn.dim() == 1:
                hard_labels_syn = labels
                soft_labels_syn = F.one_hot(labels_syn, num_classes=data.nclass).float()
            else:
                hard_labels_syn = torch.argmax(labels_syn, dim=-1)
                soft_labels_syn = labels_syn

        # Loop over each class
        for c in range(data.nclass):
            # Retrieve a batch of real data samples for class c
            batch_size, n_id, adjs = data.retrieve_class_sampler(c, adj, args)
            adjs = [adj[0].to(self.device) for adj in adjs]
            input_real = features[n_id].to(self.device)
            if soft:
                labels_real = soft_labels[n_id[:batch_size]].to(self.device)
            else:
                labels_real = hard_labels[n_id[:batch_size]].to(self.device)

            output_real = model(input_real, adjs)
            loss_real = loss_fn(output_real, labels_real)

            gw_real = torch.autograd.grad(loss_real, model.parameters(), retain_graph=True)
            gw_real = [g.detach().clone() for g in gw_real]

            output_syn = model(feat_syn, adj_syn)
            if soft:
                loss_syn = loss_fn(output_syn[hard_labels_syn == c], soft_labels_syn[hard_labels_syn == c])
            else:
                loss_syn = loss_fn(output_syn[hard_labels_syn == c], hard_labels_syn[hard_labels_syn == c])


            # Compute gradients w.r.t. model parameters for synthetic data
            gw_syn = torch.autograd.grad(loss_syn, model.parameters(), create_graph=True)

            # Compute matching loss between gradients
            coeff = self.num_class_dict[c] / self.nnodes_syn
            ml = match_loss(gw_syn, gw_real, args, device=self.device)
            loss += coeff * ml

        return loss

    def get_loops(self, args):
        # Get the two hyper-parameters of outer-loop and inner-loop.
        # The following values are empirically good.
        """
        Retrieves the outer-loop and inner-loop hyperparameters.

        Parameters
        ----------
        args : Namespace
            Arguments object containing hyperparameters for training and model.

        Returns
        -------
        tuple
            Outer-loop and inner-loop hyperparameters.
        """
        return args.outer_loop, args.inner_loop

    def check_bn(self, model):
        """
        Checks if the model contains BatchNorm layers and fixes their mean and variance after training.

        Parameters
        ----------
        model : torch.nn.Module
            The model object.

        Returns
        -------
        torch.nn.Module
            The model with BatchNorm layers fixed.
        """
        BN_flag = False
        for module in model.modules():
            if 'BatchNorm' in module._get_name():  # BatchNorm
                BN_flag = True
        if BN_flag:
            model.train()  # for updating the mu, sigma of BatchNorm
            # output_real = model.forward(features, adj)
            for module in model.modules():
                if 'BatchNorm' in module._get_name():  # BatchNorm
                    module.eval()  # fix mu and sigma of every BatchNorm layer
        return model

    def intermediate_evaluation(self, best_val, loss_avg=None, save=True, save_valid_acc=False):
        """
        Performs intermediate evaluation and saves the best model.
 
        Returns
        -------
        float
            The updated best validation accuracy.
        """
        data = self.data
        args = self.args
        if args.verbose:
            print('loss_avg: {}'.format(loss_avg))

        res = []

        for i in range(args.run_inter_eval):
            res.append(
                self.test_with_val(verbose=False, setting=args.setting, iters=args.eval_epochs))

        res = np.array(res).T
        current_val = res[0].mean()
        args.logger.info('\nVal:  {:.4f} +/- {:.4f}'.format(100*current_val, 100*res[0].std()))
        args.logger.info('Test: {:.4f} +/- {:.4f}'.format(100*res[1].mean(), 100*res[1].std()))

        if save and current_val > best_val:
            best_val = current_val
            save_reduced(data.adj_syn, data.feat_syn, data.labels_syn, args)
        return best_val

    def test_with_val(self, verbose=False, setting='trans', iters=200, best_val=None):
        """
        Conducts validation testing and returns results.

        Returns
        -------
        list
            A list containing validation results.
        """

        args, data, device = self.args, self.data, self.device

        # Initialize model
        model = self._initialize_model(args.final_eval_model, data.feat_syn.shape[1], args.hidden, data.nclass, args).to(device)

        # Train with validation
        acc_val = self._train_model_with_val(model, data, iters, setting, best_val)

        # Test model performance
        acc_test = self._test_model(model, data, setting)

        if verbose:
            self._log_results(acc_val, acc_test)
        
        return [acc_val, acc_test]

    def _train_model_with_val(self, model, data, iters, setting, best_val):
        """
        Trains the model and performs validation. 
        
        Returns
        -------
        float
            The validation accuracy after training.
        """
        return model.fit_with_val(data, train_iters=iters, normadj=True, verbose=False, setting=setting, reduced=True, best_val=best_val)

    def _log_results(self, acc_val, acc_test):
        print(f"Validation Accuracy: {acc_val}")
        print(f"Test Accuracy: {acc_test}")

    def _test_model(self, model, data, setting):
        """
        Tests the model performance. 

        Returns
        -------
        float
            The test accuracy.
        """
        model.eval()
        return model.test(data, setting=setting, verbose=False)

    def _initialize_model(self, model_name, input_dim, hidden_dim, output_dim, args):
        """
        Initializes the model based on the provided model name.
        """
        try:
            model_class = getattr(__import__('models', fromlist=[model_name]), model_name)
        except AttributeError:
            raise ValueError(f"Model '{model_name}' not found.")
        return model_class(input_dim, hidden_dim, output_dim, args, mode='eval')
