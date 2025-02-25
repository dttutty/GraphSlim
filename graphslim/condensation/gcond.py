from tqdm import trange

from graphslim.condensation.gcond_base import GCondBase
from graphslim.dataset.utils import save_reduced
from graphslim.evaluation.utils import verbose_time_memory
from graphslim.utils import *
from graphslim.models import *


class GCond(GCondBase):
    """
    "Graph Condensation for Graph Neural Networks" https://cse.msu.edu/~jinwei2/files/GCond.pdf
    """
    def __init__(self, setting, data, args, **kwargs):
        super(GCond, self).__init__(setting, data, args, **kwargs)

    @verbose_time_memory
    def reduce(self, data, verbose=True):
        args = self.args
        pge = self.pge
        # Convert synthetic features and labels to tensors
        self.feat_syn, labels_syn = to_tensor(self.feat_syn, label=data.labels_syn, device=self.device)
        
        # Convert full or training data to tensors based on the setting
        if args.setting == 'trans':
            features, adj, labels = to_tensor(data.feat_full, data.adj_full, label=data.labels_full, device=self.device)
        else:
            features, adj, labels = to_tensor(data.feat_train, data.adj_train, label=data.labels_train, device=self.device)

        # Initialize the synthetic features
        feat_init = self.initialize_synthetic_features()
        self.feat_syn.data.copy_(feat_init)

        # Normalize the adjacency matrix
        adj = normalize_adj_tensor(adj, sparse=True)

        # Get the number of outer and inner loops
        outer_loop, inner_loop = args.outer_loop, args.inner_loop
        loss_avg = 0
        best_val = 0
        
        # Initialize the model
        model = eval(args.condense_model)(self.d, args.hidden, data.nclass, args).to(self.device)
        
        for it in trange(args.epochs):
            model.initialize()
            model_parameters = list(model.parameters())
            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr)
            model.train()

            for ol in range(outer_loop):
                # Generate synthetic adjacency matrix
                adj_syn = pge(self.feat_syn)
                self.adj_syn = normalize_adj_tensor(adj_syn, sparse=False)
                
                # Check batch normalization
                model = self.check_bn(model)
                
                # Train the model and compute the loss
                loss = self.train_class(model, adj, features, labels, labels_syn, args)
                loss_avg += loss.item()

                # Zero the gradients
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()

                # Update the parameters
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                # Inner loop for model training
                feat_syn_inner = self.feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step()

            loss_avg /= (data.nclass * outer_loop)

            # Intermediate evaluation at checkpoints
            if it in args.checkpoints:
                self.adj_syn = adj_syn_inner
                data.adj_syn, data.feat_syn, data.labels_syn = self.adj_syn.detach(), self.feat_syn.detach(), labels_syn.detach()
                best_val = self.intermediate_evaluation(best_val, loss_avg)

        return data
