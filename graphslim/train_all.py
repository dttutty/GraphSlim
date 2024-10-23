import os
import sys

if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))

from graphslim.config import get_args
from graphslim.dataset import *
from graphslim.evaluation import *
from graphslim.sparsification import *
from graphslim.condensation import *
from graphslim.coarsening import *
from graphslim.utils import to_camel_case, seed_everything


method_map = {
    'kcenter': (KCenter, KCenterAgg),
    'herding': (Herding, HerdingAgg),
    'random': (Random, None),
    'cent_p': (CentP, None),
    'cent_d': (CentD, None),
    'simgc': (SimGC, None),
    'gcond': (GCond, None),
    'doscond': (DosCond, None),
    'doscondx': (DosCondX, None),
    'gcondx': (GCondX, None),
    'sfgc': (SFGC, None),
    'sgdd': (SGDD, None),
    'gcsntk': (GCSNTK, None),
    'msgc': (MSGC, None),
    'geom': (GEOM, None),
    'vng': (VNG, None),
    'clustering': (Cluster, ClusterAgg),
    'averaging': (Average, None),
    'gcdm': (GCDM, None),
    'gcdmx': (GCDMX, None),
    'gdem': (GDEM, None),
    'variation_edges': (VariationEdges, None),
    'variation_neighborhoods': (VariationNeighborhoods, None),
    'algebraic_JC': (AlgebraicJc, None),
    'affinity_GS': (AffinityGs, None),
    't_spanner': (TSpanner, None)
}

if __name__ == '__main__':
    args = get_args()
    graph = get_dataset(args.dataset, args, args.load_path)
    seed_everything(args.seed)
    if args.attack is not None:
        data = attack(graph, args)

    if args.method in method_map:
        agent_class = method_map[args.method][1] if args.agg and method_map[args.method][1] else method_map[args.method][0]
        agent = agent_class(setting=args.setting, data=graph, args=args)
    else:
        agent = eval(to_camel_case(args.method))(setting=args.setting, data=graph, args=args)
    reduced_graph = agent.reduce(graph, verbose=args.verbose)
    evaluator = Evaluator(args)
    res_mean, res_std = evaluator.evaluate(reduced_graph, model_type=args.final_eval_model)
    # args.logger.info(f'Test Mean Accuracy: {100 * all_res[:, 0].mean():.2f} +/- {100 * all_res[:, 1].mean():.2f}')
