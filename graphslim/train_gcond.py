import os
import sys
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from configs import *
from evaluation.eval_agent import Evaluator
from graphslim.condensation import *
from graphslim.dataset import *
import logging
import time

args = cli(standalone_mode=False)

graph = get_dataset(args.dataset, args)
if args.method == 'gcond':
    agent = GCond(setting=args.setting, data=graph, args=args)
elif args.method == 'doscond':
    agent = DosCond(setting=args.setting, data=graph, args=args)
elif args.method in ['doscondx', 'gcondx']:
    agent = DosCondX(setting=args.setting, data=graph, args=args)
elif args.method == 'sfgc':
    agent = SFGC(setting=args.setting, data=graph, args=args)
elif args.method == 'sgdd':
    agent = SGDD(setting=args.setting, data=graph, args=args)
elif args.method == 'gcsntk':
    agent = GCSNTK(setting=args.setting, data=graph, args=args)
elif args.method == 'msgc':
    agent = MSGC(setting=args.setting, data=graph, args=args)
elif args.method == 'geom':
    agent = GEOM(setting=args.setting, data=graph, args=args)
start = time.perf_counter()
reduced_graph = agent.reduce(graph, verbose=args.verbose)
end = time.perf_counter()
args.logger.info(f'Function Time: {end - start}s')
# reduced_graph = graph
evaluator = Evaluator(args)
evaluator.evaluate(reduced_graph, model_type=args.eval_model)
