import os
import time
import subprocess
from multiprocessing import Pool

methods = [
    'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge',
    'algebraic_JC', 'affinity_GS', 'kron', 'vng', 'clustering', 'averaging', 'gcond', 'doscond', 
    'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom', 'cadm', 
    'cent_d', 'cent_p', 'kcenter', 'herding', 'random', 'random_edge', 'tspanner'
]
standard_names = [
    'cora', 'citeseer', 'cora_ml', 'dblp', 'pubmed', 'photo', 'computers', 'cs', 
    'reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 
    'ogbn-papers100M', 'amazon'
]




# 生成当前时间戳
timestamp = time.strftime("%Y%m%d%H%M%S")

def get_free_gpu_memory():
    """Returns a list of available GPU memory."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE
    )
    return [int(x) for x in result.stdout.decode('utf-8').strip().split('\n')]

def run_task(method, standard_name):
    # log_file_path = f"/home/sqp17/Projects/GraphSlim/graphslim/logs/graphsage/{standard_name}/log_{method}.txt"
    # if os.path.exists(log_file_path):
    #     print(f"Log file {log_file_path} already exists. Skipping task: {method} on dataset: {standard_name}")
    #     return
    """Runs a task on an available GPU."""
    while True:
        free_memory = get_free_gpu_memory()
        for gpu_id, mem in enumerate(free_memory):
            if mem > 6000:  # 假设每个任务需要6000MB以上的空闲内存
                log_dir = f"logs/{timestamp}/{standard_name}"
                os.makedirs(log_dir, exist_ok=True)
                
                command = [
                    "python", "train_all.py", 
                    "--dataset", standard_name, 
                    "--method", method, 
                    "--verbose", 
                    "--gpu_id", str(gpu_id),
                    "--eval_model", 'SGFormer'
                ]
                
                log_file = os.path.join(log_dir, f"log_{method}.txt")
                print(f"Running task: {method} on dataset: {standard_name} using GPU {gpu_id}")

                # 启动任务并等待
                with open(log_file, "w") as f:
                    subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
                
                # 等待几分钟，让任务占用 GPU 资源，避免过多任务提交
                time.sleep(10)  # 等待 5 分钟
                return  # 任务已启动，退出循环
            
        print("No available GPU with enough memory, waiting...")
        time.sleep(300)  # 每60秒检查一次GPU状态

def main():
    tasks = []

    for standard_name in standard_names:
        for method in methods:
            tasks.append((method, standard_name))

    # 按顺序执行任务，并且让任务分批启动
    for task in tasks:
        run_task(*task)

if __name__ == "__main__":
    main()
