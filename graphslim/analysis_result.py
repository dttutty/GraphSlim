import os
import re
import csv

# 正则表达式来提取所需信息
# 调整正则表达式，将每一行信息单独提取
# 修改代码以符合新的路径要求，并遍历所有文件提取信息
methods = [
    'variation_neighborhoods', 'variation_edges', 'variation_cliques', 'heavy_edge',
    'algebraic_JC', 'affinity_GS', 'kron', 'vng', 'clustering', 'averaging', 'gcond', 'doscond', 
    'gcondx', 'doscondx', 'sfgc', 'msgc', 'disco', 'sgdd', 'gcsntk', 'geom', 'cadm', 
    'cent_d', 'cent_p', 'kcenter', 'herding', 'random', 'random_edge'
]

standard_names = [
    'cora', 'citeseer', 'cora_ml', 'dblp', 'pubmed', 'photo', 'computers', 'cs', 
    'reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 
    'ogbn-papers100m', 'amazon'
]

# 定义日志文件的根目录路径
log_root_dir = "/home/sqp17/Projects/GraphSlim/graphslim/logs/20241023055446"

# 重新定义正则表达式来提取所需信息
pattern_original = re.compile(r"Original graph:(\d+\.\d+) Mb")
pattern_condensed = re.compile(r"Condensed graph:(\d+\.\d+) Mb")
pattern_reduction = re.compile(r"'reduction_rate': (\d+\.\d+),")
pattern_seed_accuracy = re.compile(r"Seed:(\d+), Test Mean Accuracy: (\d+\.\d+)")
pattern_function_time = re.compile(r"Function Time: (\d+\.\d+) ms")

# 清空数据，重新提取
data = []

# 遍历所有的 standard_name 和 method 组合
for standard_name in standard_names:
    for method in methods:
        log_file = os.path.join(log_root_dir, standard_name, f"log_{method}.txt")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()

                # 使用正则表达式提取信息，找不到则使用默认值'-'
                original_graph_size = pattern_original.search(content)
                original_graph_size = original_graph_size.group(1) if original_graph_size else "-"

                condensed_graph_size = pattern_condensed.search(content)
                condensed_graph_size = condensed_graph_size.group(1) if condensed_graph_size else "-"

                reduction_rate = pattern_reduction.search(content)
                reduction_rate = reduction_rate.group(1) if reduction_rate else "-"

                seed_accuracy = pattern_seed_accuracy.search(content)
                seed = seed_accuracy.group(1) if seed_accuracy else "-"
                accuracy = seed_accuracy.group(2) if seed_accuracy else "-"

                function_time = pattern_function_time.search(content)
                function_time = function_time.group(1) if function_time else "-"
                if function_time != "-":
                    function_time = f"{float(function_time):.2f}"

                # 添加到结果列表
                data.append([standard_name, method, original_graph_size, condensed_graph_size, reduction_rate, seed, accuracy, function_time])

# 将结果保存到 CSV 文件
csv_file = "extracted_log_data_standard_methods.csv"
with open(csv_file, "w", newline='') as f:
    writer = csv.writer(f)
    # 写入表头
    writer.writerow(["Dataset", "Method", "Original Graph Size (Mb)", "Condensed Graph Size (Mb)", "Reduction Rate", "Seed", "Test Mean Accuracy", "Function Time (ms)"])
    # 写入数据
    writer.writerows(data)
