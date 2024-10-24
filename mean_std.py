import re
import numpy as np
import os
import csv

# 定义日志文件目录和输出 CSV 文件路径
log_dir = '/home/sqp17/Projects/GraphSlim/graphslim/logs'
csv_output_path = os.path.join(log_dir, 'it_s_summary.csv')

# 初始化存储结果的列表
results = []

# 遍历日志目录下的所有 .log 文件
for filename in sorted(os.listdir(log_dir)):
    if filename.endswith('.txt'):
        log_path = os.path.join(log_dir, filename)
        
        # 从日志文件中读取内容
        with open(log_path, 'r') as file:
            log_content = file.read()
        
        # 使用正则表达式提取从 3/10 到 10/10 的 it/s 值
        pattern = r"(\d+\.\d{2})it/s"
        it_s_values = [float(match) for match i~n re.findall(pattern, log_content)[-18:]]
        
        # 计算平均值和标准差
        if it_s_values:
            mean_it_s = round(np.mean(it_s_values), 2)
            std_it_s = round(np.std(it_s_values), 2)
        else:
            mean_it_s = 0
            std_it_s = 0
        
        
        # 提取文件中的 Test Mean Accuracy
        accuracy_pattern = r"Seed:\d+,\s+Test Mean Accuracy:\s+(\d+\.\d{2})\s\+/-\s+(\d+\.\d{2})"
        accuracy_match = re.search(accuracy_pattern, log_content)
        
        if accuracy_match:
            mean_accuracy = round(float(accuracy_match.group(1)), 2)
            std_accuracy = round(float(accuracy_match.group(2)), 2)
        else:
            mean_accuracy = 0
            std_accuracy = 0
            
            
        # 将结果添加到列表中
        results.append([filename, mean_it_s, std_it_s, mean_accuracy, std_accuracy])

# 将结果写入 CSV 文件
with open(csv_output_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # 写入表头
    csv_writer.writerow(['Filename', 'Average it/s', 'Standard Deviation', 'Mean Accuracy', 'Standard Deviation Accuracy'])
    # 写入每个文件的结果
    csv_writer.writerows(results)

print(f"CSV file saved at: {csv_output_path}")
