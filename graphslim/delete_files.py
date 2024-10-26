import os

def delete_files_without_text(directory, target_text):
    # 遍历指定目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 检查是否是文本文件
            if file.endswith(".txt"):  # 这里假设只对.txt文件进行处理
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # 如果文件中不包含目标字符串，删除该文件
                        if target_text not in content:
                            os.remove(file_path)
                            print(f"已删除文件: {file_path}")
                except Exception as e:
                    print(f"处理文件时出错: {file_path}, 错误: {e}")

# 使用示例
directory = "/home/sqp17/Projects/GraphSlim/graphslim/logs/Cheby"  # 指定要搜索的目录路径
target_text = "Test Mean Accuracy"
delete_files_without_text(directory, target_text)
