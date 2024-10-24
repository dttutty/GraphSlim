# 删除所有txt文件中包含“ModuleNotFoundError: No module named 'torch'”的文件
# find . -name "*.txt" | xargs grep -l "No such option: --device" | xargs rm

# 删除所有文件夹内没有文件的文件夹
# find . -type d -empty | xargs rm -rf
