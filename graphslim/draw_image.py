# Modify your existing code by adding the color setting for x-axis labels

import pandas as pd
import os
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('extracted_log_data_standard_methods.csv')

# Clean the column names
df.columns = df.columns.str.strip()

# Replace '-' with 0
df.replace('-', 0, inplace=True)

# Convert data types to numeric
df['Condensed Graph Size (Mb)'] = pd.to_numeric(df['Condensed Graph Size (Mb)'], errors='coerce').fillna(0)
df['Test Mean Accuracy'] = pd.to_numeric(df['Test Mean Accuracy'], errors='coerce').fillna(0)
df['Function Time (ms)'] = pd.to_numeric(df['Function Time (ms)'], errors='coerce').fillna(0)

# Define the algorithm order
algorithm_order = [
    'affinity_GS', 'algebraic_JC', 'averaging', 'clustering', 'heavy_edge', 'kron',
    'variation_cliques', 'variation_edges', 'variation_neighborhoods', 'vng', 'doscond', 
    'doscondx', 'gcdm', 'gcdmx', 'gcond', 'gcondx', 'gcsntk', 'gdem', 'geom', 'msgc',
    'sfgc', 'sgdd', 'simgc', 'cent_d', 'cent_p', 'herding', 'kcenter', 'random',
    'random_edge'
]

# Set the 'Method' column as a categorical type with the specified order
df['Method'] = pd.Categorical(df['Method'], categories=algorithm_order, ordered=True)

# Get all unique dataset names
datasets = df['Dataset'].unique()
# Remove specific datasets from the list
datasets = [dataset for dataset in datasets if dataset not in ['yelp', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100m', 'amazon']]

# Define the color mapping
red_algorithms = [
    'affinity_GS', 'algebraic_JC', 'averaging', 'clustering', 'heavy_edge', 'kron',
    'variation_cliques', 'variation_edges', 'variation_neighborhoods', 'vng'
]

green_algorithms = [
    'doscond', 'doscondx', 'gcdm', 'gcdmx', 'gcond', 'gcondx', 'gcsntk', 'gdem', 'geom', 'msgc',
    'sfgc', 'sgdd', 'simgc'
]

blue_algorithms = [
    'cent_d', 'cent_p', 'herding', 'kcenter', 'random', 'random_edge'
]


data_dict = {
    "Dataset": [
        "cora", "citeseer", "cora_ml", "dblp", "pubmed", "photo",
        "computers", "cs",  "reddit", "flickr", 'ogbn-arxiv'
    ],
    "Original Graph Size (Mb)": [
        0.77, 1.7, 26.46,  89.75, 0.11,  19.77,  36.97,  382.3, 518.75, 88.78, 56.36
    ],
    "Reduction Rate": [
        0.5, 0.5, 0.1,0.1, 0.1, 0.1, 0.1, 0.1, 0.001, 0.01, 0.01
    ]
}



# Iterate over each dataset and generate the plots
for dataset in datasets:
    # Filter rows for the current dataset
    dataset_df = df[df['Dataset'] == dataset].sort_values(by='Method')  # Sort by algorithm order
    
    # Get algorithms and corresponding data
    algorithms = dataset_df['Method']
    condensed_graph_size = dataset_df['Condensed Graph Size (Mb)']
    test_mean_accuracy = dataset_df['Test Mean Accuracy']
    function_time = dataset_df['Function Time (ms)']
    
    # Create directory if it doesn't exist
    path = f'images/{dataset}'
    if not os.path.exists(path):
        os.makedirs(path)
        
    # Plot 1: Condensed Graph Size (Mb)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out 'random_edge' from algorithms and corresponding data，因为 random_edge 有一些错误
    filtered_algorithms = algorithms[algorithms != 'random_edge']
    filtered_condensed_graph_size = condensed_graph_size[algorithms != 'random_edge']
    
    ax.bar(filtered_algorithms.astype(str), filtered_condensed_graph_size, color='skyblue')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Condensed Graph Size (Mb)')
    original_size = data_dict["Original Graph Size (Mb)"][data_dict["Dataset"].index(dataset)] if dataset in data_dict["Dataset"] else 0
    reduction_rate = data_dict["Reduction Rate"][data_dict["Dataset"].index(dataset)] if dataset in data_dict["Dataset"] else 0
    ax.set_title(f'{dataset}: Original Size = {original_size} Mb, Reduction Rate = {reduction_rate}')
    ax.set_xticklabels(filtered_algorithms, rotation=45, ha='right')
    
    # Set x-tick label colors
    for tick_label, algorithm in zip(ax.get_xticklabels(), filtered_algorithms):
        if algorithm in red_algorithms:
            tick_label.set_color('red')
        elif algorithm in green_algorithms:
            tick_label.set_color('green')
        elif algorithm in blue_algorithms:
            tick_label.set_color('blue')
    
    plt.tight_layout()
    plt.savefig(f'images/{dataset}/Condensed_Graph_Size.jpg')
    plt.close()
    
    # Plot 2: Test Mean Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(algorithms.astype(str), test_mean_accuracy, color='lightgreen')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Test Mean Accuracy')
    original_size = data_dict["Original Graph Size (Mb)"][data_dict["Dataset"].index(dataset)]
    reduction_rate = data_dict["Reduction Rate"][data_dict["Dataset"].index(dataset)]
    ax.set_title(f'{dataset}: Original Size = {original_size} Mb, Reduction Rate = {reduction_rate}')
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Set x-tick label colors
    for tick_label, algorithm in zip(ax.get_xticklabels(), algorithms):
        if algorithm in red_algorithms:
            tick_label.set_color('red')
        elif algorithm in green_algorithms:
            tick_label.set_color('green')
        elif algorithm in blue_algorithms:
            tick_label.set_color('blue')
    
    plt.tight_layout()
    plt.savefig(f'images/{dataset}/Test_Mean_Accuracy.jpg')
    plt.close()
    
    # Plot 3: Function Time (ms) with logarithmic scale
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(algorithms.astype(str), function_time, color='lightcoral')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Function Time (ms)')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    original_size = data_dict["Original Graph Size (Mb)"][data_dict["Dataset"].index(dataset)]
    reduction_rate = data_dict["Reduction Rate"][data_dict["Dataset"].index(dataset)]
    ax.set_title(f'{dataset}: Original Size = {original_size} Mb, Reduction Rate = {reduction_rate}')
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Set x-tick label colors
    for tick_label, algorithm in zip(ax.get_xticklabels(), algorithms):
        if algorithm in red_algorithms:
            tick_label.set_color('red')
        elif algorithm in green_algorithms:
            tick_label.set_color('green')
        elif algorithm in blue_algorithms:
            tick_label.set_color('blue')
    
    plt.tight_layout()
    plt.savefig(f'images/{dataset}/Function_Time.jpg')
    plt.close()

print("All graphs have been saved successfully.")
