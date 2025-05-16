"""Module for generating comparison plots between different repl_cand_hnswlib configurations."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.plots.plot_utils import load_csv_data, setup_plot_style, save_plot

# Configuration names and descriptions
CONFIG_INFO = {
    "00": "No neighbor update, No tombstones",
    "01": "No neighbor update, With tombstones",
    "10": "With neighbor update, No tombstones",
    "11": "With neighbor update, With tombstones"
}

# Colors for each configuration
CONFIG_COLORS = {
    "00": "#1f77b4",  # Blue
    "01": "#ff7f0e",  # Orange
    "10": "#2ca02c",  # Green
    "11": "#d62728"   # Red
}

# Markers for each configuration
CONFIG_MARKERS = {
    "00": "o",
    "01": "s",
    "10": "^",
    "11": "D"
}

IMPLEMENTATIONS_NAME = {
    "hnswlib": "HNSWLib",
    "repl_cand_hnswlib": "RBC",
    "reset_first_candidate": "RFC",
    "usearch": "USearch"
}

def generate_config_comparison_plots(base_dir, output_dir):
    """Generate comparison plots between different configurations of repl_cand_hnswlib.

    Args:
        base_dir: Base directory containing embedded-c++
        output_dir: Directory to save comparison plots
    """
    # Path to repl_cand_hnswlib results
    repl_cand_base_dir = os.path.join(base_dir, "repl_cand_hnswlib", "results")
    
    # Configurations to compare (00, 01, 10, 11)
    configurations = ["00", "01", "10", "11"]
    
    # Experiment types
    experiments = ["fullcoverage", "newdata", "random"]
    
    # Create output directory if it doesn't exist
    config_comparison_dir = os.path.join(output_dir, "config_comparison")
    os.makedirs(config_comparison_dir, exist_ok=True)
    
    # For each experiment type
    for experiment in experiments:
        # Create experiment subdirectory
        experiment_dir = os.path.join(config_comparison_dir, experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Get all datasets from each configuration for this experiment
        all_datasets = {}
        for config in configurations:
            config_dir = os.path.join(repl_cand_base_dir, config, experiment)
            if not os.path.exists(config_dir):
                print(f"Warning: Directory {config_dir} does not exist")
                continue
                
            all_datasets[config] = []
            for folder in os.listdir(config_dir):
                folder_path = os.path.join(config_dir, folder)
                if os.path.isdir(folder_path):
                    all_datasets[config].append(folder)
        
        # Extract common dataset base names
        dataset_map = {}  # Maps normalized dataset name to full folder names per config
        
        # Function to extract core dataset name
        def extract_dataset_base(folder_name):
            # Example: repl_cand_hnswlib_fashion_mnist_10000q_100i_600r
            parts = folder_name.split('_')
            # Find the part that contains the actual dataset name (fashion_mnist, sift, etc.)
            for idx, part in enumerate(parts):
                if part in ["fashion", "mnist", "sift", "gist"]:
                    if part == "fashion":
                        return "fashion-mnist"
                    return part
            return None
        
        # Map each dataset to its folder in each configuration
        for config, folders in all_datasets.items():
            for folder in folders:
                base_name = extract_dataset_base(folder)
                if base_name:
                    if base_name not in dataset_map:
                        dataset_map[base_name] = {}
                    dataset_map[base_name][config] = folder
        
        # For each dataset, create comparison plots
        for dataset, config_folders in dataset_map.items():
            dataset_dir = os.path.join(experiment_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"Creating config comparison plots for {dataset} in {experiment}")
            
            # Generate all comparison plots for this dataset
            plot_comparison_metrics(
                repl_cand_base_dir, 
                configurations, 
                experiment, 
                dataset, 
                config_folders, 
                dataset_dir
            )

def plot_comparison_metrics(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Plot all comparison metrics for a given dataset across configurations."""
    # Plot recall comparison
    plot_recall_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)
    
    # Plot unreachable points comparison
    plot_unreachable_points_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)
    
    # Plot avg node connectivity comparison
    plot_avg_connectivity_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)
    
    # Plot benchmark comparisons
    plot_add_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)
    plot_search_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)
    plot_delete_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir)

def plot_recall_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing recall between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Path to the search_query_stats.csv file
        search_stats_path = os.path.join(base_dir, config, experiment, folder, 'search_query_stats.csv')
        
        if not os.path.exists(search_stats_path):
            print(f"Warning: File {search_stats_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(search_stats_path)
        
        if 'mean_recall' in df.columns:
            has_data = True
            # Plot with different colors and markers for different configurations
            ax.plot(df['iteration'], df['mean_recall'],
                   label=f"{CONFIG_INFO[config]}",
                   color=CONFIG_COLORS[config],
                   marker=CONFIG_MARKERS[config],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Recall')
        ax.set_title(f'Recall Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis range for recall to 0-1
        ax.set_ylim(0, 1.05)
        
        # Save plot
        save_plot(fig, output_dir, f"recall_comparison")
    else:
        print(f"No recall data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_unreachable_points_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing unreachable points between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Try both node_connectivity.csv and unreachable_points.csv
        connectivity_path = os.path.join(base_dir, config, experiment, folder, 'node_connectivity.csv')
        unreachable_path = os.path.join(base_dir, config, experiment, folder, 'unreachable_points.csv')
        
        # First try connectivity file
        if os.path.exists(connectivity_path):
            df = load_csv_data(connectivity_path)
            if 'unreachable_count' in df.columns:
                has_data = True
                ax.plot(df['iteration'], df['unreachable_count'],
                       label=f"{CONFIG_INFO[config]}",
                       color=CONFIG_COLORS[config],
                       marker=CONFIG_MARKERS[config],
                       markersize=6,
                       markevery=max(1, len(df)//10),
                       linewidth=2)
                continue
                
        # If not found, try dedicated unreachable_points file
        if os.path.exists(unreachable_path):
            df = load_csv_data(unreachable_path)
            if 'unreachable_points' in df.columns:
                has_data = True
                ax.plot(df['iteration'], df['unreachable_points'],
                       label=f"{CONFIG_INFO[config]}",
                       color=CONFIG_COLORS[config],
                       marker=CONFIG_MARKERS[config],
                       markersize=6,
                       markevery=max(1, len(df)//10),
                       linewidth=2)
                continue
                
        print(f"Warning: No unreachable points data found for {config} in {dataset}")
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Unreachable Points')
        ax.set_title(f'Unreachable Points Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"unreachable_points_comparison")
    else:
        print(f"No unreachable points data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_avg_connectivity_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing average node connectivity between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Path to the node_connectivity.csv file
        connectivity_path = os.path.join(base_dir, config, experiment, folder, 'node_connectivity.csv')
        
        if not os.path.exists(connectivity_path):
            print(f"Warning: File {connectivity_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(connectivity_path)
        
        if 'avg_connections' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['avg_connections'],
                   label=f"{CONFIG_INFO[config]}",
                   color=CONFIG_COLORS[config],
                   marker=CONFIG_MARKERS[config],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Node Connectivity')
        ax.set_title(f'Node Connectivity Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"avg_connectivity_comparison")
    else:
        print(f"No connectivity data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_add_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing add operation benchmark between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Path to the bm_add.csv file
        benchmark_path = os.path.join(base_dir, config, experiment, folder, 'bm_add.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=f"{CONFIG_INFO[config]}",
                   color=CONFIG_COLORS[config],
                   marker=CONFIG_MARKERS[config],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Add Operation Time Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"add_benchmark_comparison")
    else:
        print(f"No add benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_search_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing search operation benchmark between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Path to the bm_search.csv file
        benchmark_path = os.path.join(base_dir, config, experiment, folder, 'bm_search.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=f"{CONFIG_INFO[config]}",
                   color=CONFIG_COLORS[config],
                   marker=CONFIG_MARKERS[config],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Search Operation Time Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"search_benchmark_comparison")
    else:
        print(f"No search benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_delete_benchmark_comparison(base_dir, configurations, experiment, dataset, config_folders, output_dir):
    """Create plot comparing delete operation benchmark between configurations."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for config in configurations:
        # Skip if configuration doesn't have this dataset
        if config not in config_folders:
            continue
            
        # Get the folder for this configuration
        folder = config_folders[config]
        
        # Path to the bm_delete.csv file
        benchmark_path = os.path.join(base_dir, config, experiment, folder, 'bm_delete.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=f"{CONFIG_INFO[config]}",
                   color=CONFIG_COLORS[config],
                   marker=CONFIG_MARKERS[config],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Delete Operation Time Comparison - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"delete_benchmark_comparison")
    else:
        print(f"No delete benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()

# Main function to update generate_plots.py
def main():
    """Generate comparison plots for all configurations."""
    # Get the base directory for experiment results
    python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.join(os.path.dirname(python_dir), "embedded-c++")
    
    # Save directory for comparison plots
    output_dir = os.path.join(base_dir, 'comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the configuration comparison plots
    print("Generating comparison plots between configurations...")
    generate_config_comparison_plots(base_dir, output_dir)
    
    print("All configuration comparison plots generated successfully!")

if __name__ == "__main__":
    main()