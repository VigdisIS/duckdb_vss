"""Module for comparing hnswlib with each repl_cand_hnswlib configuration."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.plots.plot_utils import load_csv_data, setup_plot_style, save_plot

# Configuration descriptions
CONFIG_INFO = {
    "00": "No neighbor update, No tombstones",
    "01": "No neighbor update, With tombstones",
    "10": "With neighbor update, No tombstones",
    "11": "With neighbor update, With tombstones"
}

# Colors for each implementation
COLORS = {
    "hnswlib": "#1f77b4",  # Blue
    "repl_cand": "#ff7f0e"  # Orange
}

# Markers for each implementation
MARKERS = {
    "hnswlib": "o",
    "repl_cand": "s"
}

DIRECTORY_MAP = {
    "RBC": "repl_cand_hnswlib",
    "RFC": "reset_first_candidate"
}

REVERSE_IMPL_MAP = {
    "repl_cand": "RBC",
    "reset_first_candidate": "RFC"
}

def generate_hnswlib_vs_configs(base_dir, output_dir, implementations):
    """Generate comparison plots between hnswlib and each repl_cand_hnswlib configuration.
    
    Args:
        base_dir: Base directory containing embedded-c++
        output_dir: Directory to save comparison plots
    """

    configs = ["00", "01", "10", "11"]

    if (implementations[1] == "RFC"):
        configs = ["01"]
    
    # Define experiment types to compare
    experiments = ["fullcoverage", "newdata", "random"]
    
    # Output directory for hnswlib vs configs comparisons
    hnswlib_vs_configs_dir = os.path.join(output_dir, f"hnswlib_vs_{implementations[1]}")
    os.makedirs(hnswlib_vs_configs_dir, exist_ok=True)
    
    # Path to hnswlib results
    hnswlib_base_dir = os.path.join(base_dir, "hnswlib", "results")
    
    # For each configuration
    for config in configs:
        # Create config directory
        config_dir = os.path.join(hnswlib_vs_configs_dir, config)
        os.makedirs(config_dir, exist_ok=True)
        
        # Path to this repl_cand_hnswlib configuration results
        repl_cand_config_dir = os.path.join(base_dir, DIRECTORY_MAP[implementations[1]], "results", config)
        
        # For each experiment type
        for experiment in experiments:
            # Create experiment subdirectory
            experiment_dir = os.path.join(config_dir, experiment)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Get all dataset folders from both implementations for this experiment
            all_folders = []
            
            # Get hnswlib folders
            hnswlib_exp_dir = os.path.join(hnswlib_base_dir, experiment)
            if os.path.exists(hnswlib_exp_dir):
                for folder in os.listdir(hnswlib_exp_dir):
                    folder_path = os.path.join(hnswlib_exp_dir, folder)
                    if os.path.isdir(folder_path) and not folder == 'images':
                        all_folders.append(("hnswlib", folder))
            
            # Get repl_cand_hnswlib folders for this config
            if (implementations[1] == "RBC"):
                repl_cand_exp_dir = os.path.join(repl_cand_config_dir, experiment)
                if os.path.exists(repl_cand_exp_dir):
                    for folder in os.listdir(repl_cand_exp_dir):
                        folder_path = os.path.join(repl_cand_exp_dir, folder)
                        if os.path.isdir(folder_path) and not folder == 'images':
                            all_folders.append(("repl_cand", folder))
            else:
                repl_cand_exp_dir = os.path.join(repl_cand_config_dir, experiment)
                if os.path.exists(repl_cand_exp_dir):
                    for folder in os.listdir(repl_cand_exp_dir):
                        folder_path = os.path.join(repl_cand_exp_dir, folder)
                        if os.path.isdir(folder_path) and not folder == 'images':
                            all_folders.append(("reset_first_candidate", folder))
            
            # Function to extract core dataset name
            def extract_dataset_base(folder_name):
                # Remove prefixes
                cleaned_name = folder_name
                for prefix in ['hnswlib_', 'repl_cand_', 'hnswlib_repl_cand_', 'repl_cand_hnswlib_', 'reset_first_candidate_']:
                    if cleaned_name.startswith(prefix):
                        cleaned_name = cleaned_name[len(prefix):]
                
                # Extract dataset name from remaining parts
                parts = cleaned_name.split('_')
                for idx, part in enumerate(parts):
                    if part in ["fashion", "mnist", "sift", "gist"]:
                        if part == "fashion":
                            return "fashion-mnist"
                        return part
                return None
            
            # Map datasets to folders for each implementation
            dataset_map = {}  # Maps dataset name to folders for each implementation
            
            for impl, folder in all_folders:
                base_name = extract_dataset_base(folder)
                if base_name:
                    if base_name not in dataset_map:
                        dataset_map[base_name] = {}
                    dataset_map[base_name][impl] = folder
            
            if (implementations[1] == "RBC"):
                prefix = "repl_cand"
            else:
                prefix = "reset_first_candidate"

            # For each dataset, create comparison plots
            for dataset, impl_folders in dataset_map.items():
                # Only process datasets that exist in both implementations
                if "hnswlib" in impl_folders and f"{prefix}" in impl_folders:
                    dataset_dir = os.path.join(experiment_dir, dataset)
                    os.makedirs(dataset_dir, exist_ok=True)
                    print(f"Creating hnswlib vs config {config} plots for {dataset} in {experiment}")
                    
                    # Generate all comparison plots for this dataset
                    plot_hnswlib_vs_config_metrics(
                        base_dir, 
                        prefix,
                        config,
                        experiment, 
                        dataset, 
                        impl_folders, 
                        dataset_dir
                    )

def plot_hnswlib_vs_config_metrics(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Plot all comparison metrics between hnswlib and a repl_cand_hnswlib configuration."""
    # Plot recall comparison
    plot_hnswlib_vs_config_recall(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)
    
    # Plot unreachable points comparison
    plot_hnswlib_vs_config_unreachable(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)
    
    # Plot avg node connectivity comparison
    plot_hnswlib_vs_config_connectivity(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)
    
    # Plot benchmark comparisons
    plot_hnswlib_vs_config_add_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)
    plot_hnswlib_vs_config_search_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)
    plot_hnswlib_vs_config_delete_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir)

def plot_hnswlib_vs_config_recall(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing recall between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Path to the search_query_stats.csv file
        search_stats_path = os.path.join(results_path, 'search_query_stats.csv')
        
        if not os.path.exists(search_stats_path):
            print(f"Warning: File {search_stats_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(search_stats_path)
        
        if 'mean_recall' in df.columns:
            has_data = True
            # Plot with different colors and markers
            ax.plot(df['iteration'], df['mean_recall'],
                   label=label,
                   color=COLORS[impl],
                   marker=MARKERS[impl],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Recall')
        ax.set_title(f'Recall: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis range for recall to 0-1
        ax.set_ylim(0, 1.05)
        
        # Save plot
        save_plot(fig, output_dir, f"recall_comparison")
    else:
        print(f"No recall data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_hnswlib_vs_config_unreachable(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing unreachable points between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Try both node_connectivity.csv and unreachable_points.csv
        connectivity_path = os.path.join(results_path, 'node_connectivity.csv')
        unreachable_path = os.path.join(results_path, 'unreachable_points.csv')
        
        # First try connectivity file
        if os.path.exists(connectivity_path):
            df = load_csv_data(connectivity_path)
            if 'unreachable_count' in df.columns:
                has_data = True
                ax.plot(df['iteration'], df['unreachable_count'],
                       label=label,
                       color=COLORS[impl],
                       marker=MARKERS[impl],
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
                       label=label,
                       color=COLORS[impl],
                       marker=MARKERS[impl],
                       markersize=6,
                       markevery=max(1, len(df)//10),
                       linewidth=2)
                continue
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Unreachable Points')
        ax.set_title(f'Unreachable Points: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"unreachable_points_comparison")
    else:
        print(f"No unreachable points data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_hnswlib_vs_config_connectivity(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing average node connectivity between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Path to the node_connectivity.csv file
        connectivity_path = os.path.join(results_path, 'node_connectivity.csv')
        
        if not os.path.exists(connectivity_path):
            print(f"Warning: File {connectivity_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(connectivity_path)
        
        if 'avg_connections' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['avg_connections'],
                   label=label,
                   color=COLORS[impl],
                   marker=MARKERS[impl],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Node Connectivity')
        ax.set_title(f'Node Connectivity: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"avg_connectivity_comparison")
    else:
        print(f"No connectivity data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_hnswlib_vs_config_add_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing add operation benchmark between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Path to the bm_add.csv file
        benchmark_path = os.path.join(results_path, 'bm_add.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=label,
                   color=COLORS[impl],
                   marker=MARKERS[impl],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Add Operation Time: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"add_benchmark_comparison")
    else:
        print(f"No add benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_hnswlib_vs_config_search_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing search operation benchmark between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Path to the bm_search.csv file
        benchmark_path = os.path.join(results_path, 'bm_search.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=label,
                   color=COLORS[impl],
                   marker=MARKERS[impl],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Search Operation Time: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"search_benchmark_comparison")
    else:
        print(f"No search benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()

def plot_hnswlib_vs_config_delete_benchmark(base_dir, implementation, config, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing delete operation benchmark between hnswlib and a repl_cand_hnswlib configuration."""
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_plot_style()
    
    has_data = False
    
    for impl, label in [("hnswlib", "HNSWLib"), (implementation, f"{REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}]")]:
        if impl not in impl_folders:
            continue
            
        # Get correct path to results directory
        if impl == "hnswlib":
            results_path = os.path.join(base_dir, impl, "results", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[REVERSE_IMPL_MAP[implementation]], "results", config, experiment, impl_folders[impl])
        
        # Path to the bm_delete.csv file
        benchmark_path = os.path.join(results_path, 'bm_delete.csv')
        
        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue
            
        # Load CSV data
        df = load_csv_data(benchmark_path)
        
        if 'mean_time' in df.columns:
            has_data = True
            ax.plot(df['iteration'], df['mean_time'],
                   label=label,
                   color=COLORS[impl],
                   marker=MARKERS[impl],
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)
    
    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Delete Operation Time: HNSWLib vs {REVERSE_IMPL_MAP[implementation]} [{CONFIG_INFO[config]}] - {experiment.title()} ({dataset})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)
        
        # Save plot
        save_plot(fig, output_dir, f"delete_benchmark_comparison")
    else:
        print(f"No delete benchmark data to plot for {dataset} in {experiment}")
        
    plt.close()