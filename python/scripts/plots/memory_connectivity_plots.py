"""Module for plotting memory and node connectivity data from HNSW index experiments."""
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scripts.plots.plot_utils import (load_csv_data, setup_plot_style, save_plot, set_axis_limits, apply_bold_styling)

CONFIG_INFO = {
    "00": "No neighbor update, No tombstones",
    "01": "No neighbor update, With tombstones",
    "10": "With neighbor update, No tombstones",
    "11": "With neighbor update, With tombstones",
    "-": ""
}


CONFIG_MAP = {
    "00": "1",
    "01": "2",
    "10": "3",
    "11": "4",
    "-": ""
}

EXPERIMENT_NAME_MAP = {
    "fullcoverage": "Full Coverage",
    "newdata": "New Data",
    "random": "Random",
    "unreachable_points_exclusive": "Unreachable Points",
    "unreachable": "Unreachable Points"
}

def get_implementation_label(impl_code, config):
    """Get the display label for an implementation based on code and config."""
    if impl_code == "RBC":
        return f"OBS-RU-B$_{{{CONFIG_MAP[config]}}}$"
    elif impl_code == "RFC":
        return f"OBS-RU-L$_{{{CONFIG_MAP[config]}}}$"
    elif impl_code == "hnswlib":
        return "HNSW-RU"
    elif impl_code == "MN-RU":
        return "MN-RU"
    elif impl_code == "MN-RBC":
        return r"O$\alpha$G-RU"
    elif impl_code == "usearch":
        return "D-RU"
    else:
        raise ValueError(f"Invalid implementation code: {impl_code}")

def plot_memory_usage(df: pd.DataFrame,
                     title: str,
                     save_dir: str,
                     filename: str,
                     experiment_paths: Dict[str, List[str]]):
    """Plot memory usage metrics over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots()
    

    # Plot each memory metric
    metrics = ['index_size', 'index_capacity', 'index_mem_usage']
    colors = ['blue', 'green', 'red']

    for metric, color in zip(metrics, colors):
        ax.plot(df.index, df[metric], label=metric, color=color)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Memory (bytes)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Set axis limits without extra space
    set_axis_limits(ax, df.index)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, filename, experiment_paths)

def plot_slot_distribution(df: pd.DataFrame,
                          title: str,
                          save_dir: str,
                          filename: str,
                          experiment_paths: Dict[str, List[str]]):
    """Plot slot distribution showing the distribution of slot types over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Stacked bar chart showing slot distribution
    x = range(len(df))
    bottom = np.zeros(len(df))

    # Plot deleted slots at the bottom
    ax.bar(x, df['slot_lookup_deleted_slots'], label='Deleted Slots', bottom=bottom)
    bottom += df['slot_lookup_deleted_slots']

    # Plot populated slots in the middle
    ax.bar(x, df['slot_lookup_populated_slots'], label='Populated Slots', bottom=bottom)
    bottom += df['slot_lookup_populated_slots']

    # Plot empty slots at the top
    empty_slots = df['slot_lookup_total_slots'] - df['slot_lookup_populated_slots'] - df['slot_lookup_deleted_slots']
    ax.bar(x, empty_slots, label='Empty Slots', bottom=bottom)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Slots')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits without extra space
    set_axis_limits(ax, pd.Series(x))

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_slot_distribution", experiment_paths)
    plt.close()

def plot_memory_usage_over_time(df: pd.DataFrame,
                            title: str,
                            save_dir: str,
                            filename: str,
                            experiment_paths: Dict[str, List[str]]):
    """Plot memory usage over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Line plot showing memory usage in megabytes
    ax.plot(range(len(df)), df['index_mem_usage'] / 1e6, label='Memory Usage', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Memory Usage (megabytes)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(range(len(df))), y_padding=0.4)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_memory_usage_over_time", experiment_paths)
    plt.close()

def plot_node_connectivity(df: pd.DataFrame,
                         dataset_name: str,
                         algorithm_name: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]],
                         config: str):
    """Plot non-level-specific node connectivity metrics over time."""
    setup_plot_style()

    # Define non-level specific metrics and their colors
    metrics = {
        'nodes_count': '#1f77b4',  # blue
        'unreachable_count': '#ff7f0e',  # orange
        'avg_connections': '#d62728'  # red
    }

    name_map = {
        'nodes_count': 'Number of Nodes',
        'unreachable_count': 'Unreachable Points',
        'avg_connections': 'Connectivity'
    }

    y_axis_label = {
        'nodes_count': 'Number of Nodes',
        'unreachable_count': 'Unreachable Points Count',
        'avg_connections': 'Mean Connectivity Count'
    }

    # Create iteration sequence
    iterations = range(len(df))

    # 1. Individual plots for each metric
    for metric, color in metrics.items():
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            if metric == 'unreachable_count':
                fig.set_size_inches(8, 6)

            ax.plot(iterations, df[metric], color=color, linewidth=2)

            ax.set_title(f'{name_map[metric]} Over Iterations - {get_implementation_label(algorithm_name, config)} - {EXPERIMENT_NAME_MAP[filename.split("_")[0]]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})')
            ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
            ax.set_ylabel(y_axis_label[metric], fontsize=plt.rcParams['axes.labelsize'])
            # Set tick label sizes explicitly
            ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
            
            # Use solid grid lines with higher alpha for better visibility
            ax.grid(True, alpha=0.5, linestyle='-')

            # Set axis limits with extra padding for better visualization
            # Add more padding for unreachable_count and avg_connections
            if metric == 'avg_connections':
                max_val = df[metric].max()
                # Add 20% padding at the top for better visualization
                padding = max_val * 0.2
                ax.set_ylim(bottom=0, top=max_val + padding)
                # Set axis limits starting at the first data point
                set_axis_limits(ax, df['iteration'], force_x_zero=True)
            else:
                y_padding = 0.4 if metric == 'unreachable_count' else 0.2
                set_axis_limits(ax, pd.Series(iterations), y_padding=y_padding)
                ax.set_ylim(bottom=0)
            
            apply_bold_styling(ax)
            save_plot(fig, save_dir, f"{filename}_{metric}", experiment_paths)
            plt.close()

def plot_level_connectivity(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot average connectivity by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for connectivity
    level_cols = [col for col in df.columns if col.startswith('avg_conn_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # distinct colors for each level

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]  # Extract level number
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Mean Node Connectivity', fontsize=plt.rcParams['axes.labelsize'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_connectivity", experiment_paths)
    plt.close()

def plot_level_unreachable(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot unreachable points by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for unreachable points
    level_cols = [col for col in df.columns if col.startswith('unreachable_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Number of Unreachable Points', fontsize=plt.rcParams['axes.labelsize'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_unreachable", experiment_paths)
    plt.close()

def plot_level_nodes(df: pd.DataFrame,
                   title: str,
                   save_dir: str,
                   filename: str,
                   experiment_paths: Dict[str, List[str]]):
    """Plot number of nodes by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for node counts
    level_cols = [col for col in df.columns if col.startswith('nodes_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Number of Nodes', fontsize=plt.rcParams['axes.labelsize'])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_nodes", experiment_paths)
    plt.close()

def plot_level_distances(df: pd.DataFrame,
                         dataset_name: str,
                         algorithm_name: str,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]],
                         config: str):
    """Plot average, minimum, and maximum distances by level over iterations."""

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Get level numbers
    level_nums = set()
    for col in df.columns:
        if col.startswith('avg_dist_l'):
            level_nums.add(col.split('_l')[1])
    level_nums = sorted(level_nums)

    # Create iteration sequence
    iterations = range(len(df))

    # 1. Average distances
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for level, color in zip(level_nums, colors):
        avg_col = f'avg_dist_l{level}'
        ax.plot(iterations, df[avg_col], label=f'Level {level}', color=color, linewidth=2)

    # Set labels and title - ensure consistent style
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Mean Euclidean Distance', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_title(title)

    # Set tick label sizes explicitly
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Use solid grid lines with higher alpha for better visibility
    ax.grid(True, alpha=0.5, linestyle='-')
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_avg_distances", experiment_paths)
    plt.close()

    # 2. Minimum distances
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for level, color in zip(level_nums, colors):
        min_col = f'min_dist_l{level}'
        ax.plot(iterations, df[min_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(f'Minimum Distances Between Neighbor Nodes by Level - {get_implementation_label(algorithm_name, config)}\n{EXPERIMENT_NAME_MAP[filename.split("_")[0]]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})')
    # Set labels and title - ensure consistent style
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Euclidean Distance', fontsize=plt.rcParams['axes.labelsize'])

    # Set tick label sizes explicitly
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Use solid grid lines with higher alpha for better visibility
    ax.grid(True, alpha=0.5, linestyle='-')
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_min_distances", experiment_paths)
    plt.close()

    # 3. Maximum distances
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for level, color in zip(level_nums, colors):
        max_col = f'max_dist_l{level}'
        ax.plot(iterations, df[max_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(f'Maximum Distances Between Neighbor Nodes by Level - {get_implementation_label(algorithm_name, config)}\n{EXPERIMENT_NAME_MAP[filename.split("_")[0]]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})')
    # Set labels and title - ensure consistent style
    ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Euclidean Distance', fontsize=plt.rcParams['axes.labelsize'])

    # Set tick label sizes explicitly
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Use solid grid lines with higher alpha for better visibility
    ax.grid(True, alpha=0.5, linestyle='-')
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_level_max_distances", experiment_paths)
    plt.close()

def plot_connectivity_scatter(df: pd.DataFrame,
                          search_df: pd.DataFrame,
                          title: str,
                          save_dir: str,
                          filename: str,
                          experiment_paths: Dict[str, List[str]]):
    """Plot scatter plot of connectivity vs search performance."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with color gradient based on recall
    scatter = ax.scatter(df['avg_connections'],
                        search_df['mean_computed_distances'],
                        c=search_df['mean_recall'],
                        cmap='viridis',
                        alpha=0.6)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Recall', rotation=270, labelpad=15, fontsize=plt.rcParams['axes.labelsize'])

    ax.set_xlabel('Mean Node Connectivity', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_ylabel('Number of Distances Computed at Search', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_title(title)

    # Set tick label sizes explicitly
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Use solid grid lines with higher alpha for better visibility
    ax.grid(True, alpha=0.5, linestyle='-')

    # Set axis limits without forcing x-axis to start at 0
    set_axis_limits(ax, df['avg_connections'], force_x_zero=False, y_padding=0.2)
    ax.ticklabel_format(style='plain', axis='y')

    apply_bold_styling(ax)
    save_plot(fig, save_dir, f"{filename}_node_connectivity_vs_distances_computed", experiment_paths)
    plt.close()

def generate_memory_connectivity_plots(experiment_paths: Dict[str, List[str]], config: str):
    """Generate all memory and connectivity related plots."""
    for scenario, paths in experiment_paths.items():
        # # Create scenario-level directory for combined plots
        # scenario_dir = os.path.dirname(paths[0])  # Get directory containing dataset paths
        # scenario_images_dir = os.path.join(scenario_dir, 'images')
        # os.makedirs(scenario_images_dir, exist_ok=True)

        # Generate individual dataset plots
        for dataset_path in paths:

            dataset_name = os.path.basename(dataset_path)
            algorithm_name = os.path.basename(dataset_path)
            # Extract dataset name from the folder path
            dataset_name = os.path.basename(dataset_path)
            algorithm_name = os.path.basename(dataset_path)

            if "fashion_mnist" in dataset_name:
                dataset_name = "fashion-mnist"
            elif "mnist" in dataset_name:
                dataset_name = "mnist"
            elif "sift" in dataset_name:
                dataset_name = "sift"
            elif "gist" in dataset_name:
                dataset_name = "gist"
            else:
                # throw error
                raise ValueError(f"Invalid dataset name: {dataset_name}")

            if "repl_cand" in algorithm_name:
                algorithm_name = "RBC"
            elif "reset_first" in algorithm_name:
                algorithm_name = "RFC"
            elif "MN_RU" in algorithm_name:
                algorithm_name = "MN-RU"
            elif "MN_RBC" in algorithm_name:
                algorithm_name = "MN-RBC"
            elif "usearch" in algorithm_name:
                algorithm_name = "usearch"
            elif "hnswlib" in algorithm_name:
                algorithm_name = "hnswlib"
            else:
                # throw error
                raise ValueError(f"Invalid algorithm name: {algorithm_name}")

            save_dir = os.path.join(dataset_path, "..", "..","..","..","figures", algorithm_name, scenario, dataset_name)
            if(algorithm_name == "RBC" or algorithm_name == "RFC"):
                save_dir = os.path.join(dataset_path,  "..", "..", "..","..","..","figures", algorithm_name, config, scenario, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            

            # # Memory stats plots
            # memory_stats_path = os.path.join(dataset_path, 'memory_stats.csv')
            # if os.path.exists(memory_stats_path):
            #     memory_stats_df = load_csv_data(memory_stats_path)

            #     plot_slot_distribution(
            #         memory_stats_df,
            #         f'Slot Distribution - {scenario.title()} ({dataset_name})',
            #         save_dir,
            #         f'{scenario}_memory_stats',
            #         experiment_paths
            #     )

            #     plot_memory_usage_over_time(
            #         memory_stats_df,
            #         f'Memory Usage Over Iterations - {scenario.title()} ({dataset_name})',
            #         save_dir,
            #         f'{scenario}_memory_stats',
            #         experiment_paths
            #     )

            # # Memory usage plots
            # memory_path = os.path.join(dataset_path, 'memory_usage.csv')
            # if os.path.exists(memory_path):
            #     memory_df = load_csv_data(memory_path)
            #     plot_memory_usage(
            #         memory_df,
            #         f'Memory Usage - {scenario.title()} ({dataset_name})',
            #         save_dir,
            #         f'{scenario}_memory_usage',
            #         experiment_paths
            #     )

            # Node connectivity plots
            connectivity_path = os.path.join(dataset_path, 'node_connectivity.csv')
            search_stats_path = os.path.join(dataset_path, 'search_query_stats.csv')

            if os.path.exists(connectivity_path):
                print(connectivity_path)
                connectivity_df = load_csv_data(connectivity_path)
                dataset_name = os.path.basename(dataset_path)
                if "fashion_mnist" in dataset_name:
                    dataset_name = "fashion-mnist"
                else:
                    if("cand" in dataset_name):
                        dataset_name = dataset_name.split('_')[3]
                    else:
                        dataset_name = dataset_name.split('_')[1]

                # Original connectivity plots
                plot_node_connectivity(
                    connectivity_df,
                    dataset_name,
                    algorithm_name,
                    save_dir,
                    f'{scenario}_node_connectivity',
                    experiment_paths,
                    config
                )

                # New level-specific plots
                plot_level_connectivity(
                    connectivity_df,
                    f'Node Connectivity by Level - {get_implementation_label(algorithm_name, config)} - {EXPERIMENT_NAME_MAP[scenario]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_unreachable(
                    connectivity_df,
                    f'Unreachable Points by Level - {get_implementation_label(algorithm_name, config)} - {EXPERIMENT_NAME_MAP[scenario]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_nodes(
                    connectivity_df,
                    f'Number of Nodes by Level - {get_implementation_label(algorithm_name, config)} - {EXPERIMENT_NAME_MAP[scenario]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_distances(
                    connectivity_df,
                    dataset_name,
                    algorithm_name,
                    f'Distance Between Neighbor Nodes by Level - {get_implementation_label(algorithm_name, config)}\n{EXPERIMENT_NAME_MAP[scenario]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths,
                    config
                )

                # Plot scatter and recall plots if search stats exist
                if os.path.exists(search_stats_path):
                    search_df = load_csv_data(search_stats_path)
                    # dataset_name = search_df['dataset'].iloc[0]
                    if len(search_df) == len(connectivity_df):
                        # New scatter plot with mean recall colormap
                        plot_connectivity_scatter(
                            connectivity_df,
                            search_df,
                            f'Node Connectivity vs Distances Computed at Search - {get_implementation_label(algorithm_name, config)}\n{EXPERIMENT_NAME_MAP[scenario]} ({"fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                            save_dir,
                            f'{scenario}',
                            experiment_paths
                        )

