"""Module for plotting memory and node connectivity data from HNSW index experiments."""
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scripts.plots.plot_utils import (load_csv_data, setup_plot_style, save_plot, set_axis_limits)

def plot_memory_usage(df: pd.DataFrame,
                     title: str,
                     save_dir: str,
                     filename: str):
    """Plot memory usage metrics over iterations."""
    fig, ax = plt.subplots()
    setup_plot_style()

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

    save_plot(fig, save_dir, filename)

def plot_slot_lookup_stats(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str):
    """Plot slot lookup statistics showing the distribution of slot types."""
    setup_plot_style()

    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 1. Stacked bar chart showing slot distribution
    x = range(len(df))
    bottom = np.zeros(len(df))

    # Plot deleted slots at the bottom
    ax1.bar(x, df['slot_lookup_deleted_slots'], label='Deleted Slots', bottom=bottom)
    bottom += df['slot_lookup_deleted_slots']

    # Plot populated slots in the middle
    ax1.bar(x, df['slot_lookup_populated_slots'], label='Populated Slots', bottom=bottom)
    bottom += df['slot_lookup_populated_slots']

    # Plot empty slots at the top
    empty_slots = df['slot_lookup_total_slots'] - df['slot_lookup_populated_slots'] - df['slot_lookup_deleted_slots']
    ax1.bar(x, empty_slots, label='Empty Slots', bottom=bottom)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Number of Slots')
    ax1.set_title('Slot Distribution Over Time')
    ax1.legend()
    ax1.grid(True)

    # Set axis limits without extra space
    set_axis_limits(ax1, pd.Series(x))

    # 2. Line plot showing memory usage
    ax2.plot(df.index, df['index_mem_usage'], label='Memory Usage')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Memory Usage (bytes)')
    ax2.set_title('Index Memory Usage Over Time')
    ax2.legend()
    ax2.grid(True)

    # Set axis limits without extra space
    set_axis_limits(ax2, df.index)

    # Adjust layout
    fig.set_constrained_layout(True)
    save_plot(fig, save_dir, filename)
    plt.close()

def plot_node_connectivity(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str):
    """Plot non-level-specific node connectivity metrics over time."""
    setup_plot_style()

    # Define non-level specific metrics and their colors
    metrics = {
        'nodes_count': '#1f77b4',  # blue
        'unreachable_count': '#ff7f0e',  # orange
        'disconnected_nodes': '#2ca02c',  # green
        'avg_connections': '#d62728'  # red
    }

    # Create iteration sequence
    iterations = range(len(df))

    # 1. Individual plots for each metric
    for metric, color in metrics.items():
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iterations, df[metric], color=color, linewidth=2)

            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

            # Set x-axis limits to start at 0 and end at max iteration
            ax.set_xlim(0, len(df) - 1)

            save_plot(fig, save_dir, f"{filename}_{metric}")
            plt.close()

    # 2. Combined plot with all metrics
    fig, ax = plt.subplots(figsize=(12, 6))

    for metric, color in metrics.items():
        if metric in df.columns:
            ax.plot(iterations, df[metric],
                   label=metric.replace('_', ' ').title(),
                   color=color,
                   linewidth=2)

    ax.set_title('Node Connectivity Metrics Over Time')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to start at 0 and end at max iteration
    ax.set_xlim(0, len(df) - 1)

    save_plot(fig, save_dir, f"{filename}_combined")
    plt.close()

def plot_connectivity_heatmap(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str):
    """Plot heatmaps of node connectivity metrics - average across all iterations, first iteration, and last iteration."""
    setup_plot_style()

    # Extract level-specific metrics (columns ending with _l0, _l1, _l2, etc.)
    level_cols = [col for col in df.columns if any(col.endswith(f'_l{i}') for i in range(10))]
    if not level_cols:
        print("No level-specific metrics found for heatmap")
        return

    # Group columns by metric type
    metrics = {}
    for col in level_cols:
        # Split from the right to handle metrics with underscores
        parts = col.rsplit('_l', 1)
        if len(parts) != 2:
            continue
        metric = parts[0]
        level = f'l{parts[1]}'

        if metric not in metrics:
            metrics[metric] = []
        metrics[metric].append((level, col))

    # Prepare data for heatmap
    metric_names = list(metrics.keys())
    levels = sorted(set(level for metric in metrics.values() for level, _ in metric))

    if not metric_names or not levels:
        print("No data available for heatmap")
        return

    def create_heatmap(data, subtitle, save_suffix):
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

        # Set labels
        ax.set_xticks(np.arange(len(metric_names)))
        ax.set_yticks(np.arange(len(levels)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
        ax.set_yticklabels([f'Level {l[1:]}' for l in levels])

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add value annotations
        for i in range(len(levels)):
            for j in range(len(metric_names)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black")

        ax.set_title(f'{title}\n{subtitle}')

        # Save plot
        save_plot(fig, save_dir, f"{filename}_level_metrics_heatmap_{save_suffix}")
        plt.close()

    # 1. Average across all iterations (original heatmap)
    data_avg = np.zeros((len(levels), len(metric_names)))
    for i, level in enumerate(levels):
        for j, metric in enumerate(metric_names):
            col = next((col for lvl, col in metrics[metric] if lvl == level), None)
            if col and col in df.columns:
                data_avg[i, j] = df[col].mean()
    create_heatmap(data_avg, "Average Across All Iterations", "avg")

    # 2. First iteration
    data_first = np.zeros((len(levels), len(metric_names)))
    for i, level in enumerate(levels):
        for j, metric in enumerate(metric_names):
            col = next((col for lvl, col in metrics[metric] if lvl == level), None)
            if col and col in df.columns:
                data_first[i, j] = df[col].iloc[0]
    create_heatmap(data_first, "First Iteration (Initial State)", "first")

    # 3. Last iteration
    data_last = np.zeros((len(levels), len(metric_names)))
    for i, level in enumerate(levels):
        for j, metric in enumerate(metric_names):
            col = next((col for lvl, col in metrics[metric] if lvl == level), None)
            if col and col in df.columns:
                data_last[i, j] = df[col].iloc[-1]
    create_heatmap(data_last, "Last Iteration (Final State)", "last")

def plot_level_connectivity(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str):
    """Plot average connectivity per level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for connectivity
    level_cols = [col for col in df.columns if col.startswith('avg_conn_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # distinct colors for each level

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]  # Extract level number
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Average Connectivity by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Connections')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to start at 0 and end at max iteration
    ax.set_xlim(0, len(df) - 1)

    save_plot(fig, save_dir, f"{filename}_level_connectivity")
    plt.close()

def plot_level_unreachable(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str):
    """Plot unreachable points per level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for unreachable points
    level_cols = [col for col in df.columns if col.startswith('unreachable_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Unreachable Points by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Unreachable Points')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to start at 0 and end at max iteration
    ax.set_xlim(0, len(df) - 1)

    save_plot(fig, save_dir, f"{filename}_level_unreachable")
    plt.close()

def plot_level_nodes(df: pd.DataFrame,
                   title: str,
                   save_dir: str,
                   filename: str):
    """Plot number of nodes per level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for node counts
    level_cols = [col for col in df.columns if col.startswith('nodes_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Number of Nodes by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Nodes')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set x-axis limits to start at 0 and end at max iteration
    ax.set_xlim(0, len(df) - 1)

    save_plot(fig, save_dir, f"{filename}_level_nodes")
    plt.close()

def plot_level_distances(df: pd.DataFrame,
                      title: str,
                      save_dir: str,
                      filename: str):
    """Plot average, minimum, and maximum distances per level over iterations."""
    setup_plot_style()
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
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        avg_col = f'avg_dist_l{level}'
        ax.plot(iterations, df[avg_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Average Distances by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, len(df) - 1)
    save_plot(fig, save_dir, f"{filename}_level_avg_distances")
    plt.close()

    # 2. Minimum distances
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        min_col = f'min_dist_l{level}'
        ax.plot(iterations, df[min_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Minimum Distances by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, len(df) - 1)
    save_plot(fig, save_dir, f"{filename}_level_min_distances")
    plt.close()

    # 3. Maximum distances
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        max_col = f'max_dist_l{level}'
        ax.plot(iterations, df[max_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title('Maximum Distances by Level')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, len(df) - 1)
    save_plot(fig, save_dir, f"{filename}_level_max_distances")
    plt.close()

def plot_connectivity_scatter(df: pd.DataFrame,
                          search_df: pd.DataFrame,
                          title: str,
                          save_dir: str,
                          filename: str):
    """Plot scatter plot of average connections vs node count, colored by mean recall."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Create scatter plot with avg_connections on x-axis and nodes_count on y-axis
    # Color points by mean recall instead of iteration
    scatter = ax.scatter(df['avg_connections'], df['nodes_count'],
                        alpha=0.6, c=search_df['mean_recall'], cmap='viridis')

    # Add colorbar to show mean recall values
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Recall')

    ax.set_xlabel('Average Connections per Node')
    ax.set_ylabel('Total Number of Nodes')
    ax.set_title('Node Count vs Average Connections\nColored by Mean Recall')
    ax.grid(True, alpha=0.3)

    save_plot(fig, save_dir, f"{filename}_connectivity_scatter")
    plt.close()

def plot_recall_vs_connectivity(df: pd.DataFrame,
                              search_df: pd.DataFrame,
                              title: str,
                              save_dir: str,
                              filename: str):
    """Plot mean recall vs connectivity metrics."""
    setup_plot_style()
    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Create scatter plots for different metrics
    metrics = {
        'avg_connections': ('Average Connections per Node', ax1),
        'nodes_count': ('Total Number of Nodes', ax2),
        'disconnected_nodes': ('Number of Disconnected Nodes', ax3),
        'unreachable_count': ('Number of Unreachable Points', ax4)
    }

    for metric, (metric_label, ax) in metrics.items():
        if metric in df.columns:
            scatter = ax.scatter(df[metric], search_df['mean_recall'],
                               alpha=0.6, c=range(len(df)), cmap='viridis')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Iteration')

            ax.set_xlabel(metric_label)
            ax.set_ylabel('Mean Recall')
            ax.grid(True, alpha=0.3)

    fig.suptitle('Mean Recall vs Connectivity Metrics')

    save_plot(fig, save_dir, f"{filename}_recall_vs_connectivity")
    plt.close()

def generate_memory_connectivity_plots(experiment_paths: Dict[str, List[str]]):
    """Generate all memory and connectivity related plots."""
    for scenario, paths in experiment_paths.items():
        for dataset_path in paths:
            save_dir = os.path.join(dataset_path, 'images')

            # Memory usage plots
            memory_path = os.path.join(dataset_path, 'memory_usage.csv')
            if os.path.exists(memory_path):
                memory_df = load_csv_data(memory_path)
                plot_memory_usage(
                    memory_df,
                    f'{scenario} - Memory Usage',
                    save_dir,
                    f'{scenario}_memory_usage'
                )

            # Slot lookup stats plots
            slot_lookup_path = os.path.join(dataset_path, 'slot_lookup_stats.csv')
            if os.path.exists(slot_lookup_path):
                slot_df = load_csv_data(slot_lookup_path)
                plot_slot_lookup_stats(
                    slot_df,
                    f'{scenario} - Slot Lookup Statistics',
                    save_dir,
                    f'{scenario}_slot_lookup'
                )

            # Node connectivity plots
            connectivity_path = os.path.join(dataset_path, 'node_connectivity.csv')
            search_stats_path = os.path.join(dataset_path, 'search_query_stats.csv')

            if os.path.exists(connectivity_path):
                connectivity_df = load_csv_data(connectivity_path)

                # Original connectivity plots
                plot_node_connectivity(
                    connectivity_df,
                    f'{scenario} - Node Connectivity',
                    save_dir,
                    f'{scenario}_node_connectivity'
                )

                plot_connectivity_heatmap(
                    connectivity_df,
                    f'{scenario} - Connectivity Heatmap',
                    save_dir,
                    f'{scenario}_connectivity_heatmap'
                )

                # New level-specific plots
                plot_level_connectivity(
                    connectivity_df,
                    f'{scenario} - Level Connectivity',
                    save_dir,
                    f'{scenario}'
                )

                plot_level_unreachable(
                    connectivity_df,
                    f'{scenario} - Level Unreachable',
                    save_dir,
                    f'{scenario}'
                )

                plot_level_nodes(
                    connectivity_df,
                    f'{scenario} - Level Nodes',
                    save_dir,
                    f'{scenario}'
                )

                plot_level_distances(
                    connectivity_df,
                    f'{scenario} - Level Distances',
                    save_dir,
                    f'{scenario}'
                )

                # Plot scatter and recall plots if search stats exist
                if os.path.exists(search_stats_path):
                    search_df = load_csv_data(search_stats_path)
                    if len(search_df) == len(connectivity_df):
                        # New scatter plot with mean recall colormap
                        plot_connectivity_scatter(
                            connectivity_df,
                            search_df,
                            f'{scenario} - Connectivity Scatter',
                            save_dir,
                            f'{scenario}'
                        )

                        plot_recall_vs_connectivity(
                            connectivity_df,
                            search_df,
                            f'{scenario} - Recall vs Connectivity',
                            save_dir,
                            f'{scenario}'
                        )
