"""Module for plotting benchmark data from HNSW index experiments."""

import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.plots.plot_utils import (load_csv_data, calculate_error_bounds,
                        plot_with_error_bounds, setup_plot_style, save_plot, set_axis_limits, apply_bold_styling)

# Configuration descriptions
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
    "unreachable_points_exclusive": "Unreachable Points"
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

def plot_benchmark_metrics(df: pd.DataFrame,
                         metric: str,
                         title: str,
                         ylabel: str,
                         save_dir: str,
                         filename: str):
    """Plot benchmark metrics with error bounds."""
    try:
        # Validate inputs
        if df.empty:
            print(f"Warning: Empty DataFrame for {filename}")
            return

        # Check if required columns exist
        required_cols = ['iteration', f'mean_{metric}']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns for {filename}")
            return

        # Apply the plot style
        setup_plot_style()

        # Create figure with specific size to match the good example
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate error bounds
        try:
            lower_bound, upper_bound = calculate_error_bounds(
                df,
                f'mean_{metric}',
                std_col=f'stddev_{metric}'
            )
        except ValueError as e:
            print(f"Warning: Could not calculate error bounds for {filename}: {str(e)}")
            return

        # Plot with error bounds for mean - use blue color for consistency
        plot_with_error_bounds(
            ax,
            df['iteration'],
            df[f'mean_{metric}'],
            lower_bound,
            upper_bound,
            label=f'Mean {metric.title()}',
            color='blue'
        )

        # Add median line if median column exists - use red dashed line
        if f'median_{metric}' in df.columns:
            ax.plot(df['iteration'], df[f'median_{metric}'],
                   label=f'Median {metric.title()}',
                   color='red',
                   linestyle='--',
                   linewidth=2)

        # Set labels and title - ensure consistent style
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel(ylabel, fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(title)

        # Set tick label sizes explicitly
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Use solid grid lines with higher alpha for better visibility
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Set axis limits starting at the first data point
        set_axis_limits(ax, df['iteration'], force_x_zero=True)

        # Disable scientific notation on y-axis for consistency
        ax.ticklabel_format(style='plain', axis='y')

        # Save with consistent settings
        apply_bold_styling(ax)
        save_plot(fig, save_dir, filename)
    except Exception as e:
        print(f"Error plotting benchmark metrics for {filename}: {str(e)}")

def plot_benchmark_correlations(dfs: Dict[str, pd.DataFrame],
                             title: str,
                             save_dir: str,
                             filename: str):
    """Plot mean and median times for add, search, and delete operations over iterations."""
    try:
        # Validate inputs
        if not dfs or len(dfs) == 0:
            print(f"Warning: No data provided for {filename}")
            return

        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))


        # Colors for different operations
        colors = {
            'bm_add.csv': 'blue',
            'bm_search.csv': 'green',
            'bm_delete.csv': 'red'
        }

        # Plot mean and median time for each operation
        for bm_file, df in dfs.items():
            if df is None or df.empty:
                continue

            operation = bm_file.replace('.csv', '').replace('bm_', '')
            base_color = colors.get(bm_file, 'gray')

            # Plot mean time
            if 'iteration' in df.columns and 'mean_time' in df.columns:
                ax.plot(df['iteration'],
                       df['mean_time'],
                       label=f'{operation.title()} Mean',
                       color=base_color,
                       linewidth=2)

            # Plot median time with dashed line
            if 'iteration' in df.columns and 'median_time' in df.columns:
                ax.plot(df['iteration'],
                       df['median_time'],
                       label=f'{operation.title()} Median',
                       color=base_color,
                       linestyle='--',
                       linewidth=2)

        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Time (seconds)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set axis limits without extra space
        # Use the maximum iteration across all dataframes
        max_iter = max(df['iteration'].max() for df in dfs.values() if df is not None and not df.empty)
        set_axis_limits(ax, pd.Series(range(max_iter + 1)))

        apply_bold_styling(ax)
        save_plot(fig, save_dir, filename)
    except Exception as e:
        print(f"Error plotting benchmark correlations for {filename}: {str(e)}")

def generate_benchmark_plots(experiment_paths: Dict[str, List[str]], config: str):
    """Generate all benchmark-related plots."""
    metrics = ['time']
    benchmark_files = ['bm_add.csv', 'bm_delete.csv', 'bm_search.csv']

    for scenario, paths in experiment_paths.items():
        for dataset_path in paths:

            # Extract dataset name from the folder path
            dataset_name = os.path.basename(dataset_path)
            algorithm_name = os.path.basename(dataset_path)
            # Handle special case for fashion-mnist
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

            # Step one back to get the scenario directory
            save_dir = os.path.join(dataset_path, "..", "..","..","..","figures", algorithm_name, scenario, dataset_name)
            if(algorithm_name == "RBC" or algorithm_name == "RFC"):
                save_dir = os.path.join(dataset_path,  "..", "..", "..","..","..","figures", algorithm_name, config, scenario, dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            # Load benchmark data
            benchmark_dfs = {}
            for bm_file in benchmark_files:
                filepath = os.path.join(dataset_path, bm_file)
                if os.path.exists(filepath):
                    try:
                        benchmark_dfs[bm_file] = load_csv_data(filepath)
                    except Exception as e:
                        print(f"Error loading {bm_file}: {str(e)}")
                        continue

            # Generate individual benchmark plots
            for bm_file, df in benchmark_dfs.items():
                for metric in metrics:
                    plot_benchmark_metrics(
                        df,
                        metric,
                        f'{bm_file.replace(".csv", "").split("_")[1].title()} {metric.title()} - {get_implementation_label(algorithm_name, config)} - {EXPERIMENT_NAME_MAP[scenario]} ({"Fashion-MNIST" if "fashion-mnist" in dataset_name else dataset_name.upper()})',
                        f'{metric.title()} (seconds)',
                        save_dir,
                        f'{scenario}_{bm_file.replace(".csv", "")}_{metric}.png'
                    )

            # # Generate comparison plot with all operations
            # if benchmark_dfs:
            #     plot_benchmark_correlations(
            #         benchmark_dfs,
            #         f'Operation Times Comparison - {algorithm_name} {[{CONFIG_INFO[config]}] if config != "-" else ""} - {scenario.title()} ({dataset_name})',
            #         save_dir,
            #         f'{scenario}_bm_times_comparison.png'
            #     )
