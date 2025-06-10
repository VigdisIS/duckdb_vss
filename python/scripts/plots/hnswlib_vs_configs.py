"""Module for comparing multiple implementations with configurable labels."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.plots.plot_utils import load_csv_data, setup_plot_style, save_plot, apply_bold_styling

# Configuration descriptions
CONFIG_INFO = {
    "00": "No neighbor update, No tombstones",
    "01": "No neighbor update, With tombstones",
    "10": "With neighbor update, No tombstones",
    "11": "With neighbor update, With tombstones",
    "-": ""
}

# Colors for each implementation (expanded palette)
COLORS = {
    "hnswlib": "#1f77b4",      # Blue
    "repl_cand": "#ff7f0e",    # Orange
    "reset_first_candidate": "#2ca02c",  # Green
    "mn_ru": "#d62728",        # Red
    "mn_rbc": "#9467bd",        # Purple
    "usearch": "#8c564b"        # Brown
}

# Markers for each implementation
MARKERS = {
    "hnswlib": "o",
    "repl_cand": "s",
    "reset_first_candidate": "^",
    "mn_ru": "D",
    "mn_rbc": "v",
    "usearch": "x"
}

DIRECTORY_MAP = {
    "hnswlib": "hnswlib",
    "RBC": "repl_cand_hnswlib",
    "RFC": "reset_first_candidate",
    "MN-RU": "mn_ru",
    "MN-RBC": "mn_rbc",
    "usearch": "usearch"
}

EXPERIMENT_NAME_MAP = {
    "fullcoverage": "Full Coverage",
    "newdata": "New Data",
    "random": "Random",
    "unreachable_points_exclusive": "Unreachable Points"
}

def get_implementation_label(impl_code):
    """Get the display label for an implementation based on code and config."""
    if impl_code == "RBC":
        return r"OBS-RU-B$_2$"
    elif impl_code == "RFC":
        return r"OBS-RU-L$_2$"
    elif impl_code == "hnswlib":
        return "HNSW-RU"
    elif impl_code == "MN-RU":
        return "MN-RU"
    elif impl_code == "MN-RBC":
        return r"O$\alpha$G-RU"
    elif impl_code == "usearch":
        return "D-RU"
    else:
        # Throw error
        raise ValueError(f"Invalid implementation code: {impl_code}")

def generate_multi_implementation_comparison(base_dir, output_dir, implementations):
    """Generate comparison plots between multiple implementations.

    Args:
        base_dir: Base directory containing embedded-c++
        output_dir: Directory to save comparison plots
        implementations: List of implementation codes (e.g., ["hnswlib", "RBC", "RFC"])
    """

    # Define experiment types to compare
    experiments = ["fullcoverage", "newdata", "random", "unreachable_points_exclusive"]

    # Output directory for multi-implementation comparisons
    comparison_dir = os.path.join(output_dir, f"{'_vs_'.join(implementations)}")
    os.makedirs(comparison_dir, exist_ok=True)

    # For each experiment type
    for experiment in experiments:
        # Create experiment subdirectory
        experiment_dir = os.path.join(comparison_dir, experiment)
        os.makedirs(experiment_dir, exist_ok=True)

        # Get all dataset folders from all implementations for this experiment
        all_folders = []

        for impl in implementations:
            if impl == "RBC" or impl == "RFC":
                impl_exp_dir = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment)
            else:
                impl_exp_dir = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment)

            if os.path.exists(impl_exp_dir):
                for folder in os.listdir(impl_exp_dir):
                    folder_path = os.path.join(impl_exp_dir, folder)
                    if os.path.isdir(folder_path) and folder != 'images' and folder != 'thesis_output':
                        all_folders.append((impl, folder))

        # Function to extract core dataset name
        def extract_dataset_base(folder_name):
            # Remove prefixes
            cleaned_name = folder_name
            prefixes = ['hnswlib_', 'repl_cand_', 'hnswlib_repl_cand_', 'repl_cand_hnswlib_',
                        'reset_first_candidate_', 'MN_RU_', 'MN_RBC_', 'usearch_']
            for prefix in prefixes:
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

        # For each dataset, create comparison plots
        for dataset, impl_folders in dataset_map.items():
            # Only process datasets that exist in at least 2 implementations
            if len(impl_folders) >= 2:
                dataset_dir = os.path.join(experiment_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                print(f"Creating multi-implementation plots for experiment {experiment}, dataset {dataset} with implementations {impl_folders}")

                # Generate all comparison plots for this dataset
                plot_multi_implementation_metrics(
                    base_dir,
                    implementations,
                    experiment,
                    dataset,
                    impl_folders,
                    dataset_dir
                )

def plot_multi_implementation_metrics(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Plot all comparison metrics between multiple implementations."""
    # Plot recall comparison
    plot_multi_implementation_recall(base_dir, implementations, experiment, dataset, impl_folders, output_dir)

    # Plot unreachable points comparison
    plot_multi_implementation_unreachable(base_dir, implementations, experiment, dataset, impl_folders, output_dir)

    # Plot avg node connectivity comparison
    plot_multi_implementation_connectivity(base_dir, implementations, experiment, dataset, impl_folders, output_dir)

    # Plot benchmark comparisons
    plot_multi_implementation_add_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir)
    plot_multi_implementation_search_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir)
    plot_multi_implementation_delete_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir)

def plot_multi_implementation_recall(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing recall between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False

    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Path to the search_query_stats.csv file
        search_stats_path = os.path.join(results_path, 'search_query_stats.csv')

        if not os.path.exists(search_stats_path):
            print(f"Warning: File {search_stats_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(search_stats_path)

        if 'mean_recall' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers (map back to original keys)
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            # Plot with different colors and markers
            ax.plot(df['iteration'], df['mean_recall'],
                   label=label,
                   color=COLORS.get(impl_key, "#000000"),
                   marker=MARKERS.get(impl_key, "o"),
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Mean Recall', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Recall Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Set y-axis range for recall to 0-1
        ax.set_ylim(0, 1)

        # Save plot
        apply_bold_styling(ax)
        save_plot(fig, output_dir, f"recall_comparison")
    else:
        print(f"No recall data to plot for {dataset} in {experiment}")

    plt.close()

def plot_multi_implementation_unreachable(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing unreachable points between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False

    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Try both node_connectivity.csv and unreachable_points.csv
        connectivity_path = os.path.join(results_path, 'node_connectivity.csv')
        unreachable_path = os.path.join(results_path, 'unreachable_points.csv')

        df = None
        # First try connectivity file
        if os.path.exists(connectivity_path):
            df = load_csv_data(connectivity_path)
            if 'unreachable_count' not in df.columns:
                df = None

        # If not found, try dedicated unreachable_points file
        if df is None and os.path.exists(unreachable_path):
            df = load_csv_data(unreachable_path)
            if 'unreachable_points' in df.columns:
                df = df.rename(columns={'unreachable_points': 'unreachable_count'})
            else:
                df = None

        if df is not None and 'unreachable_count' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            ax.plot(df['iteration'], df['unreachable_count'],
                   label=label,
                   color=COLORS.get(impl_key, "#000000"),
                   marker=MARKERS.get(impl_key, "o"),
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Unreachable Points', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Unreachable Points Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Save plot
        apply_bold_styling(ax)
        save_plot(fig, output_dir, f"unreachable_points_comparison")
    else:
        print(f"No unreachable points data to plot for {dataset} in {experiment}")

    plt.close()

def plot_multi_implementation_connectivity(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing average node connectivity between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False

    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Path to the node_connectivity.csv file
        connectivity_path = os.path.join(results_path, 'node_connectivity.csv')

        if not os.path.exists(connectivity_path):
            print(f"Warning: File {connectivity_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(connectivity_path)

        if 'avg_connections' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            ax.plot(df['iteration'], df['avg_connections'],
                   label=label,
                   color=COLORS.get(impl_key, "#000000"),
                   marker=MARKERS.get(impl_key, "o"),
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Average Node Connectivity', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Node Connectivity Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Save plot
        apply_bold_styling(ax)
        save_plot(fig, output_dir, f"avg_connectivity_comparison")
    else:
        print(f"No connectivity data to plot for {dataset} in {experiment}")

    plt.close()

def plot_multi_implementation_add_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing add operation benchmark between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Path to the bm_add.csv file
        benchmark_path = os.path.join(results_path, 'bm_add.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            impl_data[impl] = {
                'df': df,
                'label': label,
                'color': COLORS.get(impl_key, "#000000"),
                'marker': MARKERS.get(impl_key, "o")
            }

            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        label = data['label']
        color = data['color']
        marker = data['marker']

        ax.plot(df['iteration'], df['mean_time'],
               label=label,
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Time (seconds)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Add Operation Time Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        apply_bold_styling(ax)
        save_plot(fig, output_dir, f"add_benchmark_comparison")
    else:
        print(f"No add benchmark data to plot for {dataset} in {experiment}")

    plt.close()

def plot_multi_implementation_search_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing search operation benchmark between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Path to the bm_search.csv file
        benchmark_path = os.path.join(results_path, 'bm_search.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            impl_data[impl] = {
                'df': df,
                'label': label,
                'color': COLORS.get(impl_key, "#000000"),
                'marker': MARKERS.get(impl_key, "o")
            }

            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        label = data['label']
        color = data['color']
        marker = data['marker']

        ax.plot(df['iteration'], df['mean_time'],
               label=label,
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Time (seconds)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Search Operation Time Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        apply_bold_styling(ax)
        save_plot(fig, output_dir, f"search_benchmark_comparison")
    else:
        print(f"No search benchmark data to plot for {dataset} in {experiment}")

    plt.close()

def plot_multi_implementation_delete_benchmark(base_dir, implementations, experiment, dataset, impl_folders, output_dir):
    """Create plot comparing delete operation benchmark between multiple implementations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for impl in implementations:
        if impl not in impl_folders:
            continue

        # Get correct path to results directory
        if impl == "RBC" or impl == "RFC":
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", "01", experiment, impl_folders[impl])
        else:
            results_path = os.path.join(base_dir, DIRECTORY_MAP[impl], "results", experiment, impl_folders[impl])

        # Path to the bm_delete.csv file
        benchmark_path = os.path.join(results_path, 'bm_delete.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            label = get_implementation_label(impl)

            # Get implementation key for colors/markers
            impl_key = impl
            if impl == "RBC":
                impl_key = "repl_cand"
            elif impl == "RFC":
                impl_key = "reset_first_candidate"
            elif impl in ["MN-RU", "MN-RBC"]:
                impl_key = impl.lower().replace('-', '_')

            impl_data[impl] = {
                'df': df,
                'label': label,
                'color': COLORS.get(impl_key, "#000000"),
                'marker': MARKERS.get(impl_key, "o")
            }

            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        label = data['label']
        color = data['color']
        marker = data['marker']

        ax.plot(df['iteration'], df['mean_time'],
               label=label,
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_ylabel('Time (seconds)', fontsize=plt.rcParams['axes.labelsize'])
        ax.set_title(f'Delete Operation Time Comparison - {EXPERIMENT_NAME_MAP[experiment]} ({"fashion-MNIST" if "fashion-mnist" in dataset else dataset.upper()})')
        ax.tick_params(axis='both', which='major', labelsize=14, rotation=0)
        ax.grid(True, alpha=0.5, linestyle='-')
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        save_plot(fig, output_dir, f"delete_benchmark_comparison")
    else:
        print(f"No delete benchmark data to plot for {dataset} in {experiment}")

    plt.close()
