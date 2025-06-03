"""Utility functions for plotting HNSW index experiment results."""

import os
import matplotlib
# Force matplotlib to use the Agg backend
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colorbar import Colorbar

experiments = ['fullcoverage', 'newdata', 'random', 'unreachable_points_exclusive']

# experiments = ['unreachable_points_exclusive']

implementations_map = {
    'hnswlib': 'HNSWLib',
    'repl_cand_hnswlib': 'RBC',
    'usearch': 'USearch',
    'reset_first_candidate': 'RFC'
}

vs_title = "RBC vs HNSWLib"

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and handle missing iteration column."""
    df = pd.read_csv(filepath)
    if 'iteration' not in df.columns:
        df['iteration'] = range(len(df))
    return df

def calculate_error_bounds(df: pd.DataFrame,
                         mean_col: str,
                         std_col: str = None) -> Tuple[pd.Series, pd.Series]:
    """Calculate error bounds using standard deviation if available."""
    if std_col and std_col in df.columns:
        # Use standard deviation for error bounds
        lower_bound = df[mean_col] - df[std_col]
        upper_bound = df[mean_col] + df[std_col]
    else:
        raise ValueError("Standard deviation column not available.")
    return lower_bound, upper_bound

def plot_with_error_bounds(ax: plt.Axes,
                         x: pd.Series,
                         y: pd.Series,
                         lower_bound: pd.Series,
                         upper_bound: pd.Series,
                         label: str,
                         color: str = 'blue'):
    """Plot data with error bounds."""
    # Plot error bounds
    ax.fill_between(x, lower_bound, upper_bound,
                   alpha=0.2, color=color, label=f'{label} ±1 Std Dev')
    # Plot mean line
    ax.plot(x, y, label=label, color=color, linewidth=2)

def setup_plot_style():
    """Set up the plot style for publication-quality figures."""
    plt.style.use('seaborn-paper')

    # Use serif with fallbacks
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']

    # Font sizes - increased for better readability
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16      # Increased from 16 (for "Iteration", "Time (seconds)", etc.)
    plt.rcParams['xtick.labelsize'] = 14     # Increased from 14 (for x-axis tick numbers)
    plt.rcParams['ytick.labelsize'] = 14     # Increased from 14 (for y-axis tick numbers)
    plt.rcParams['legend.fontsize'] = 14

    # Figure size and DPI
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Line widths and styles
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.8  # Increased grid line width
    plt.rcParams['lines.linewidth'] = 2.0  # Increased line width
    plt.rcParams['lines.markersize'] = 6

    # Grid settings - made more visible
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5  # Increased grid visibility
    plt.rcParams['grid.linestyle'] = '-'  # Solid grid lines

    # Remove default margins
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.1
    plt.rcParams['figure.constrained_layout.use'] = True

    # Color cycle - colorblind friendly
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Gray
    ])

def set_axis_limits(ax: plt.Axes, x_data: pd.Series, y_padding: float = 0.0, force_x_zero: bool = True):
    """Set axis limits without extra space.

    Args:
        ax: The matplotlib axes to modify
        x_data: The x-axis data
        y_padding: Additional padding to add to the top of the y-axis (as a fraction of the y-range)
        force_x_zero: Whether to force x-axis to start at 0
    """
    # Set x-axis limits
    if force_x_zero:
        ax.set_xlim(0, x_data.max())
    else:
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min
        x_margin = x_range * 0.05  # 5% margin
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # Set y-axis to start at 0 with optional padding
    y_min, y_max = ax.get_ylim()
    if y_padding > 0:
        y_range = y_max - y_min
        y_max = y_max + (y_range * y_padding)
    ax.set_ylim(0, y_max)

    # Ensure 0 is shown as first tick and no rotation
    yticks = ax.get_yticks()
    if yticks[0] != 0:
        yticks = [0] + list(yticks)
        ax.set_yticks(yticks)

    # Ensure y-axis tick labels are not rotated
    ax.tick_params(axis='y', rotation=0)

def create_combined_plot(plot_files: List[str],
                     scenario: str,
                     plot_type: str,
                     save_dir: str):
    """Create a combined plot from individual plot files.

    Args:
        plot_files: List of paths to individual plot files
        scenario: Name of the scenario
        plot_type: Type/name of the plot (used in title and filename)
        save_dir: Directory to save the combined plot
    """
    if not plot_files:
        return

    # Calculate grid dimensions
    num_plots = len(plot_files)
    n_cols = min(2, num_plots)  # Use 2 columns unless only 1 plot
    n_rows = (num_plots + 1) // 2  # Round up for odd numbers

    # Create figure with gridspec for more control over spacing
    fig = plt.figure(figsize=(15 * n_cols/2, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.2)

    # Add overall title
    fig.suptitle(f'{scenario} - {plot_type}', fontsize=16, y=0.95)

    # Create subplot for each plot
    for idx, plot_file in enumerate(sorted(plot_files)):
        # Get dataset name from plot file path
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(plot_file)))
        algorithm_name = os.path.basename(os.path.dirname(os.path.dirname(plot_file)))
        if "fashion_mnist" in dataset_name:
                dataset_name = "fashion-mnist"
        else:
            if("cand" in dataset_name): 
                dataset_name = dataset_name.split('_')[3]
            else:
                dataset_name = dataset_name.split('_')[1]

        if "reset_first_candidate" in algorithm_name:
            algorithm_name = "RFC"
        elif "repl_cand" in algorithm_name:
            algorithm_name = "RBC"
        elif "hnswlib" in algorithm_name:
            algorithm_name = "HNSWLib"
        elif "usearch" in algorithm_name:
            algorithm_name = "USearch"

        # Create subplot using gridspec
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Load and display image
        img = plt.imread(plot_file)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(f'{algorithm_name} {dataset_name}', pad=10)  # Reduce padding between title and plot

    # Save combined plot
    combined_filename = f'{scenario}_combined_{plot_type}.png'
    plt.savefig(os.path.join(save_dir, combined_filename),
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)  # Reduce padding around the entire figure
    plt.close()

def combine_scenario_plots(experiment_paths: Dict[str, List[str]]):
    """Combine plots from all datasets in each scenario.

    Args:
        experiment_paths: Dictionary mapping scenario names to lists of dataset paths
    """
    for scenario, dataset_paths in experiment_paths.items():
        # Create scenario-level directory for combined plots if it doesn't exist
        scenario_dir = os.path.dirname(dataset_paths[0])
        # scenario_images_dir = os.path.join(scenario_dir, 'images')
        # os.makedirs(scenario_images_dir, exist_ok=True)

        # Get all unique plot types from the first dataset's images directory
        first_dataset_images = os.path.join(dataset_paths[0], 'images')
        if not os.path.exists(first_dataset_images):
            continue

        plot_types = set()
        for filename in os.listdir(first_dataset_images):
            if filename.endswith('.png'):
                # Extract the plot type from the filename
                # Assuming format: scenario_plottype.png or scenario_plottype_suffix.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) > 1:
                    plot_type = '_'.join(parts[1:])  # Join all parts after scenario
                    plot_types.add(plot_type)

        # For each plot type, collect corresponding plots from all datasets
        for plot_type in plot_types:
            plot_files = []
            for dataset_path in dataset_paths:
                images_dir = os.path.join(dataset_path, 'images')
                if not os.path.exists(images_dir):
                    continue

                # Look for matching plot file
                for filename in os.listdir(images_dir):
                    if filename.endswith('.png') and plot_type in filename:
                        plot_files.append(os.path.join(images_dir, filename))
                        break  # Take the first matching file

            if plot_files:
                create_combined_plot(plot_files, scenario, plot_type, scenario_images_dir)

def generate_comparison_plots(base_dir: str, output_dir: str, implementations: List[str]):
    """Generate comparison plots between hnswlib and repl_cand_hnswlib for key metrics.

    Args:
        base_dir: Base directory containing both implementations
        output_dir: Directory to save comparison plots
        implementations: List of implementations to compare
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # For each experiment type
    for experiment in experiments:
        # Create experiment subdirectory
        experiment_dir = os.path.join(output_dir, experiment)
        os.makedirs(experiment_dir, exist_ok=True)

        # Get all unique dataset suffixes
        dataset_suffixes = {}  # Maps normalized suffix to original suffix
        impl_dataset_map = {}  # Maps dataset suffixes to full dataset names per implementation

        # Get all dataset folders from both implementations
        all_folders = []

        for impl in implementations:
            impl_dir = os.path.join(base_dir, impl, 'results', experiment)
            if not os.path.exists(impl_dir):
                print(f"Warning: Directory {impl_dir} does not exist")
                continue

            impl_dataset_map[impl] = {}

            for folder in os.listdir(impl_dir):
                folder_path = os.path.join(impl_dir, folder)
                if os.path.isdir(folder_path) and not folder == 'images':
                    all_folders.append((impl, folder))

        # Function to extract core dataset name from any folder name
        def extract_dataset_suffix(folder_name):
            # Remove known prefixes
            cleaned_name = folder_name
            for prefix in ['hnswlib_', 'repl_cand_', 'hnswlib_repl_cand_', 'repl_cand_hnswlib_']:
                if cleaned_name.startswith(prefix):
                    cleaned_name = cleaned_name[len(prefix):]
            return cleaned_name

        # Function to normalize dataset suffix to a unique identifier
        def normalize_suffix(suffix):
            # Remove any remaining prefix-like parts
            for prefix in ['hnswlib_', 'repl_cand_', 'hnswlib_repl_cand_', 'repl_cand_hnswlib_']:
                if suffix.startswith(prefix):
                    suffix = suffix[len(prefix):]
            return suffix

        # Build mapping of dataset suffixes to full folder names
        for impl, folder in all_folders:
            suffix = extract_dataset_suffix(folder)
            normalized_suffix = normalize_suffix(suffix)

            # Store the normalized suffix
            dataset_suffixes[normalized_suffix] = suffix

            # Map implementation and suffix to folder
            if impl not in impl_dataset_map:
                impl_dataset_map[impl] = {}
            impl_dataset_map[impl][normalized_suffix] = folder

        # For each unique normalized dataset suffix
        for normalized_suffix in dataset_suffixes:
            dataset_dir = os.path.join(experiment_dir, normalized_suffix)
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"Creating plots for {normalized_suffix} in {experiment}")

            # Now we pass the normalized suffix to ensure consistent lookup
            plot_comparison_metrics(base_dir, implementations, experiment, normalized_suffix, impl_dataset_map, dataset_dir)

def plot_comparison_metrics(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Plot all comparison metrics for a given dataset suffix."""
    # Plot recall comparison
    plot_recall_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)

    # Plot unreachable points comparison
    plot_unreachable_points_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)

    # Plot avg node connectivity comparison
    plot_avg_connectivity_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)

    # Plot benchmark comparisons
    plot_add_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)
    plot_search_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)
    plot_delete_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir)

def plot_recall_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing recall between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False

    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Path to the search_query_stats.csv file
        search_stats_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'search_query_stats.csv')

        if not os.path.exists(search_stats_path):
            print(f"Warning: File {search_stats_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(search_stats_path)

        if 'mean_recall' in df.columns:
            has_data = True
            # Plot with different colors and markers for different implementations
            marker = 'o' if i == 0 else 's'
            color = '#1f77b4' if i == 0 else '#ff7f0e'  # Blue for impl 1, orange for impl 2
            ax.plot(df['iteration'], df['mean_recall'],
                   label=f"{impl}",
                   color=color,
                   marker=marker,
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Recall')
        ax.set_title(f'Recall Comparison {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set y-axis range for recall to 0-1
        ax.set_ylim(0, 1)

        # Save plot
        save_plot(fig, output_dir, f"recall_comparison")
    else:
        print(f"No recall data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def plot_unreachable_points_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing unreachable points between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False

    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Try both node_connectivity.csv and unreachable_points.csv
        connectivity_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'node_connectivity.csv')
        unreachable_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'unreachable_points.csv')

        # First try connectivity file
        if os.path.exists(connectivity_path):
            df = load_csv_data(connectivity_path)
            if 'unreachable_count' in df.columns:
                has_data = True
                # Plot with different colors and markers for different implementations
                marker = 'o' if i == 0 else 's'
                color = '#1f77b4' if i == 0 else '#ff7f0e'  # Blue for impl 1, orange for impl 2
                ax.plot(df['iteration'], df['unreachable_count'],
                       label=f"{impl}",
                       color=color,
                       marker=marker,
                       markersize=6,
                       markevery=max(1, len(df)//10),
                       linewidth=2)
                continue

        # If not found, try dedicated unreachable_points file
        if os.path.exists(unreachable_path):
            df = load_csv_data(unreachable_path)
            if 'unreachable_points' in df.columns:
                has_data = True
                # Plot with different colors and markers for different implementations
                marker = 'o' if i == 0 else 's'
                color = '#1f77b4' if i == 0 else '#ff7f0e'  # Blue for impl 1, orange for impl 2
                ax.plot(df['iteration'], df['unreachable_points'],
                       label=f"{impl}",
                       color=color,
                       marker=marker,
                       markersize=6,
                       markevery=max(1, len(df)//10),
                       linewidth=2)
                continue

        print(f"Warning: No unreachable points data found for {impl} in {dataset_suffix}")

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Unreachable Points')
        ax.set_title(f'Unreachable Points Comparison - {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Save plot
        save_plot(fig, output_dir, f"unreachable_points_comparison")
    else:
        print(f"No unreachable points data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def plot_avg_connectivity_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing average node connectivity between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False

    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Path to the node_connectivity.csv file
        connectivity_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'node_connectivity.csv')

        if not os.path.exists(connectivity_path):
            print(f"Warning: File {connectivity_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(connectivity_path)

        if 'avg_connections' in df.columns:
            has_data = True
            # Plot with different colors and markers for different implementations
            marker = 'o' if i == 0 else 's'
            color = '#1f77b4' if i == 0 else '#ff7f0e'  # Blue for impl 1, orange for impl 2
            ax.plot(df['iteration'], df['avg_connections'],
                   label=f"{impl}",
                   color=color,
                   marker=marker,
                   markersize=6,
                   markevery=max(1, len(df)//10),
                   linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Average Node Connectivity')
        ax.set_title(f'Node Connectivity Comparison - {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ensure y-axis starts at 0
        ax.set_ylim(bottom=0)

        # Save plot
        save_plot(fig, output_dir, f"avg_connectivity_comparison")
    else:
        print(f"No connectivity data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def plot_add_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing add operation benchmark between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across all implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Path to the bm_add.csv file
        benchmark_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'bm_add.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            impl_data[impl] = {
                'df': df,
                'color': '#1f77b4' if i == 0 else '#ff7f0e',  # Blue for impl 1, orange for impl 2
                'marker': 'o' if i == 0 else 's'
            }
            
            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        marker = data['marker']
        color = data['color']
        
        ax.plot(df['iteration'], df['mean_time'],
               label=f"{impl}",
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Add Operation Time Comparison - {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        save_plot(fig, output_dir, f"add_benchmark_comparison")
    else:
        print(f"No add benchmark data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def plot_search_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing search operation benchmark between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across all implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Path to the bm_search.csv file
        benchmark_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'bm_search.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            impl_data[impl] = {
                'df': df,
                'color': '#1f77b4' if i == 0 else '#ff7f0e',  # Blue for impl 1, orange for impl 2
                'marker': 'o' if i == 0 else 's'
            }
            
            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        marker = data['marker']
        color = data['color']
        
        ax.plot(df['iteration'], df['mean_time'],
               label=f"{impl}",
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Search Operation Time Comparison - {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        save_plot(fig, output_dir, f"search_benchmark_comparison")
    else:
        print(f"No search benchmark data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def plot_delete_benchmark_comparison(base_dir, implementations, experiment, dataset_suffix, impl_dataset_map, output_dir):
    """Create plot comparing delete operation benchmark between implementations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_plot_style()

    has_data = False
    min_iteration = float('inf')  # Track minimum iteration across all implementations

    # First pass to collect data and find minimum iteration
    impl_data = {}
    for i, impl in enumerate(implementations):
        # Skip if implementation doesn't have this dataset
        if impl not in impl_dataset_map or dataset_suffix not in impl_dataset_map[impl]:
            continue

        # Get the actual dataset folder name for this implementation
        dataset_folder = impl_dataset_map[impl][dataset_suffix]

        # Path to the bm_delete.csv file
        benchmark_path = os.path.join(base_dir, impl, 'results', experiment, dataset_folder, 'bm_delete.csv')

        if not os.path.exists(benchmark_path):
            print(f"Warning: File {benchmark_path} does not exist")
            continue

        # Load CSV data
        df = load_csv_data(benchmark_path)

        if 'mean_time' in df.columns:
            has_data = True
            impl_data[impl] = {
                'df': df,
                'color': '#1f77b4' if i == 0 else '#ff7f0e',  # Blue for impl 1, orange for impl 2
                'marker': 'o' if i == 0 else 's'
            }
            
            # Update minimum iteration if needed
            if not df.empty and df['iteration'].min() < min_iteration:
                min_iteration = df['iteration'].min()

    # Second pass to plot the data with proper x-axis limits
    for impl, data in impl_data.items():
        df = data['df']
        marker = data['marker']
        color = data['color']
        
        ax.plot(df['iteration'], df['mean_time'],
               label=f"{impl}",
               color=color,
               marker=marker,
               markersize=6,
               markevery=max(1, len(df)//10),
               linewidth=2)

    if has_data:
        # Set plot labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Delete Operation Time Comparison - {vs_title} - {experiment.title()} ({dataset_suffix})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Ensure y-axis starts at 0 but x-axis starts at min_iteration
        ax.set_ylim(bottom=0)
        if min_iteration != float('inf'):
            ax.set_xlim(left=min_iteration)

        # Save plot
        save_plot(fig, output_dir, f"delete_benchmark_comparison")
    else:
        print(f"No delete benchmark data to plot for {dataset_suffix} in {experiment}")

    plt.close()

def save_plot(fig: plt.Figure,
             save_dir: str,
             filename: str,
             experiment_paths: Optional[Dict[str, List[str]]] = None,
             dpi: int = 300) -> None:
    """Save a plot with publication-quality settings.

    Args:
        fig: The matplotlib figure to save
        save_dir: Directory to save the plot in
        filename: Name of the plot file (without extension)
        experiment_paths: Optional dictionary mapping scenarios to dataset paths
        dpi: Resolution in dots per inch
    """
    try:
        # Validate inputs
        if fig is None:
            print("Error: Figure is None")
            return

        if not save_dir or not filename:
            print("Error: save_dir and filename must be provided")
            return

        # Create main directory if it doesn't exist
        # os.makedirs(save_dir, exist_ok=True)

        # Create png subdirectory
        png_dir = os.path.join(save_dir, "png")
        os.makedirs(png_dir, exist_ok=True)

        # Remove any existing extensions from the filename
        base_name = os.path.splitext(filename)[0]
        base_path = os.path.join(save_dir, base_name)
        pdf_path = f"{base_path}.pdf"
        png_path = os.path.join(png_dir, base_name + ".png")

        # Check if figure has any content
        if not fig.get_axes():
            print(f"Warning: Figure has no axes, skipping save for {filename}")
            return

        # Adjust layout based on whether the figure has a colorbar
        has_colorbar = any(isinstance(c, Colorbar) for c in fig.get_children())
        if has_colorbar:
            # For plots with colorbars, use tight_layout with minimal padding
            fig.tight_layout(pad=0.5)
        else:
            # For regular plots, use constrained_layout with minimal padding
            fig.set_constrained_layout(True)
            fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.5)

        # Save as PNG with minimal padding
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

        # Save as PDF (vector format) with minimal padding
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.05)

        print(f"Plot saved successfully as PNG and PDF: {os.path.basename(png_path)}")

    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")
    finally:
        plt.close(fig)  # Close the figure to free memory

def get_experiment_paths(base_dir: str) -> Dict[str, List[str]]:
    """Get paths for all experiment data files."""

    paths = {}

    # If last character is -, remove it
    if base_dir.endswith('-'):
        base_dir = base_dir[:-1]

    print(base_dir)

    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        if not os.path.exists(exp_dir):
            continue

        paths[exp] = []
        for dataset_dir in os.listdir(exp_dir):
            dataset_path = os.path.join(exp_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                paths[exp].append(dataset_path)

    return paths
