"""Module for plotting search query statistics and early termination analysis."""
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.plots.plot_utils import (load_csv_data, calculate_error_bounds,
                        plot_with_error_bounds, setup_plot_style, save_plot, set_axis_limits)

def plot_search_metrics(df: pd.DataFrame,
                       metrics: List[str],
                       title: str,
                       save_dir: str,
                       filename: str):
    """Plot search metrics with error bounds."""
    fig, ax = plt.subplots()
    setup_plot_style()

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for metric, color in zip(metrics, colors):
        lower_bound, upper_bound = calculate_error_bounds(
            df,
            f'mean_{metric}',
            std_col=f'stddev_{metric}',
            min_col=f'min_{metric}',
            max_col=f'max_{metric}'
        )

        plot_with_error_bounds(
            ax,
            df['iteration'],
            df[f'mean_{metric}'],
            lower_bound,
            upper_bound,
            label=metric,
            color=color
        )

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Set axis limits without extra space
    set_axis_limits(ax, df['iteration'])

    save_plot(fig, save_dir, filename)

def plot_early_termination_analysis(
    df: pd.DataFrame,
    save_dir: str,
    filename: str
) -> None:
    """Create multiple plots for early termination analysis using raw query data."""
    setup_plot_style()

    # Check if DataFrame is empty
    if df.empty:
        print(f"Warning: No data available for {filename}")
        return

    # Check if required columns exist
    required_cols = ['iteration']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns for {filename}")
        return

    # Get max iteration for consistent x-axis limits
    max_iter = df['iteration'].max()
    min_iter = df['iteration'].min()

    # 1. Distribution of Early Termination Iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    # Count number of early terminated queries per iteration
    early_term_counts = df.groupby('iteration').size()

    if not early_term_counts.empty:
        # Calculate number of iterations for tick spacing
        tick_spacing = max(1, max_iter // 10)  # One tick per 10 iterations, or 1 if less than 10

        # Create bar plot using matplotlib with lighter blue fill and darker blue edge
        ax.bar(early_term_counts.index,
               early_term_counts.values,
               color='#6495ED',  # Cornflower blue for fill
               edgecolor='#4169E1',  # Royal blue for edge
               linewidth=0.5,  # Subtle border
               width=1.0)
        ax.set_title('Distribution of Early Termination Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Early Terminated Queries')

        # Set x-axis to start at 0 and end exactly at max iteration
        ax.set_xlim(0, max_iter)

        # Set x-axis ticks with appropriate spacing
        ax.set_xticks(range(0, max_iter + 1, tick_spacing))

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        save_plot(fig, save_dir, f"{filename}_distribution")
    plt.close()

    # 2. Recall by Early Termination Iteration
    if 'recall' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Calculate mean recall per iteration
        recall_by_iter = df.groupby('iteration')['recall'].mean()
        if not recall_by_iter.empty:
            ax.plot(recall_by_iter.index, recall_by_iter.values, linewidth=2)
            ax.set_title('Recall by Iteration')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Recall')

            # Let matplotlib automatically determine good x-axis limits
            ax.margins(x=0.02)  # Add just a tiny bit of padding (2%)

            # Set x-axis ticks with appropriate spacing
            ax.set_xticks(range(min_iter, max_iter + 1, tick_spacing))

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            save_plot(fig, save_dir, f"{filename}_recall_by_iter")
        plt.close()

    # 3. Correlation Analysis
    # Select relevant columns for correlation
    corr_cols = ['recall', 'computed_distances', 'visited_members']
    if all(col in df.columns for col in corr_cols):
        corr_matrix = df[corr_cols].corr()
        if not corr_matrix.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Analysis of Early Termination Factors')
            save_plot(fig, save_dir, f"{filename}_correlations")
            plt.close()

    # 4. Scatter plots for key relationships
    scatter_cols = ['computed_distances', 'visited_members', 'recall']
    if any(col in df.columns for col in scatter_cols):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Early Termination Analysis: Key Relationships')

        # Computed distances vs Iteration
        if 'computed_distances' in df.columns:
            sns.scatterplot(data=df, x='iteration', y='computed_distances', ax=axes[0,0])
            axes[0,0].margins(x=0.02)  # Add just a tiny bit of padding (2%)
            axes[0,0].set_xticks(range(min(df['iteration']), max_iter + 1, tick_spacing))
        axes[0,0].set_title('Computed Distances vs Iteration')

        # Visited members vs Iteration
        if 'visited_members' in df.columns:
            sns.scatterplot(data=df, x='iteration', y='visited_members', ax=axes[0,1])
            axes[0,1].margins(x=0.02)  # Add just a tiny bit of padding (2%)
            axes[0,1].set_xticks(range(min(df['iteration']), max_iter + 1, tick_spacing))
        axes[0,1].set_title('Visited Members vs Iteration')

        # Recall vs Computed Distances
        if 'recall' in df.columns and 'computed_distances' in df.columns:
            sns.scatterplot(data=df, x='computed_distances', y='recall', ax=axes[1,0])
        axes[1,0].set_title('Recall vs Computed Distances')

        # Recall vs Visited Members
        if 'recall' in df.columns and 'visited_members' in df.columns:
            sns.scatterplot(data=df, x='visited_members', y='recall', ax=axes[1,1])
        axes[1,1].set_title('Recall vs Visited Members')

        # Rotate x-axis labels for all subplots
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        save_plot(fig, save_dir, f"{filename}_relationships")

def plot_visited_vs_computed(df: pd.DataFrame,
                      title: str,
                      save_dir: str,
                      filename: str):
    """Plot the relationship between visited members and computed distances."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient
    scatter = ax.scatter(df['mean_visited_members'],
                        df['mean_computed_distances'],
                        c=df['mean_recall'],
                        cmap='viridis',
                        alpha=0.6)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Recall', rotation=270, labelpad=15)

    ax.set_xlabel('Mean Number of Visited Members')
    ax.set_ylabel('Mean Number of Computed Distances')
    ax.set_title('Search Efficiency: Visited Members vs Computed Distances')
    ax.grid(True)

    # Adjust layout
    fig.set_constrained_layout(True)
    save_plot(fig, save_dir, f"{filename}_visited_vs_computed")
    plt.close()

def plot_efficiency_distribution(df: pd.DataFrame,
                      title: str,
                      save_dir: str,
                      filename: str):
    """Plot the distribution of search efficiency metrics."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Box plot showing distribution of efficiency metrics
    efficiency_data = pd.DataFrame({
        'Metric': ['Visited Members'] * len(df) + ['Computed Distances'] * len(df),
        'Value': pd.concat([df['mean_visited_members'], df['mean_computed_distances']])
    })

    sns.boxplot(data=efficiency_data, x='Metric', y='Value', ax=ax)
    ax.set_title('Distribution of Search Efficiency Metrics')
    ax.set_ylabel('Count')
    ax.grid(True)

    # Adjust layout
    fig.set_constrained_layout(True)
    save_plot(fig, save_dir, f"{filename}_efficiency_distribution")
    plt.close()

def generate_search_analysis_plots(experiment_paths: Dict[str, List[str]]):
    """Generate all search-related plots."""
    search_metrics = ['recall', 'computed_distances', 'visited_members', 'results_count']

    for scenario, paths in experiment_paths.items():
        for dataset_path in paths:
            save_dir = os.path.join(dataset_path, 'images')

            # Search query stats plots (aggregated data)
            search_stats_path = os.path.join(dataset_path, 'search_query_stats.csv')
            if os.path.exists(search_stats_path):
                search_df = load_csv_data(search_stats_path)

                plot_search_metrics(
                    search_df,
                    search_metrics,
                    f'{scenario} - Search Metrics Over Time',
                    save_dir,
                    f'{scenario}_search_metrics'
                )

                plot_visited_vs_computed(
                    search_df,
                    f'{scenario} - Search Efficiency',
                    save_dir,
                    f'{scenario}_search_efficiency'
                )

                plot_efficiency_distribution(
                    search_df,
                    f'{scenario} - Search Efficiency Distribution',
                    save_dir,
                    f'{scenario}_search_efficiency'
                )

            # Early termination analysis (raw data)
            early_termination_path = os.path.join(dataset_path, 'early_terminated_queries.csv')
            if os.path.exists(early_termination_path):
                early_termination_df = load_csv_data(early_termination_path)
                plot_early_termination_analysis(
                    early_termination_df,
                    save_dir,
                    f'{scenario}_early_termination'
                )
