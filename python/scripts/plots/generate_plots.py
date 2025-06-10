"""Main script for generating all plots from HNSW index experiments."""

import os
from scripts.plots.plot_utils import get_experiment_paths, combine_scenario_plots
from scripts.plots.benchmark_plots import generate_benchmark_plots
from scripts.plots.memory_connectivity_plots import generate_memory_connectivity_plots
from scripts.plots.search_analysis_plots import generate_search_analysis_plots
from scripts.plots.config_comparison_plots import generate_config_comparison_plots
from scripts.plots.hnswlib_vs_configs import generate_multi_implementation_comparison

def main():
    """Generate all plots for the HNSW index experiments."""
    # Get the base directory for experiment results
    # Go up 2 levels from the script location to reach the python directory
    python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Base directory containing both implementations
    base_dir = os.path.join(os.path.dirname(python_dir), "embedded-c++")

    # Path to repl_cand_hnswlib results
    repl_cand_base_dir = os.path.join(base_dir, "repl_cand_hnswlib", "results")
    # Path to reset_first_candidate results
    reset_first_candidate_base_dir = os.path.join(base_dir, "reset_first_candidate", "results")
    # Path to hnswlib results
    hnswlib_base_dir = os.path.join(base_dir, "hnswlib", "results")
    # Path to usearch results
    usearch_base_dir = os.path.join(base_dir, "usearch", "results")
    # Path to mn_ru results
    mn_ru_base_dir = os.path.join(base_dir, "mn_ru", "results")
    # Path to mn_rbc results
    mn_rbc_base_dir = os.path.join(base_dir, "mn_rbc", "results")

    all_plots = ["mn_ru", "mn_rbc", "usearch", "hnswlib", "repl_cand_hnswlib", "reset_first_candidate"]

    for plot in all_plots:
        generate_plots_for = plot

        configs = []
        if generate_plots_for == "repl_cand_hnswlib":
            configs = ["00", "01", "10", "11"]
        elif generate_plots_for == "reset_first_candidate":
            configs = ["01"]
        elif generate_plots_for == "hnswlib":
            configs = ["-"]
        elif generate_plots_for == "usearch":
            configs = ["-"]
        elif generate_plots_for == "mn_ru":
            configs = ["-"]
        elif generate_plots_for == "mn_rbc":
            configs = ["-"]

        # Save directory for comparison plots
        comparison_dir = os.path.join(base_dir, 'comparison_plots')
        os.makedirs(comparison_dir, exist_ok=True)

        # Get paths to experiment results
        experiment_paths = {}
        for config in configs:
            if generate_plots_for == "repl_cand_hnswlib":
                experiment_paths[config] = get_experiment_paths(os.path.join(repl_cand_base_dir, config))
            elif generate_plots_for == "reset_first_candidate":
                experiment_paths[config] = get_experiment_paths(os.path.join(reset_first_candidate_base_dir, config))
            elif generate_plots_for == "hnswlib":
                experiment_paths[config] = get_experiment_paths(os.path.join(hnswlib_base_dir, config))
            elif generate_plots_for == "usearch":
                experiment_paths[config] = get_experiment_paths(os.path.join(usearch_base_dir, config))
            elif generate_plots_for == "mn_ru":
                experiment_paths[config] = get_experiment_paths(os.path.join(mn_ru_base_dir, config))
            elif generate_plots_for == "mn_rbc":
                experiment_paths[config] = get_experiment_paths(os.path.join(mn_rbc_base_dir, config))

        for config in configs:
            print(f"Generating plots for config {config}...")

            print("Generating benchmark plots...")
            generate_benchmark_plots(experiment_paths[config], config)

            print("Generating memory and connectivity plots...")
            generate_memory_connectivity_plots(experiment_paths[config], config)

            print("Generating search analysis plots...")
            generate_search_analysis_plots(experiment_paths[config], config)


        # Generate comparison plots between configurations (00 vs 01 vs 10 vs 11)
        if generate_plots_for == "repl_cand_hnswlib":
            print("Generating comparison plots between configurations...")
            generate_config_comparison_plots(base_dir, comparison_dir)

        # Generate comparison plots between implementations

        # generate_multi_implementation_comparison(base_dir, comparison_dir, ["usearch", "hnswlib"])
        generate_multi_implementation_comparison(base_dir, comparison_dir, ["hnswlib", "RFC"])
        generate_multi_implementation_comparison(base_dir, comparison_dir, ["hnswlib", "RBC"])
        # generate_multi_implementation_comparison(base_dir, comparison_dir, ["hnswlib", "RFC", "RBC"])
        generate_multi_implementation_comparison(base_dir, comparison_dir, ["RFC", "RBC"])
        generate_multi_implementation_comparison(base_dir, comparison_dir, ["MN-RU", "MN-RBC"])
        # generate_multi_implementation_comparison(base_dir, comparison_dir, ["hnswlib", "MN-RU", "MN-RBC"])
        generate_multi_implementation_comparison(base_dir, comparison_dir, ["hnswlib", "RFC", "RBC", "MN-RU", "MN-RBC"])


        print("All plots generated successfully!")

if __name__ == "__main__":
    main()
