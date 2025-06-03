#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "hnswlib/helpers/hnswlib_index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"
#include "hnswlib/helpers/util.h"
#include <random>

using namespace duckdb;
using namespace hnswlib;

std::string experiment;

// ==================== Main Exclusive Unreachable Points MNγ-RU Runner ====================
class HNSWLibExclusiveUPRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
HNSWLibExclusiveUPRunner(int iterations, int threads) : db(nullptr), con(db), max_iterations(iterations), threads(threads) {
        con.Query("SET THREADS TO " + std::to_string(threads) + ";");
        datasets = DatabaseSetup::getDatasetConfigs();
    }

    void runTest(int datasetIdx) {
        try {
            // Cleanup intermediate files
            FileOperations::cleanupOutputFiles(std::filesystem::current_path());

            // Limit to valid dataset indices
            if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
                std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
                return;
            }

            const auto& dataset = datasets[datasetIdx];

            std::cout << "📊 TESTING DATASET: " << dataset.name << " 📊" << std::endl;

            // Setup database tables
            DatabaseSetup::initializeResultsTable(con, dataset.name);
            DatabaseSetup::initializeBMTable(con, dataset.name + "_del");
            DatabaseSetup::initializeBMTable(con, dataset.name + "_add");
            DatabaseSetup::initializeBMTable(con, dataset.name + "_search");
            DatabaseSetup::intializeEarlyTermTable(con);
            DatabaseSetup::setupFullDataset(con, dataset);

            // Load the hnswlib index
            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            L2Space space(dataset.dimensions);
            std::string index_path = "hnswlib/indexes/" + dataset.name + "_index.bin";
            HierarchicalNSW<float> index(&space, index_path, false, dataset_cardinality, true);

            // Load the index_map
            std::string index_map_path = "hnswlib/indexes/" + dataset.name + "_index_map.txt";
            std::unordered_map<size_t, size_t> index_map;
            index_map.reserve(dataset_cardinality);
            std::ifstream index_map_file(index_map_path);
            size_t key, value;
            while (index_map_file >> key >> value) {
                index_map[key] = value;
            }
            index_map_file.close();

            // Log initial index stats
            index.log_memory_stats();
            index.log_connectivity_stats(&space);
           
            // Get test vectors
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");
            auto test_vectors_count = test_vectors->RowCount();

            // Get first 5 test queries to run unreachable points search
            std::vector<std::vector<float>> queries;
            std::vector<int> query_indices;
            queries.reserve(5);
            query_indices.reserve(5);
            for (idx_t i = 0; i < 5; i++) {
                query_indices.push_back(test_vectors->GetValue(0, i).GetValue<int>());
                queries.push_back(ExtractFloatVector(test_vectors->GetValue(1, i)));
            }
            int num_queries = queries.size();

            // Dataset vectors
            auto dataset_vectors = con.Query("SELECT * FROM " + dataset.name + "_train;");

            std::unordered_set<size_t> available_points;
            available_points.reserve(dataset_cardinality);
            for (size_t i = 0; i < dataset_cardinality; i++) {
                available_points.insert(i);
            }

            // TODO: hardcoded value
            auto perc = 0.05;
            std::ostringstream perc_str;
            perc_str << std::fixed << std::setprecision(2) << perc;
            auto sample_size = (int) (perc * dataset_cardinality);
            std::random_device rd;
            std::mt19937 gen(rd());

            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            std::size_t executor_threads = (std::thread::hardware_concurrency());
            std::cout << "Threads: " << executor_threads << std::endl;


            // Initial query run (multi-threaded)
            index.setEf(100);
            HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality, index_map);

            // Get unreachable points
            std::vector<std::vector<float>> queries_tmp(queries.begin(),queries.begin()+num_queries);
            index.setEf(1000000);
            std::vector<std::vector<size_t>> results = {std::vector<size_t>(query_indices.begin(), query_indices.end())};
            util::query_hnsw_unreachable(index, queries_tmp, 1000000, executor_threads, results);
            for (size_t j = 0; j < queries_tmp.size(); ++j) {
                std::cout << "Query " << j << ":" << std::endl;
                std::cout << "Only found " << results[j].size() << " points" << std::endl;
            }
            index.setEf(100);
            std::string iteration_number = std::to_string(0);
            std::string unreachable_points_number = std::to_string(dataset_cardinality - results.front().size());
            std::vector<std::vector <std::string>> result_data = {{iteration_number, unreachable_points_number}};

            std::vector<std::pair<string, string>> unreachable_points;
            unreachable_points.reserve(3000);
            
            unreachable_points.push_back(std::make_pair(iteration_number, unreachable_points_number));

            std::cout << "Unreachable points: " << unreachable_points_number << " out of " << dataset_cardinality << std::endl;

            std::unordered_set<size_t> found_points;
            found_points.reserve(dataset_cardinality);
            for (const auto& idx : results[0]) {
                if (idx < 1000000) {
                    found_points.insert(idx);
                } else {
                    found_points.insert(idx - 1000000);
                }
            }

            for (auto it = available_points.begin(); it != available_points.end();) {
                if (found_points.find(*it) == found_points.end()) {
                    it = available_points.erase(it);
                } else {
                    ++it;
                }
            }

            // Run iterations
            for (int iteration = 1; iteration <= max_iterations; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;

                // Get sample vectors to delete and re-add
                std::vector<size_t> available_points_vec(available_points.begin(), available_points.end());
                std::shuffle(available_points_vec.begin(), available_points_vec.end(), gen);

                std::cout << "Available points size: " << available_points.size() << std::endl;

                int num_to_delete = std::min(static_cast<int>(dataset_cardinality * 0.05), static_cast<int>(available_points.size()));
                std::vector<size_t> delete_indices(available_points_vec.begin(), available_points_vec.begin() + num_to_delete);

                // Save the vectors and their labels to be deleted before deleting them
                std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dataset.dimensions));
                for (size_t i = 0; i < delete_indices.size(); ++i) {
                    size_t idx = delete_indices[i];
                    deleted_vectors[i] = ExtractFloatVector(dataset_vectors->GetValue(1, idx));
                }

                // Delete sample vectors (multi-threaded)
                size_t removed = HNSWLibIndexOperations::singleRemove(index, delete_indices, index_map, dataset.name, 
                    iteration, del_bm_appender);

                // Re-add the deleted vectors with their original labels
                std::vector<size_t> new_indices(delete_indices.size());
                for (size_t i = 0; i < delete_indices.size(); ++i) {
                    size_t idx = index_map[delete_indices[i]];
                    size_t new_idx = (idx < dataset_cardinality) ? idx + dataset_cardinality : idx - dataset_cardinality;
                    new_indices[i] = new_idx;
                    index_map[delete_indices[i]] = new_idx;
                }
                
                // Re-add vectors from this partition to the index
                size_t added = HNSWLibIndexOperations::parallelAddMNRU(index, deleted_vectors, new_indices,  dataset.name, 
                    iteration, add_bm_appender, executor_threads);                       

                // Log index stats
                index.log_memory_stats();
                index.log_connectivity_stats(&space);

                // Run test queries (multi-threaded)
                HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, 
                                        search_bm_appender, early_termination_appender, 
                                        iteration, dataset_cardinality, index_map);
            
                // Get unreachable points
                index.setEf(1000000);
                std::vector<std::vector<size_t>> results = {std::vector<size_t>(query_indices.begin(), query_indices.end())};
                util::query_hnsw_unreachable(index, queries_tmp, 1000000, executor_threads, results);
                for (size_t j = 0; j < queries_tmp.size(); ++j) {
                    std::cout << "Query " << j << ":" << std::endl;
                    std::cout << "Only found " << results[j].size() << " points" << std::endl;
                }
                index.setEf(100);
                iteration_number = std::to_string(iteration);
                unreachable_points_number = std::to_string(dataset_cardinality - results.front().size());
                result_data = {{iteration_number, unreachable_points_number}};

                unreachable_points.push_back(std::make_pair(iteration_number, unreachable_points_number));

                std::cout << "Unreachable points: " << unreachable_points_number << " out of " << dataset_cardinality << std::endl;

                found_points.clear();
                for (const auto& idx : results[0]) {
                    if (idx < 1000000) {
                        found_points.insert(idx);
                    } else {
                        found_points.insert(idx - 1000000);
                    }
                }

                for (auto it = available_points.begin(); it != available_points.end();) {
                    if (found_points.find(*it) == found_points.end()) {
                        it = available_points.erase(it);
                    } else {
                        ++it;
                    }
                }

                std::cout << "✅ FINISHED ITERATION " << iteration << " ✅" << std::endl;
            }

            appender.Close();
            early_termination_appender.Close();
            del_bm_appender.Close();
            add_bm_appender.Close();
            search_bm_appender.Close();

            // Calculate recall and aggregate stats
            QueryRunner::calculateRecall(con, dataset.name);
            QueryRunner::aggregateRecallStats(con, dataset.name);

            // Aggregate bm stats
            QueryRunner::aggregateBMStats(con, dataset.name + "_del", test_vectors_count, sample_size);
            QueryRunner::aggregateBMStats(con, dataset.name + "_add", test_vectors_count, sample_size);
            QueryRunner::aggregateBMStats(con, dataset.name + "_search", test_vectors_count, sample_size);

            // Output experiment results to CSV
            // dir name: MN_RU/results/{experiment}/{dataset_name}_{num_queries}q_{num_iterations}i_{sample_fraction}s/
            std::string output_dir = "MN_RU/results/unreachable_points_exclusive/" +  experiment + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + std::to_string(sample_size) + "r/";
            // Create the directory if it doesn't exist
            std::filesystem::create_directories(output_dir);
            FileOperations::cleanupOutputFiles(output_dir);
            QueryRunner::outputTableAsCSV(con, "recall_stats", output_dir + "search_query_stats.csv");
            QueryRunner::outputTableAsCSV(con, "early_terminated_queries", output_dir + "early_terminated_queries.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_del_bm_stats", output_dir + "bm_delete.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_add_bm_stats", output_dir + "bm_add.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_search_bm_stats", output_dir + "bm_search.csv");
            // Output unreachable points as csv
            std::ofstream unreachable_points_file(output_dir + "unreachable_points.csv");
            unreachable_points_file << "iteration,unreachable_points" << std::endl;
            for (const auto& point : unreachable_points) {
                unreachable_points_file << point.first << "," << point.second << std::endl;
            }
            unreachable_points_file.close();

            // Move lib output files to output dir
            FileOperations::copyFileTo("node_connectivity.csv", output_dir + "node_connectivity.csv");
            FileOperations::copyFileTo("memory_stats.csv", output_dir + "memory_stats.csv");

            // Cleanup intermediate files
            FileOperations::cleanupOutputFiles(std::filesystem::current_path());

        } catch (std::exception& e) {
            std::cerr << "Error running test: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {

    /**
     * UNREACHABLEPOINTS:
     * 3000 iterations. Within each iteration, 5% of vectors that are NOT unreachable in
     * the index are deleted and then reinserted to track the growth of unreachable
     * points over iterations. Same experiment as Enhancing HNSW paper. 
     * k = dataset_cardinality. Should prove that points already in the unreachable 
     * points set will not be searched in future search processes.
     */
    
    int max_iterations = 3000;
    std::size_t executor_threads = (std::thread::hardware_concurrency());

    experiment = "MN_RU_";

    try {

        // sift
        HNSWLibExclusiveUPRunner s_runner(max_iterations, executor_threads);
        s_runner.runTest(2);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
