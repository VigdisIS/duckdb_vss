#include "duckdb.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace unum::usearch;
using namespace duckdb;

// ==================== Dataset Configuration ====================
struct DatasetConfig {
    std::string name;
    int dimensions;
};

std::vector<DatasetConfig> getDatasetConfigs() {
    return {
        {"fashion_mnist", 784},
        {"mnist", 784},
        {"sift", 128},
        {"gist", 960}
    };
}

// ==================== HNSW Index Operations ====================
class HNSWIndex {
private:
    metric_punned_t metric;
    index_dense_t index;
    int dimensions;

public:
    HNSWIndex(int dims, int max_elements = 60000, int ef_construction = 200, int m = 16)
        : metric(3, metric_kind_t::l2sq_k, scalar_kind_t::f32_k), dimensions(dims) {
        std::cout << "🆕 Created new in-memory HNSW index" << std::endl;
    }
    
    void initializeIndex(Connection& con, const std::string& table_name) {

        index = index_dense_t::make(metric);
        auto train_vectors = con.Query("SELECT id, vec FROM " + table_name + "_train;");
        std::cout << "🔄 Indexing " << train_vectors->RowCount() << " vectors..." << std::endl;

        index.reserve(train_vectors->RowCount());

        for (idx_t i = 0; i < train_vectors->RowCount(); i++) {
          
            int id = train_vectors->GetValue(0, i).GetValue<int>();
            std::vector<float> vec(dimensions);
            train_vectors->GetValue(1, i);
            addPoint(id, vec);

            // Display progress bar
            if (i % (train_vectors->RowCount() / 100) == 0 || i == train_vectors->RowCount() - 1) {
                int progress = (i * 100) / train_vectors->RowCount();
                std::cout << "\rIndexing Progress: [" << std::string(progress / 2, '=') 
                    << std::string(50 - progress / 2, ' ') << "] " 
                    << progress << "%";
                std::cout.flush();
            }
        }
        std::cout << std::endl; // Move to the next line after progress bar
        std::cout << "✅ Finished indexing vectors" << std::endl;
    }

    void addPoint(int id, const std::vector<float>& vec) {
            index.add(id, vec.data());
        }

    std::vector<int> search(const std::vector<float>& vec, int k = 100) {
        auto results = index.search(vec.data(), k);
        std::vector<int> neighbors;

        for (std::size_t i = 0; i < results.size(); i++) {
            neighbors.push_back(results[i].member.key);
        }
        return neighbors;
    }
};

// ==================== Database Setup Functions ====================
class DatabaseSetup {
public:
    static void setupFullDataset(Connection& con, const DatasetConfig& config) {
        con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

        con.Query("CREATE OR REPLACE TABLE memory." + config.name + "_train AS SELECT * FROM raw." + config.name + "_train;");
        con.Query("CREATE OR REPLACE TABLE memory." + config.name + "_test AS SELECT * FROM raw." + config.name + "_test LIMIT 100;");

        con.Query("DETACH raw;");
    }

    static void initializeResultsTable(Connection& con, const std::string& table_name) {
        std::string recall_stats_query = "CREATE OR REPLACE TABLE memory.recall_stats (" +
                  std::string("dataset VARCHAR, iteration INT, num_queries INT, ") +
                  std::string("mean_recall FLOAT, median_recall FLOAT, stddev_recall FLOAT, ") +
                  std::string("var_recall FLOAT, min_recall FLOAT, max_recall FLOAT, ") +
                  std::string("p25_recall FLOAT, p75_recall FLOAT, p95_recall FLOAT);");
        con.Query(recall_stats_query);
        
        std::string results_query = "CREATE OR REPLACE TABLE " + table_name + "_results (" +
                  std::string("dataset VARCHAR, iteration INT, test_vec_id INT, ") +
                  std::string("neighbor_vec_ids INTEGER[100], result_vec_ids INTEGER[], recall FLOAT);");
        con.Query(results_query);
    }

    static void exportResultsToCSV(Connection& con, const std::string& table_name) {
        con.Query("COPY memory.recall_stats TO '" + table_name + "_output.csv' (HEADER, DELIMITER ',');");
    }   

    
};

// ==================== Query Runner ====================
class QueryRunner {
public:
    static void runTestQueries(Connection& con, const std::string& table_name, int vector_dimensionality,
                               HNSWIndex& hnsw_index, const unique_ptr<MaterializedQueryResult>& test_vectors,
                               Appender& appender, int iteration) {
        std::cout << "🧪 Running test queries using HNSWLib 🧪" << std::endl;

        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            std::vector<float> test_query_vector(vector_dimensionality);
            test_vectors->GetValue(1, i);

            int test_query_vector_index = test_vectors->GetValue(0, i).GetValue<int>();
            Value neighbor_ids = test_vectors->GetValue(2, i);

            auto result = hnsw_index.search(test_query_vector, 100);

            std::cout << "Query " << i << " returned " << result.size() << " results" << std::endl;
           

            appender.AppendRow(
                Value(table_name), Value::INTEGER(iteration), Value::INTEGER(test_query_vector_index),
                neighbor_ids, Value::LIST(std::vector<Value>(result.begin(), result.end())), Value::FLOAT(0.0)
            );
        }
    }

    static void calculateRecall(Connection& con, const std::string& table_name) {
        std::cout << "🧮 CALCULATING RECALL 🧮" << std::endl;
        con.Query("UPDATE " + table_name + "_results " + 
                  "SET recall = len(list_intersect(neighbor_vec_ids, result_vec_ids)) / 100.0;");
    }

    static void aggregateRecallStats(Connection& con, const std::string& table_name) {
        std::cout << "🧮 AGGREGATING RECALL STATS 🧮" << std::endl;
        con.Query(
            "INSERT INTO recall_stats "
            "SELECT "
            "dataset, "
            "iteration, "
            "COUNT(*) AS num_queries, "
            "favg(recall) AS mean_recall, "
            "MEDIAN(recall) AS median_recall, "
            "STDDEV_POP(recall) AS stddev_recall, "
            "VAR_POP(recall) AS var_recall, "
            "MIN(recall) AS min_recall, "
            "MAX(recall) AS max_recall, "
            "APPROX_QUANTILE(recall, 0.25) AS p25_recall, "
            "APPROX_QUANTILE(recall, 0.75) AS p75_recall, "
            "APPROX_QUANTILE(recall, 0.95) AS p95_recall "
            "FROM " + table_name + "_results GROUP BY dataset, iteration ORDER BY iteration ASC;"
        );
    }
};

class FileOperations {
    public:
        static void initConnectivitySummaryFile() {
            std::ofstream csv_file("connectivity_summary.csv", std::ios::trunc);
            csv_file << "nodes_count,unreachable_count,orphaned_count" << std::endl;
            csv_file << "0,0,0" << std::endl;
            csv_file.close();
        }
        
        static void mergeCSVFiles(const std::string& table_name) {
            std::string output_file = table_name + "_output.csv";
            std::string connectivity_file = "connectivity_summary.csv";
            std::string merged_file = table_name + "_merged_report.csv";
            
            std::ifstream output_stream(output_file);
            std::ifstream connectivity_stream(connectivity_file);
            std::ofstream merged_stream(merged_file);
            
            if (!output_stream.is_open() || !connectivity_stream.is_open() || !merged_stream.is_open()) {
                std::cerr << "Error: Could not open files for merging" << std::endl;
                return;
            }
            
            // Read and combine headers
            std::string output_header, connectivity_header;
            std::getline(output_stream, output_header);
            std::getline(connectivity_stream, connectivity_header);
            
            merged_stream << output_header << "," << connectivity_header << std::endl;
            
            // Combine the data rows
            std::string output_line, connectivity_line;
            while (std::getline(output_stream, output_line) && std::getline(connectivity_stream, connectivity_line)) {
                merged_stream << output_line << "," << connectivity_line << std::endl;
            }
            
            std::cout << "✅ Created merged report: " << merged_file << " ✅" << std::endl;
            
            output_stream.close();
            connectivity_stream.close();
            merged_stream.close();
        }

        
    };

// ==================== Main Test Runner ====================
class RecallTestRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;

public:
    RecallTestRunner(int iterations = 120) : db(nullptr), con(db), max_iterations(iterations) {
        con.Query("SET THREADS TO 1;");
        datasets = getDatasetConfigs();
    }

    void runTest(int datasetIdx = 0) {
        try {
            if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
                std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
                return;
            }

            const auto& dataset = datasets[datasetIdx];
            std::cout << "📊 Testing dataset: " << dataset.name << " 📊" << std::endl;


            DatabaseSetup::initializeResultsTable(con, dataset.name);
            DatabaseSetup::setupFullDataset(con, dataset);
            

            HNSWIndex hnsw_index(dataset.dimensions);
            hnsw_index.initializeIndex(con, dataset.name);

            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");

            Appender appender(con, dataset.name + "_results");

            for (int iteration = 0; iteration <= max_iterations; iteration++) {
                QueryRunner::runTestQueries(con, dataset.name, dataset.dimensions, hnsw_index, test_vectors, appender, iteration);
                std::cout << "✅ Finished iteration " << iteration << " ✅" << std::endl;
            }

            appender.Close();

            // Calculate recall and aggregate stats
            QueryRunner::calculateRecall(con, dataset.name);
            QueryRunner::aggregateRecallStats(con, dataset.name);
                        
            // Export results
            DatabaseSetup::exportResultsToCSV(con, dataset.name);
                        
            // Merge CSV files
            FileOperations::mergeCSVFiles(dataset.name);
            

        } catch (std::exception& e) {
            std::cerr << "Error running test: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {
    try {
        RecallTestRunner runner(119);
        runner.runTest(1);
        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}