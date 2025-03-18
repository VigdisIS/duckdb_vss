#include "duckdb.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <hnswlib/hnswlib.h>

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

class HelperFunctions {
    public:
        static std::vector<float> parseVector(const std::string& vec_str) {
            std::vector<float> vec;
            size_t start = vec_str.find_first_of("[");
            size_t end = vec_str.find_last_of("]");
            if (start == std::string::npos || end == std::string::npos) {
                return vec;
            }
            std::string values_str = vec_str.substr(start + 1, end - start - 1);
            size_t pos = 0;
            while ((pos = values_str.find(",")) != std::string::npos) {
                vec.push_back(std::stof(values_str.substr(0, pos)));
                values_str.erase(0, pos + 1);
            }
            vec.push_back(std::stof(values_str));
            return vec;
        }
    
        static std::string parseVector(const std::vector<float>& vec) {
            std::string vec_str = "[";
            for (size_t i = 0; i < vec.size(); i++) {
                vec_str += std::to_string(vec[i]);
                if (i < vec.size() - 1) {
                    vec_str += ", ";
                }
            }
            vec_str += "]";
            return vec_str;
        }
    };
    

// ==================== HNSW Index Operations ====================
class HNSWIndex {
private:
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<float> index;
    int dimensions;

    // Finding:
    // Label has to be next in line and should therefore be the next available label in the list
    std::unordered_map<size_t, size_t> index_map;
    int max_elements;

public:
    HNSWIndex(int dims, int max_elements = 120000, int ef_construction = 200, int m = 16)
        : space(dims), index(&space, max_elements, m, ef_construction, 100, true), dimensions(dims), max_elements(max_elements) {
        std::cout << "🆕 Created new in-memory HNSW index" << std::endl;
    }
    
    void initializeIndex(Connection& con, const std::string& table_name) {
        auto train_vectors = con.Query("SELECT id, vec FROM " + table_name + "_train;");
        std::cout << "🔄 Indexing " << train_vectors->RowCount() << " vectors..." << std::endl;

        HelperFunctions helper;
        

        for (idx_t i = 0; i < train_vectors->RowCount(); i++) {

            index_map[train_vectors->GetValue(0, i).GetValue<int>()] = train_vectors->GetValue(0, i).GetValue<int>();
          
            int id = train_vectors->GetValue(0, i).GetValue<int>();
            std::string vec_str = train_vectors->GetValue(1, i).ToString();
            std::vector<float> vec = HelperFunctions::parseVector(vec_str);
            
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
        auto mappedId = index_map[id];
        index.addPoint(vec.data(), mappedId);
    }

    std::vector<int> search(const std::vector<float>& vec, int k = 100) {
        auto results = index.searchKnn(vec.data(), k);
        std::vector<int> neighbors;
        while (!results.empty()) {
            neighbors.push_back(results.top().second);
            results.pop();
            
        }
        return neighbors;
    }

    void deleteVectors (const std::vector<int>& ids) {
        for (int id : ids) {
            auto mappedId = index_map.at(id);
            std::cout << "Deleting vector with id: " << id << " and internal id: " << mappedId << std::endl;
            index.markDelete(mappedId);
        }
    }

    void addVectorsAfterDeletion(const std::vector<int>& ids, const std::vector<std::vector<float>>& vecs) {
        for (int id: ids) {
            auto idx = index_map.at(id);
            size_t new_idx = (idx < (max_elements / 2)) ? idx + (max_elements / 2) : idx - (max_elements / 2);
            std::cout << "Adding vector with id: " << id << " and internal id: " << new_idx << std::endl;
            index.addPoint(vecs[id].data(), new_idx, true);
            index_map[id] = new_idx;
        }
    }

    int getInternalId(int id) {
        return index_map[id];
    }

    int getExternalId(int internal_id) {
        for (auto& pair : index_map) {
            if (pair.second == internal_id) {
                return pair.first;
            }
        }
        return -1;
    }
    
};

// ==================== Database Setup Functions ====================
class DatabaseSetup {
public:
    static void setupFullDataset(Connection& con, const DatasetConfig& config) {
        con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

        con.Query("CREATE OR REPLACE TABLE memory." + config.name + "_train AS SELECT * FROM raw." + config.name + "_train;");
        con.Query("CREATE OR REPLACE TABLE memory." + config.name + "_test AS SELECT * FROM raw." + config.name + "_test;");

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
                               HNSWIndex& hnsw_index, const unique_ptr<MaterializedQueryResult>& test_vectors, const unique_ptr<MaterializedQueryResult>& delete_vectors,
                               Appender& appender, int iteration) {
        std::cout << "🧪 Running test queries using HNSWLib 🧪" << std::endl;

        HelperFunctions helper;
                            
        std::vector<int> deletion_ids;
        std::vector<std::vector<float>> deletion_vecs;


        for (idx_t j = 0; j < delete_vectors->RowCount(); j++) {
            int id = delete_vectors->GetValue(0, j).GetValue<int>();
            std::string vec_str = delete_vectors->GetValue(1, j).ToString();
            std::vector<float> vec = HelperFunctions::parseVector(vec_str);

            deletion_ids.push_back(id);
            deletion_vecs.push_back(vec);
        }

      
        hnsw_index.deleteVectors(deletion_ids); 

        hnsw_index.addVectorsAfterDeletion(deletion_ids, deletion_vecs);


        std::cout << "🔍 Running test queries iteration: " << iteration << " 🔍" << std::endl;
        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            
            std::string vec_str = test_vectors->GetValue(1, i).ToString();
            std::vector<float> vec = HelperFunctions::parseVector(vec_str);
    
            int test_query_vector_index = test_vectors->GetValue(0, i).GetValue<int>();
            Value neighbor_ids = test_vectors->GetValue(2, i);
    

            auto result = hnsw_index.search(vec, 100);

            std::vector<Value> mapped_result_ids;
            for (int id : result) {
                mapped_result_ids.push_back(Value(hnsw_index.getExternalId(id)));
            }
           

            appender.AppendRow(
                Value(table_name), Value::INTEGER(iteration), Value::INTEGER(test_query_vector_index),
                neighbor_ids, Value::LIST(std::vector<Value>(mapped_result_ids.begin(), mapped_result_ids.end())), Value::FLOAT(0.0)
            );
        }
    }

    static void calculateRecall(Connection& con, const std::string& table_name, HNSWIndex& hnsw_index) {
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
            

            Appender appender(con, dataset.name + "_results");



            for (int iteration = 0; iteration <= max_iterations; iteration++) {
                auto delete_vectors = con.Query("SELECT id, vec FROM " + dataset.name + "_train LIMIT 600;");
                auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test USING SAMPLE 100 (reservoir);");



                std::cout << "✅ Fetched delete items:  " << delete_vectors->RowCount() << "vectors ✅" << std::endl;

                QueryRunner::runTestQueries(con, dataset.name, dataset.dimensions, hnsw_index, test_vectors, delete_vectors, appender, iteration);
                std::cout << "✅ Finished iteration " << iteration << " ✅" << std::endl;
            }

            appender.Close();

            // Calculate recall and aggregate stats
            QueryRunner::calculateRecall(con, dataset.name, hnsw_index);
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
        RecallTestRunner runner(3000);
        // Run test on fashion_mnist
        runner.runTest(0);
        //Run test on mnist
        runner.runTest(1);
        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}