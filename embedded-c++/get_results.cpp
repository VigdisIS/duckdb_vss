#include "duckdb.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono> 
#include <thread> 

using namespace duckdb;

// void DisplayProgressBar(int currentStep, int totalSteps) {
//     // Calculate the percentage of completion
//     float progress = (float)currentStep / totalSteps;
//     int barWidth = 50;

//     std::cout << "[";
//     int pos = barWidth * progress;
//     for (int i = 0; i < barWidth; ++i) {
//         if (i < pos) std::cout << "=";
//         else if (i == pos) std::cout << ">";
//         else std::cout << " ";
//     }
//     std::cout << "] " << int(progress * 100.0) << " %\r";
//     std::cout.flush();
// }

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(Connection& con, const std::string& table_name, int& vector_dimensionality) {
	con.Query("SET threads = 10;"); // My puter has 10 cores
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_train" + " SELECT * FROM raw." + table_name + "_train" + ";");

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test" + " SELECT * FROM raw." + table_name + "_test LIMIT 100;");

    con.Query("DETACH raw;");
}

std::string GetTableName(int tableIndex) {
    switch (tableIndex) {
        case 0: return "fashion_mnist";
        case 1: return "mnist";
        case 2: return "sift";
        case 3: return "gist";
        default: return "unknown";
    }
}

int GetVectorDimensionality(int tableIndex) {
    switch (tableIndex) {
        case 0: return 784;
        case 1: return 784;
        case 2: return 128;
        case 3: return 960;
        default: return 0;
    }
}

void InitializeResultsTable(Connection& con, const std::string& table_name, int& vector_dimensionality) {
    con.Query("CREATE OR REPLACE TABLE results." + table_name + "_results (dataset_name VARCHAR, cluster_amount INT, top_k INT, test_query_vector_index INT, test_query_vector FLOAT[" + std::to_string(vector_dimensionality) + "], cluster_index INT, query_result_set FLOAT[" + std::to_string(vector_dimensionality) + "][100]);");
    // con.Query("CREATE OR REPLACE TABLE results." + table_name + "_results (dataset_name VARCHAR, cluster_amount INT, top_k INT, test_query_vector_index INT, test_query_vector FLOAT[" + std::to_string(vector_dimensionality) + "], cluster_index INT, query_result_set FLOAT[" + std::to_string(vector_dimensionality) + "][100]);");
    // con.Query("CREATE OR REPLACE TABLE results." + table_name + "_results (dataset_name VARCHAR, cluster_amount INT, top_k INT, test_query_vector_index INT, test_query_vector VARCHAR, cluster_index INT, query_result_set VARCHAR);");
}

void GetResults() {
    DuckDB db(nullptr);
    Connection con(db);

    con.Query("ATTACH 'results.db' AS results;");

    // std::vector<int> cluster_amounts = {5};
    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    std::string cluster_index;

    for (int tableIndex = 1; tableIndex <= 3; ++tableIndex) {
        auto table_name = GetTableName(tableIndex);
        std::cout << "Table name: " << table_name << std::endl;
        auto vector_dimensionality = GetVectorDimensionality(tableIndex);

        SetupTable(con, table_name, vector_dimensionality);

        auto vec_dim_string = std::to_string(vector_dimensionality);
    
        auto test_vectors = con.Query("SELECT * FROM memory." + table_name + "_test" + ";");
        std::cout << "Test vector amount: " << test_vectors->RowCount() << std::endl;

        InitializeResultsTable(con, table_name, vector_dimensionality);

        std::string full_table_name = table_name + "_results";
        std::cout << "Results table: " << full_table_name << std::endl;

        con.Query("USE results.main;");
        auto hmm = con.Query("SHOW ALL TABLES;");
        hmm->Print();

        Appender appender(con, full_table_name);

        for (int cluster_amount : cluster_amounts) {
            std::cout << "Running test queries on index with " << cluster_amount << " clusters" << std::endl;
            auto cluster_amount_string = std::to_string(cluster_amount);
            con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train" + " USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
            std::cout << "Index created" << std::endl;

            for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
                // DisplayProgressBar(i, test_vectors->RowCount());
                std::cout << "Running test query " << i << " of " << test_vectors->RowCount() << std::endl;
                auto test_query_vector = test_vectors->GetValue(0, i);
                auto test_query_vector_string = test_vectors->GetValue(0, i).ToString();
                int test_query_vector_index = i;
                auto test_query_vector_index_string = std::to_string(test_query_vector_index);

                auto result = con.Query("SELECT * FROM memory." + table_name + "_train" + " ORDER BY array_distance(vec, " + test_query_vector_string + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");

                std::string result_string = "[";

                for (idx_t j = 0; j < result->RowCount(); j++) {
                    auto row = result->GetValue(0, j);
                    result_string += row.ToString();
                    // Check if this is not the last iteration
                    if (j < result->RowCount() - 1) {
                        result_string += ", ";
                    }
                }

                result_string += "]";

                std::ifstream file("cluster_indexes.txt");
                if (file.is_open()) {
                    if (!std::getline(file, cluster_index)) {
                        std::cout << "File is empty or error occurred while reading." << std::endl;
                    }
                    file.close();
                } else {
                    std::cout << "Unable to open file." << std::endl;
                }

                appender.AppendRow(Value(table_name), Value::INTEGER(cluster_amount), Value::INTEGER(100), Value::INTEGER(test_query_vector_index), Value(test_query_vector), Value(std::stoi(cluster_index)), Value(result_string));
                
            }
            auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = '" + table_name + "_train';");
            for (idx_t i = 0; i < indexes->RowCount(); i++) {
                con.Query("DROP INDEX IF EXISTS \"" + indexes->GetValue(0, i).ToString() + "\";");
            }
        }
        appender.Close();
    }

    con.Query("DETACH results;");
}

void OutputResultTables() {
    DuckDB db(nullptr);
    Connection con(db);

    con.Query("SET threads = 10;");

    con.Query("ATTACH 'results.db' AS results;");

    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        auto table_name = GetTableName(tableIndex);
        con.Query("COPY results." + table_name + "_results TO '" + table_name + "_results.csv' (HEADER, DELIMITER ',');");
    }

    auto res = con.Query("SELECT * FROM results.fashion_mnist_results limit 1;");
    res->Print();

    con.Query("DETACH results;");
}

int main() {
    // GetResults();
    OutputResultTables();

    return 0;
}
