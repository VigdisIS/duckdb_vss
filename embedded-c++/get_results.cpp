#include "duckdb.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono> 
#include <thread> 

using namespace duckdb;

void SetupTrainTable(Connection& con, const std::string& table_name, int& vector_dimensionality) {
    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_train" + " SELECT * FROM raw." + table_name + "_train" + ";");
    auto res = con.Query("SELECT * FROM memory." + table_name + "_train" + " limit 1;");
    assert(res->RowCount() == 1);
}

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(Connection& con, const std::string& table_name, int& vector_dimensionality) {
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    SetupTrainTable(con, table_name, vector_dimensionality);

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test" + " SELECT * FROM raw." + table_name + "_test LIMIT 100;");
    auto test_res = con.Query("SELECT * FROM memory." + table_name + "_test" + " limit 1;");
    assert(test_res->RowCount() == 1);

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
    con.Query("CREATE OR REPLACE TABLE results." + table_name + "_results (dataset_name VARCHAR, cluster_amount INT, top_k INT, test_query_vector_index INT, test_query_vector FLOAT[" + std::to_string(vector_dimensionality) + "], cluster_index INT, result_vector_index INT, result_vector FLOAT[" + std::to_string(vector_dimensionality) + "]);");

    con.Query("INSERT INTO results." + table_name + "_results SELECT * FROM memory." + table_name + "_results;");
    auto res = con.Query("SELECT * FROM results." + table_name + "_results LIMIT 1;");
    assert(res->RowCount() == 1);
}

void GetResults() {
    DuckDB db(nullptr);
    Connection con(db);

    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    std::string cluster_index;

    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        auto table_name = GetTableName(tableIndex);
        std::cout << "Table name: " << table_name << std::endl;
        auto vector_dimensionality = GetVectorDimensionality(tableIndex);

        SetupTable(con, table_name, vector_dimensionality);

        auto vec_dim_string = std::to_string(vector_dimensionality);
    
        auto test_vectors = con.Query("SELECT * FROM " + table_name + "_test" + ";");
        std::cout << "Test vector amount: " << test_vectors->RowCount() << std::endl;

        std::string full_table_name = table_name + "_results";
        std::cout << "Results table: " << full_table_name << std::endl;

        con.Query("CREATE OR REPLACE TABLE " + table_name + "_results (dataset_name VARCHAR, cluster_amount INT, top_k INT, test_query_vector_index INT, test_query_vector FLOAT[" + std::to_string(vector_dimensionality) + "],  cluster_index INT, result_vector_index INT, result_vector FLOAT[" + std::to_string(vector_dimensionality) + "]);");

        auto abb = con.Query("CREATE SEQUENCE " + table_name + "_id_sequence START 1;");
        abb->Print();
        auto abbc = con.Query("ALTER TABLE memory." + table_name + "_train ADD COLUMN id INTEGER DEFAULT nextval('" + table_name + "_id_sequence');");
        abbc->Print();

        Appender appender(con, full_table_name);

        for (int cluster_amount : cluster_amounts) {
            std::cout << "Creating clustered HNSW index with " << cluster_amount << " clusters" << std::endl;
            auto cluster_amount_string = std::to_string(cluster_amount);
            con.Query("CREATE INDEX clustered_hnsw_index ON " + table_name + "_train" + " USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
            auto indexes_count = con.Query("select count(index_name) from duckdb_indexes");
            assert(indexes_count->GetValue(0, 0).ToString() == std::to_string((cluster_amount + 1))); // cluster_amount + 1 (centroid_index)
            std::cout << "Index created" << std::endl;

            for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
                std::cout << "Querying for test vector " << i + 1 << "/100" << std::endl;
                auto test_query_vector = test_vectors->GetValue(0, i);
                auto test_query_vector_string = test_vectors->GetValue(0, i).ToString();
                int test_query_vector_index = i;
                auto test_query_vector_index_string = std::to_string(test_query_vector_index);

                auto result = con.Query("SELECT * FROM " + table_name + "_train" + " ORDER BY array_distance(vec, " + test_query_vector_string + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");

                std::ifstream file("cluster_indexes.txt");
                if (file.is_open()) {
                    if (!std::getline(file, cluster_index)) {
                        std::cout << "File is empty or error occurred while reading." << std::endl;
                    }
                    file.close();
                } else {
                    std::cout << "Unable to open file." << std::endl;
                }

                for (idx_t j = 0; j < result->RowCount(); j++) {

                    auto result_vector = result->GetValue(0, j);
                    auto result_vector_string = result_vector.ToString();
                    auto result_vector_row = result->GetValue(1, j);
                    int result_vector_index_int = result_vector_row.GetValue<int>();
                    int result_vector_index;
                    if (result_vector_index_int == 0) {
                        result_vector_index = 0;
                    } else {
                        result_vector_index = result_vector_index_int - 1;
                        assert(result_vector_index_int - result_vector_index == 1);
                    }

                    appender.AppendRow(Value(table_name), Value::INTEGER(cluster_amount), Value::INTEGER(100), Value::INTEGER(test_query_vector_index), Value(test_query_vector), Value(std::stoi(cluster_index)), Value::INTEGER(result_vector_index), Value(result_vector));
                }
            }
            auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = '" + table_name + "_train';");
            for (idx_t i = 0; i < indexes->RowCount(); i++) {
                con.Query("DROP INDEX \"" + indexes->GetValue(0, i).ToString() + "\";");
            }
            auto indexes_none = con.Query("select * from duckdb_indexes;");
            assert(indexes_none->RowCount() == 0);
        }
        appender.Close();

        con.Query("ATTACH 'results.db' AS results;");
        InitializeResultsTable(con, table_name, vector_dimensionality);
        con.Query("DETACH results;");
    }
}

void OutputResultTables() {
    DuckDB db(nullptr);
    Connection con(db);

    con.Query("SET threads = 10;");

    con.Query("ATTACH 'results.db' AS results;");

    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        auto table_name = GetTableName(tableIndex);
        auto check = con.Query("SELECT * FROM results." + table_name + "_results LIMIT 1;");
        assert(check->RowCount() == 1);
        con.Query("COPY results." + table_name + "_results TO 'clustering_" + table_name + "_results.parquet' (FORMAT PARQUET);");
        auto check_pq = con.Query("SELECT * from 'clustering_" + table_name + "_results.parquet' limit 1;");
        check_pq->Print();
        assert(check_pq->RowCount() == 1);
    }

    con.Query("DETACH results;");
}

int main() {
    GetResults();
    OutputResultTables();

    return 0;
}
