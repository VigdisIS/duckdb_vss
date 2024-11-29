#include "duckdb.hpp"
#include <iostream>
#include <fstream>

using namespace duckdb;

DuckDB db(nullptr); // Open the memory database
Connection con(db); // Establish a connection

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(const std::string& table_name, int& vector_dimensionality) {
    std::cout << "Setting up table " << table_name << std::endl;
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_train SELECT * FROM raw." + table_name + "_train;");

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test SELECT * FROM raw." + table_name + "_test LIMIT 1;"); // We only need one row for the query vector

    con.Query("DETACH raw;");
}

std::string ProfilingFormat(int profileIndex) {
    switch (profileIndex) {
        case 0: return "query_tree";
        case 1: return "json";
        default: return "unknown";
    }
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

// Get first vector from test set
std::string GetFirstRow(const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test LIMIT 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
}

void CleanUpIndexes() {
    auto indexes = con.Query("select distinct index_name from duckdb_indexes;");
	for (idx_t i = 0; i < indexes->RowCount(); i++) {
		con.Query("DROP INDEX IF EXISTS \"" + indexes->GetValue(0, i).ToString() + "\";");
 	}
    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
    assert(indexes_count->GetValue(0, 0).ToString() == "0");
}

void RunProgram(int dataset, int cluster_amount) {
    auto table_name = GetTableName(dataset);
    auto vector_dimensionality = GetVectorDimensionality(dataset);
    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(cluster_amount);

    for (int profileIndex = 0; profileIndex <= 1; ++profileIndex) {
        std::string profile = ProfilingFormat(profileIndex);

        std::cout << "Running " << table_name << " with cluster amount " << cluster_amount_string << " and profile " << profile << std::endl;

        // Profile index creation
        auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
        assert(indexes_count->GetValue(0, 0).ToString() == "0");

        con.Query("PRAGMA enable_profiling = '" + profile + "';");
        con.Query("SET profiling_mode = 'detailed';");
        std::string profiling_output = "profiling/" + table_name + "/" + cluster_amount_string +"/idx_creation_" + profile + "." + ((profileIndex == 1) ? "json" : "txt");
        con.Query("SET profiling_output = '" + profiling_output +"';");
        con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
        con.Query("PRAGMA disable_profiling;");
        auto indexes_count_updated = con.Query("select count(index_name) from duckdb_indexes;");
        assert(indexes_count_updated->GetValue(0, 0) == (cluster_amount + 1)); // cluster_amount + 1 (centroid_index)

        // Profile search
        auto query_vector = GetFirstRow(table_name);

        con.Query("PRAGMA enable_profiling = '" + profile + "';");
        con.Query("SET profiling_mode = 'detailed';");
        profiling_output = "profiling/" + table_name + "/" + cluster_amount_string +"/search_" + profile + "." + ((profileIndex == 1) ? "json" : "txt");
        con.Query("SET profiling_output = '" + profiling_output +"';");
        auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
        con.Query("PRAGMA disable_profiling;");
        assert(result->RowCount() == 100); // Top k = 100

        CleanUpIndexes();
    }
}

void RunOnceAll(int dataset, int cluster_amount) {
    auto table_name = GetTableName(dataset);
    auto vector_dimensionality = GetVectorDimensionality(dataset);
    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(cluster_amount);

        std::cout << "Running " << table_name << " with cluster amount " << cluster_amount_string << std::endl;

        auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
        assert(indexes_count->GetValue(0, 0).ToString() == "0");

        auto res = con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
        res->Print();

        auto indexes_count_updated = con.Query("select count(index_name) from duckdb_indexes;");
        assert(indexes_count_updated->GetValue(0, 0) == (cluster_amount + 1)); // cluster_amount + 1 (centroid_index)

        auto query_vector = GetFirstRow(table_name);

        auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");

        assert(result->RowCount() == 100); // Top k = 100

        CleanUpIndexes();
}

int main(int argc, char** argv) {
    std::vector<int> cluster_amounts = {5, 10, 15, 20};

    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        auto table_name = GetTableName(tableIndex);
        auto vector_dimensionality = GetVectorDimensionality(tableIndex);

        SetupTable(table_name, vector_dimensionality);

        for (int cluster_amount : cluster_amounts) {
            RunOnceAll(tableIndex, cluster_amount);
        }

        con.Query("DROP TABLE memory." + table_name + "_train;");
        con.Query("DROP TABLE memory." + table_name + "_test;");
    }

    return 0;
}
