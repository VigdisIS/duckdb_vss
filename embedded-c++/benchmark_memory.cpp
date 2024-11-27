#include "duckdb.hpp"
#include <iostream>
#include <vector>
#include <sstream>
#include <sys/resource.h>

using namespace duckdb;

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(Connection &con, const std::string& table_name, int& vector_dimensionality) {
    std::cout << "Setting up table " << table_name << std::endl;
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_train SELECT * FROM raw." + table_name + "_train;");

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test SELECT * FROM raw." + table_name + "_test;");

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

int GetClusterAmount(int clusterAmount) {
    return clusterAmount;
}

void CleanUpIndexes(Connection &con) {
    auto indexes = con.Query("select distinct index_name from duckdb_indexes;");
	for (idx_t i = 0; i < indexes->RowCount(); i++) {
		con.Query("DROP INDEX IF EXISTS \"" + indexes->GetValue(0, i).ToString() + "\";");
 	}
    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
    assert(indexes_count->GetValue(0, 0).ToString() == "0");
}

void SetupIndex(Connection &con, const std::string& table_name, int cluster_amount) {
    auto cluster_amount_string = std::to_string(GetClusterAmount(cluster_amount));

    CleanUpIndexes(con);

    std::cout << "Setting up index for " << table_name << " with cluster amount " << cluster_amount_string << std::endl;
    con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    auto indexes_count_updated = con.Query("select count(index_name) from duckdb_indexes;");
    std::cout << "Index size: " << indexes_count_updated->GetValue(0, 0).ToString() << " Expected "  << std::to_string(cluster_amount + 1) << std::endl;
    assert(indexes_count_updated->GetValue(0, 0) == (cluster_amount + 1)); // cluster_amount + 1 (centroid_index)
}

// Function to get current memory usage in KB
long get_mem_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // in KB
}

void GetMemoryUsage(const std::string& table_name, int& vector_dimensionality) {
    DuckDB db(nullptr); // Open the memory database
    Connection con(db); // Establish a connection

    SetupTable(con, table_name, vector_dimensionality);
    std::vector<int> cluster_amounts = {5, 10, 15, 20};

    for (int cluster_amount : cluster_amounts) {
        CleanUpIndexes(con);
        SetupIndex(con, table_name, cluster_amount);
    }
}

int main(int argc, char** argv) {
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
            auto table_name = GetTableName(tableIndex);
            auto vector_dimensionality = GetVectorDimensionality(tableIndex);

            GetMemoryUsage(table_name, vector_dimensionality);
    }

    // // Measure memory before setting up the table
    // mem_before = get_mem_usage();
    // SetupTable(table_name, vector_dimensionality);
    // // Measure memory after setting up the table
    // mem_after = get_mem_usage();
    // std::cout << "Memory used by SetupTable: " << (mem_after - mem_before) << " KB\n";

    // // Measure memory before query execution
    // mem_before = get_mem_usage();
    // auto result = con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    // // Measure memory after query execution
    // mem_after = get_mem_usage();
    // std::cout << "Memory used by Query execution: " << (mem_after - mem_before) << " KB\n";

    // result->Print();

    return 0;
}
