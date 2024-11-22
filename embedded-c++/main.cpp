#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>
#include <vector>
#include <sstream>

using namespace duckdb;

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(Connection& con, const std::string& table_name, int& vector_dimensionality) {
	con.Query("SET threads = 10;"); // My puter has 10 cores
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

// Get random vector from train set to use as query vector
// Incurs extra overhead as the select query is run every iteration
std::string GetRandomRow(Connection& con, const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test USING SAMPLE 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
}

// Get first vector from test set
std::string GetFirstRow(Connection& con, const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test LIMIT 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
}

// Benchmark index creation
static void BM_ClusteringIndexCreation(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));

	SetupTable(con, table_name, vector_dimensionality);

    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    for (auto _ : state) {
		// Need to find some way to clean up after each cycle...
		// Or benchmark the time it takes to query and remove indexes and
		// remove this from the benchmark time
		auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = '" + table_name + "_train';");
		for (idx_t i = 0; i < indexes->RowCount(); i++) {
			con.Query("DROP INDEX IF EXISTS \"" + indexes->GetValue(0, i).ToString() + "\";");
    	}
        con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    }
}

// Benchmark for search after clustering
static void BM_ClusteringSearchRandomQuery(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);

	SetupTable(con, table_name, vector_dimensionality);
	con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + std::to_string(GetClusterAmount(state.range(1))) + ");");

    auto query_vector = GetRandomRow(con, table_name);

	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
    }
}

// Benchmark for search after clustering
static void BM_ClusteringSearchControlledQuery(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);

	SetupTable(con, table_name, vector_dimensionality);
	con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + std::to_string(GetClusterAmount(state.range(1))) + ");");

    auto query_vector = GetFirstRow(con, table_name);

	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
    }
}

void RegisterBenchmarks() {
    std::vector<int> cluster_amounts = {5, 10, 15};
    for (int tableIndex = 0; tableIndex <= 0; ++tableIndex) {
        for (int cluster_amount : cluster_amounts) {
            benchmark::RegisterBenchmark("BM_ClusteringIndexCreation", BM_ClusteringIndexCreation)
                ->Args({tableIndex, cluster_amount});
			benchmark::RegisterBenchmark("BM_ClusteringSearchControlledQuery", BM_ClusteringSearchControlledQuery)
                    ->Args({tableIndex, cluster_amount});
            benchmark::RegisterBenchmark("BM_ClusteringSearchRandomQuery", BM_ClusteringSearchRandomQuery)
                    ->Args({tableIndex, cluster_amount});
        }
    }
}

int main(int argc, char** argv) {
    RegisterBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    

    // DuckDB db(nullptr);
    // Connection con(db);


    // auto table_name = GetTableName(1);
    // auto vector_dimensionality = GetVectorDimensionality(1);
    // auto vec_dim_string = std::to_string(vector_dimensionality);

	// con.Query("SET threads = 10;"); // My puter has 10 cores
    // con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    // con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    // con.Query("INSERT INTO memory." + table_name + "_train SELECT * FROM raw." + table_name + "_train;");

	// con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    // con.Query("INSERT INTO memory." + table_name + "_test SELECT * FROM raw." + table_name + "_test limit 1;");

    // con.Query("DETACH raw;");

    // auto query_vector = GetFirstRow(con, table_name);

    // con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + std::to_string(GetClusterAmount(20)) + ");");
    // auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
    // std::cout << result->RowCount() << std::endl;

    // auto indexes = con.Query("select distinct index_name from duckdb_indexes;");
    // indexes->Print();
    // auto indexes_count = con.Query("select count(index_name) from duckdb_indexes");
    // assert(indexes_count->GetValue(0, 0).ToString() == "6"); // cluster_amount + 1 (centroid_index)
    // indexes_count->Print();

    // // con.Query("CREATE TABLE my_vector_table (vec FLOAT[3]);");
    // // con.Query("INSERT INTO my_vector_table SELECT array_value(a, b, c) FROM range(1, 10) ra(a), range(1, 10) rb(b), range(1, 10) rc(c);");
    // // con.Query("CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec) WITH (cluster_amount = 5);");

    // // auto result = con.Query("SELECT * FROM my_vector_table ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3]) LIMIT 3;");
    // // result->Print();

    // // auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = 'my_vector_table';");
    // // indexes->Print();
    // // auto indexes_count = con.Query("select count(index_name) from duckdb_indexes");
    // // assert(indexes_count->GetValue(0, 0).ToString() == "6"); // cluster_amount + 1 (centroid_index)
    // // indexes_count->Print();

    return 0;
}
