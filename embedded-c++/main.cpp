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
    con.Query("INSERT INTO memory." + table_name + "_train" + " SELECT * FROM raw." + table_name + "_train" + ";");

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test" + " SELECT * FROM raw." + table_name + "_test" + ";");

    con.Query("DETACH raw;");
}

std::string GetTableName(int tableIndex) {
    switch (tableIndex) {
        case 0: return "fashion_mnist";
        case 1: return "mnist";
        case 2: return "sift";
        case 3: return "gist"; // Too large for now..(?)
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

// Get random vector from train set to use as query vector
// Incurs extra overhead as the select query is run every iteration
std::string GetRandomRow(Connection& con, const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test" + " USING SAMPLE 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
}

// Get first vector from test set
std::string GetFirstRow(Connection& con, const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test" + " LIMIT 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
}

// Benchmark index creation
static void BM_VSSOriginalIndexCreation(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));

	SetupTable(con, table_name, vector_dimensionality);

    for (auto _ : state) {
		// Need to find some way to clean up after each cycle...
		// Or benchmark the time it takes to query and remove indexes and
		// remove this from the benchmark time
		auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = '" + table_name + "_train';");
		for (idx_t i = 0; i < indexes->RowCount(); i++) {
			con.Query("DROP INDEX \"" + indexes->GetValue(0, i).ToString() + "\";");
    	}
        // assert (con.Query("select count(index_name) from duckdb_indexes")->GetValue(0, 0).ToString() == std::to_string(0));
        con.Query("CREATE INDEX hnsw_index ON memory." + table_name + "_train" + " USING HNSW (vec);");

    }
}

// Benchmark search with same query for each table - 1st row from test set
static void BM_VSSOriginalSearchControlledQuery(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);

	SetupTable(con, table_name, vector_dimensionality);
	con.Query("CREATE INDEX hnsw_index ON memory." + table_name + "_train" + " USING HNSW (vec);");

    auto query_vector = GetFirstRow(con, table_name);

	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train" + " ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
    }
}

// Benchmark search with random queries each iteration
static void BM_VSSOriginalSearchRandomQuery(benchmark::State& state) {
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);

	SetupTable(con, table_name, vector_dimensionality);
	con.Query("CREATE INDEX hnsw_index ON memory." + table_name + "_train" + " USING HNSW (vec);");

    auto query_vector = GetRandomRow(con, table_name);
    
	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train" + " ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
    }
}

void RegisterBenchmarks() {
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        benchmark::RegisterBenchmark("BM_VSSOriginalIndexCreation", BM_VSSOriginalIndexCreation)
            ->Args({tableIndex});
		benchmark::RegisterBenchmark("BM_VSSOriginalSearchControlledQuery", BM_VSSOriginalSearchControlledQuery)
            ->Args({tableIndex});
        benchmark::RegisterBenchmark("BM_VSSOriginalSearchRandomQuery", BM_VSSOriginalSearchRandomQuery)
            ->Args({tableIndex});
    }
}

int main(int argc, char** argv) {
    RegisterBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    // DuckDB db(nullptr);
    // Connection con(db);

    // auto table_name = GetTableName(0);
    // auto vector_dimensionality = GetVectorDimensionality(0);
    // auto vec_dim_string = std::to_string(vector_dimensionality);

	// SetupTable(con, table_name, vector_dimensionality);

    // auto query_vector = GetFirstRow(con, table_name);

    // con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train" + " USING HNSW (vec);");
    // auto result = con.Query("SELECT * FROM memory." + table_name + "_train" + " ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
    // std::cout << result->RowCount() << std::endl;

    return 0;
}
