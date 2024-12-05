
#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>

using namespace duckdb;

// All share the same database and connection otherwise the indexes would be lost between random/controlled queries
DuckDB db(nullptr);
Connection con(db);

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(const std::string& table_name, int& vector_dimensionality) {
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

// Get random vector from train set to use as query vector
std::string GetRandomRow(const std::string& table_name) {
		auto result = con.Query("SELECT * FROM memory." + table_name + "_test USING SAMPLE 1;");
		if(result->ColumnCount() == 0 || result->RowCount() == 0) {
			std::cerr << "No data found in " << table_name << "_test" << std::endl;
			return "";
		}
		auto query_vector = result->GetValue(0, 0).ToString();
		return query_vector;
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

void SetupIndex(const std::string& table_name, int cluster_amount) {
    auto cluster_amount_string = std::to_string(GetClusterAmount(cluster_amount));

    CleanUpIndexes();

    con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    auto indexes_count_updated = con.Query("select count(index_name) from duckdb_indexes;");
    assert(indexes_count_updated->GetValue(0, 0) == (cluster_amount + 1)); // cluster_amount + 1 (centroid_index)
}

// Benchmark for search after clustering
static void BM_ClusteringSearchRandomQuery(benchmark::State& state) {
    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
    if(indexes_count->GetValue(0, 0) != (state.range(1) + 1)) {
        SetupIndex(table_name, state.range(1));
    }

    auto query_vector = GetRandomRow(table_name);

	for (auto _ : state) {
        auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

// Benchmark for search after clustering
static void BM_ClusteringSearchControlledQuery(benchmark::State& state) {
    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
    if(indexes_count->GetValue(0, 0) != (state.range(1) + 1)) {
        SetupIndex(table_name, state.range(1));
    }

    auto query_vector = GetFirstRow(table_name);

	for (auto _ : state) {
        auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
}

void RegisterBenchmarks() {
    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {

            auto table_name = GetTableName(tableIndex);
            auto vector_dimensionality = GetVectorDimensionality(tableIndex);

            SetupTable(table_name, vector_dimensionality);
        for (int cluster_amount : cluster_amounts) {
			benchmark::RegisterBenchmark("BM_ClusteringSearchControlledQuery", BM_ClusteringSearchControlledQuery)->Repetitions(100)
                ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                    return *(std::max_element(std::begin(v), std::end(v)));
                })
                ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                    return *(std::min_element(std::begin(v), std::end(v)));
                })
                ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
                ->Args({tableIndex, cluster_amount});
            benchmark::RegisterBenchmark("BM_ClusteringSearchRandomQuery", BM_ClusteringSearchRandomQuery)->Repetitions(100)
                ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                    return *(std::max_element(std::begin(v), std::end(v)));
                })
                ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                    return *(std::min_element(std::begin(v), std::end(v)));
                })
                ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
                ->Args({tableIndex, cluster_amount});
        }
    }
}

int main(int argc, char** argv) {
    RegisterBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
