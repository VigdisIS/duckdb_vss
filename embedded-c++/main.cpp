#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>
#include <vector>
#include <sstream>
#include <memory>

using namespace duckdb;

class CustomMemoryManager: public benchmark::MemoryManager {
public:

    int64_t num_allocs;
    int64_t max_bytes_used;


    void Start() BENCHMARK_OVERRIDE {
        num_allocs = 0;
        max_bytes_used = 0;
    }

    void Stop(Result& result) BENCHMARK_OVERRIDE {
        result.num_allocs = num_allocs;
        result.max_bytes_used = max_bytes_used;
    }
};

std::unique_ptr<CustomMemoryManager> mm(new CustomMemoryManager());

#ifdef MEMORY_PROFILER
void *custom_malloc(size_t size) {
    void *p = malloc(size);
    mm.get()->num_allocs += 1;
    mm.get()->max_bytes_used += size;
    return p;
}
#define malloc(size) custom_malloc(size)
#endif

DuckDB db(nullptr); // Open the memory database
Connection con(db); // Establish a connection

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(const std::string& table_name, int& vector_dimensionality) {
    std::cout << "Setting up table " << table_name << std::endl;
	// con.Query("SET threads = 10;"); // My puter has 10 cores
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

    std::cout << "Setting up index for " << table_name << " with cluster amount " << cluster_amount_string << std::endl;
    con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    auto indexes_count_updated = con.Query("select count(index_name) from duckdb_indexes;");
    std::cout << "Index size: " << indexes_count_updated->GetValue(0, 0).ToString() << " Expected "  << std::to_string(cluster_amount + 1) << std::endl;
    assert(indexes_count_updated->GetValue(0, 0) == (cluster_amount + 1)); // cluster_amount + 1 (centroid_index)
}

// Benchmark index creation
static void BM_ClusteringIndexCreation(benchmark::State& state) {
    auto table_name = GetTableName(state.range(0));
    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    CleanUpIndexes();

    for (auto _ : state) {
        con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    }
}

// Benchmark for search after clustering
static void BM_ClusteringSearchRandomQuery(benchmark::State& state) {
    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes;");
    if(indexes_count->GetValue(0, 0) != (state.range(1) + 1)) {
        std::cout << "Index size: " << indexes_count->GetValue(0, 0).ToString() << " Expected "  << std::to_string(state.range(1) + 1) << std::endl;
        SetupIndex(table_name, state.range(1));
    }

    auto query_vector = GetRandomRow(table_name);

	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
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
        std::cout << "Index size: " << indexes_count->GetValue(0, 0).ToString() << " Expected "  << std::to_string(state.range(1) + 1) << std::endl;
        SetupIndex(table_name, state.range(1));
    }

    auto query_vector = GetFirstRow(table_name);

	for (auto _ : state) {
        benchmark::DoNotOptimize(con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;"));
    }
}

void RegisterBenchmarks() {
    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {

            auto table_name = GetTableName(tableIndex);
            auto vector_dimensionality = GetVectorDimensionality(tableIndex);

            SetupTable(table_name, vector_dimensionality);
        for (int cluster_amount : cluster_amounts) {
            benchmark::RegisterBenchmark("BM_ClusteringIndexCreation", BM_ClusteringIndexCreation)->Repetitions(3)
                ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                    return *(std::max_element(std::begin(v), std::end(v)));
                })
                ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                    return *(std::min_element(std::begin(v), std::end(v)));
                })
                ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
                ->Args({tableIndex, cluster_amount});
			benchmark::RegisterBenchmark("BM_ClusteringSearchControlledQuery", BM_ClusteringSearchControlledQuery)->Repetitions(10)
                ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                    return *(std::max_element(std::begin(v), std::end(v)));
                })
                ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                    return *(std::min_element(std::begin(v), std::end(v)));
                })
                ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
                ->Args({tableIndex, cluster_amount});
            benchmark::RegisterBenchmark("BM_ClusteringSearchRandomQuery", BM_ClusteringSearchRandomQuery)->Repetitions(10)
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

void RegisterMemoryBenchmarks() {
    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {

            auto table_name = GetTableName(tableIndex);
            auto vector_dimensionality = GetVectorDimensionality(tableIndex);

            SetupTable(table_name, vector_dimensionality);
        for (int cluster_amount : cluster_amounts) {
            benchmark::RegisterBenchmark("BM_ClusteringIndexCreation", BM_ClusteringIndexCreation)
                ->Unit(benchmark::kMillisecond)
                ->Iterations(16)
                ->Args({tableIndex, cluster_amount});
			benchmark::RegisterBenchmark("BM_ClusteringSearchControlledQuery", BM_ClusteringSearchControlledQuery)
                ->Unit(benchmark::kMillisecond)
                ->Iterations(16)
                ->Args({tableIndex, cluster_amount});
            benchmark::RegisterBenchmark("BM_ClusteringSearchRandomQuery", BM_ClusteringSearchRandomQuery)
                ->Unit(benchmark::kMillisecond)
                ->Iterations(16)
                ->Args({tableIndex, cluster_amount});
        }
    }
}

void RunProgram(const std::string& table_name, int cluster_amount, int vector_dimensionality) {
    SetupTable(table_name, vector_dimensionality);

    auto query_vector = GetFirstRow(table_name);

    auto vec_dim_string = std::to_string(vector_dimensionality);
    auto cluster_amount_string = std::to_string(cluster_amount);

    con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    auto result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");
    std::cout << result->RowCount() << std::endl;

    auto indexes = con.Query("select distinct index_name from duckdb_indexes;");
    indexes->Print();
    auto indexes_count = con.Query("select count(index_name) from duckdb_indexes");
    assert(indexes_count->GetValue(0, 0).ToString() == std::to_string((cluster_amount + 1))); // cluster_amount + 1 (centroid_index)
    indexes_count->Print();
}

int main(int argc, char** argv) {

    benchmark::RegisterMemoryManager(mm.get());
    // RegisterBenchmarks(); // Pass the connection to use for benchmarks
    RegisterMemoryBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::RegisterMemoryManager(nullptr);


    // // benchmark::RegisterMemoryManager(mm.get());
    // DuckDB db(nullptr);
    // Connection con(db);

    // RegisterBenchmarks();
    // benchmark::Initialize(&argc, argv);
    // benchmark::RunSpecifiedBenchmarks();
    // benchmark::RegisterMemoryManager(nullptr);

    // DuckDB db(nullptr);
    // Connection con(db);

    // // con.Query("CREATE TABLE my_vector_table (vec FLOAT[3]);");
    // // con.Query("INSERT INTO my_vector_table SELECT array_value(a, b, c) FROM range(1, 10) ra(a), range(1, 10) rb(b), range(1, 10) rc(c);");

    // auto cluster_amount_string = std::to_string(5);

    // // con.Query("CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");

    // RunProgram(con, "fashion_mnist", 20, 784);
    // // std::vector<int> cluster_amounts = {5, 10, 15, 20};
    // // for (int tableIndex = 1; tableIndex <= 1; ++tableIndex) {
    // //     for (int cluster_amount : cluster_amounts) {
    // //         RunProgram(con, GetTableName(tableIndex), cluster_amount, GetVectorDimensionality(tableIndex));
    // //     }
    // // }


    // con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");

    // auto result = con.Query("EXPLAIN ANALYZE SELECT * FROM vector_table ORDER BY array_distance(vec, [1, 2, 3]::FLOAT[3]) LIMIT 3;");
    // result->Print();

    // auto indexes = con.Query("select distinct index_name from duckdb_indexes where table_name = 'my_vector_table';");
    // indexes->Print();
    // auto indexes_count = con.Query("select count(index_name) from duckdb_indexes");
    // assert(indexes_count->GetValue(0, 0).ToString() == "6"); // cluster_amount + 1 (centroid_index)
    // indexes_count->Print();

    return 0;
}
