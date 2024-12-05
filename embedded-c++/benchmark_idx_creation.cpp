
#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>

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

// Benchmark index creation
static void BM_ClusteringIndexCreation(benchmark::State& state) {
    // Reset the database each run
    DuckDB db(nullptr);
    Connection con(db);

    auto table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
    auto cluster_amount_string = std::to_string(GetClusterAmount(state.range(1)));

    SetupTable(con, table_name, vector_dimensionality);
    
    CleanUpIndexes(con);

    for (auto _ : state) {
        con.Query("CREATE INDEX clustered_hnsw_index ON memory." + table_name + "_train USING HNSW (vec) WITH (cluster_amount = " + cluster_amount_string + ");");
    }
}

void RegisterBenchmarks() {
    std::vector<int> cluster_amounts = {5, 10, 15, 20};
    for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
        for (int cluster_amount : cluster_amounts) {
            benchmark::RegisterBenchmark("BM_ClusteringIndexCreation", BM_ClusteringIndexCreation)->Repetitions(tableIndex < 3 ? 5 : 3) // last two are FOUL😭 I simply do not have the time
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
