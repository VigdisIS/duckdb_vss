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

void AggregateStats() {

    std::string search_path = "clustering_time_operations_search_bm.json";
    std::string index_path = "clustering_time_operations_index_bm.json";

    con.Query("INSTALL json;");
    con.Query("LOAD json;");

    // SEARCH AGGREGATION

    con.Query("CREATE TABLE results AS SELECT * FROM read_json_auto('" + search_path + "');");
    con.Query("CREATE TABLE aaares as SELECT dataset, cluster_amount, unnest(searches, recursive := true), \"total_duration (ns)\" FROM results;");
    con.Query("CREATE TABLE hum as SELECT dataset, cluster_amount, \"duration (ns)\", if(index = 'centroid_index', true, false) as is_centroid, \"total_duration (ns)\" FROM aaares;");

    auto query = R""""(
        CREATE TABLE aggregated_stats AS
        SELECT
            concat(trim(dataset, '_train'), 't') AS dataset,
            cluster_amount,
            is_centroid,
            AVG("duration (ns)") AS mean_duration_ns,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "duration (ns)") AS median_duration_ns,
            STDDEV("duration (ns)") AS stddev_duration_ns,
            STDDEV("duration (ns)") / AVG("duration (ns)") AS cv_duration_ns,
            AVG("total_duration (ns)") AS mean_total_duration_ns,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "total_duration (ns)") AS median_total_duration_ns,
            STDDEV("total_duration (ns)") AS stddev_total_duration_ns,
            STDDEV("total_duration (ns)") / AVG("total_duration (ns)") AS cv_total_duration_ns
        FROM
            hum
        GROUP BY
            dataset, cluster_amount, is_centroid ;)"""";

    con.Query(query);
    auto hem = con.Query("select * from aggregated_stats order by dataset, cluster_amount, is_centroid");
    hem->Print();

    con.Query("COPY (select * from aggregated_stats order by dataset, cluster_amount, is_centroid) TO 'clustering_scan_tasks_stats_by_cluster.json' (FORMAT JSON, ARRAY true);");


    // INDEX AGGREGATION

    con.Query("CREATE TABLE results_index AS SELECT * FROM read_json_auto('" + index_path + "');");
    con.Query("CREATE TABLE results_unnested as SELECT dataset, cluster_amount, unnest(tasks, recursive := true) FROM results_index;");

    con.Query("select * from results_unnested");

    auto query_index = R""""(
        CREATE TABLE aggregated_stats_index AS
        SELECT
            concat(trim(dataset, '_train'), 't') AS dataset,
            cluster_amount,
            task,
            AVG("duration (ns)") AS mean_duration_ns,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY "duration (ns)") AS median_duration_ns,
            STDDEV("duration (ns)") AS stddev_duration_ns,
            STDDEV("duration (ns)") / AVG("duration (ns)") AS cv_duration_ns
        FROM
            results_unnested
        GROUP BY
            dataset, cluster_amount, task;)"""";

    con.Query(query_index);

    auto hem_index = con.Query("select * from aggregated_stats_index order by dataset, cluster_amount, task");
    hem_index->Print();

    con.Query("COPY (select * from aggregated_stats_index order by dataset, cluster_amount, task) TO 'clustering_create_tasks_stats_by_cluster.json' (FORMAT JSON, ARRAY true);");

    // Can also aggregate by only dataset and task, ignore cluster_amount

}

int main(int argc, char** argv) {
    // std::vector<int> cluster_amounts = {5, 10, 15, 20};

    // for (int tableIndex = 0; tableIndex <= 3; ++tableIndex) {
    //     auto table_name = GetTableName(tableIndex);
    //     auto vector_dimensionality = GetVectorDimensionality(tableIndex);

    //     SetupTable(table_name, vector_dimensionality);

    //     for (int cluster_amount : cluster_amounts) {
    //         RunOnceAll(tableIndex, cluster_amount);
    //     }

    //     con.Query("DROP TABLE memory." + table_name + "_train;");
    //     con.Query("DROP TABLE memory." + table_name + "_test;");
    // }

    AggregateStats();

    return 0;
}
