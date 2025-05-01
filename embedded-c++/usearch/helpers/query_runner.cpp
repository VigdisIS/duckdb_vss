#include "query_runner.h"

void QueryRunner::calculateRecall(Connection& con, const std::string& table_name) {
    std::cout << "🧮 CALCULATING RECALL 🧮" << std::endl;
    con.Query("UPDATE " + table_name + "_results " +
                "SET recall = len(list_intersect(neighbor_vec_ids, result_vec_ids)) / len(neighbor_vec_ids);");
    con.Query("UPDATE early_terminated_queries "
                "SET recall = len(list_intersect(neighbor_vec_ids, result_vec_ids)) / len(neighbor_vec_ids);");
}

void QueryRunner::aggregateRecallStats(Connection& con, const std::string& table_name) {
    std::cout << "🧮 AGGREGATING RECALL STATS 🧮" << std::endl;
    con.Query(
        "INSERT INTO recall_stats "
        "SELECT "
        "dataset, "
        "iteration, "
        "COUNT(*) AS num_queries, "
        "favg(recall) AS mean_recall, "
        "MEDIAN(recall) AS median_recall, "
        "STDDEV_POP(recall) AS stddev_recall, "
        "VAR_POP(recall) AS var_recall, "
        "MIN(recall) AS min_recall, "
        "MAX(recall) AS max_recall, "
        "APPROX_QUANTILE(recall, 0.25) AS p25_recall, "
        "APPROX_QUANTILE(recall, 0.75) AS p75_recall, "
        "APPROX_QUANTILE(recall, 0.95) AS p95_recall, "
        "favg(computed_distances) AS mean_computed_distances, "
        "MEDIAN(computed_distances) AS median_computed_distances, "
        "STDDEV_POP(computed_distances) AS stddev_computed_distances, "
        "VAR_POP(computed_distances) AS var_computed_distances, "
        "MIN(computed_distances) AS min_computed_distances, "
        "MAX(computed_distances) AS max_computed_distances, "
        "favg(visited_members) AS mean_visited_members, "
        "MEDIAN(visited_members) AS median_visited_members, "
        "STDDEV_POP(visited_members) AS stddev_visited_members, "
        "VAR_POP(visited_members) AS var_visited_members, "
        "MIN(visited_members) AS min_visited_members, "
        "MAX(visited_members) AS max_visited_members, "
        "favg(results_count) AS mean_results_count, "
        "MEDIAN(results_count) AS median_results_count, "
        "STDDEV_POP(results_count) AS stddev_results_count, "
        "VAR_POP(results_count) AS var_results_count, "
        "MIN(results_count) AS min_results_count, "
        "MAX(results_count) AS max_results_count "
        "FROM " + table_name + "_results GROUP BY dataset, iteration ORDER BY iteration ASC;"
    );
}

void QueryRunner::aggregateBMStats(Connection& con, const std::string& table_name, size_t num_queries, size_t num_del_add) {
    std::cout << "🧮 AGGREGATING " + table_name + " BM STATS 🧮" << std::endl;
    con.Query(
        "INSERT INTO " + table_name + "_bm_stats "
        "SELECT "
        "dataset, "
        "iteration, "
        + std::to_string(num_queries) + " AS num_queries, "
        + std::to_string(num_del_add) + " AS num_del_add, "
        "favg(operation_time) AS mean_time, "
        "MEDIAN(operation_time) AS median_time, "
        "STDDEV_POP(operation_time) AS stddev_time, "
        "VAR_POP(operation_time) AS var_time, "
        "MIN(operation_time) AS min_time, "
        "MAX(operation_time) AS max_time "
        "FROM " + table_name + "_bm GROUP BY dataset, iteration ORDER BY iteration ASC;"
    );
}

void QueryRunner::outputTableAsCSV(Connection& con, const std::string& full_table_name, const std::string& output_file_name) {
    con.Query("COPY " + full_table_name + " TO '" + output_file_name + "' (HEADER, DELIMITER ',');");
    std::cout << "📤 Exported " << full_table_name << " to " << output_file_name << " 📤" << std::endl;
}

unique_ptr<MaterializedQueryResult> QueryRunner::getSampleVectors(Connection& con, const std::string& table_name, int rows) {
    return con.Query(
        "SELECT * FROM " + table_name + "_train USING SAMPLE " +
        std::to_string(rows)+ ";");
}

unique_ptr<MaterializedQueryResult> QueryRunner::getSampleReachableVectors(Connection& con, const std::string& table_name, int rows, std::unordered_set<size_t>& available_points) {
    std::string available_points_str;
    for (const auto& point : available_points) {
        available_points_str += std::to_string(point) + ",";
    }
    available_points_str.pop_back(); // Remove the last comma
    return con.Query(
        "SELECT * FROM " + table_name + "_train where id in (" + available_points_str + ") USING SAMPLE " +
        std::to_string(rows)+ ";");
}

std::vector<unique_ptr<MaterializedQueryResult>> QueryRunner::partitionDataset(
    Connection& con, 
    const std::string& table_name, 
    int num_partitions
) {
    std::cout << "🧩 Partitioning " << table_name << " into " << num_partitions << " parts 🧩" << std::endl;
    
    // First get the total count and row IDs range
    auto count_result = con.Query("SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM " + table_name + "_train");
    int64_t total_count = count_result->GetValue<int64_t>(0, 0);
    int64_t min_rowid = count_result->GetValue<int64_t>(1, 0);
    int64_t max_rowid = count_result->GetValue<int64_t>(2, 0);
    
    std::vector<unique_ptr<MaterializedQueryResult>> partitions;
    partitions.reserve(num_partitions);
    
    // Calculate partitions based on row counts
    int64_t rows_per_partition = (total_count + num_partitions - 1) / num_partitions; // ceiling division
    
    std::cout << "Total rows: " << total_count << ", rows per partition: " << rows_per_partition << std::endl;
    
    // Create each partition
    for (int i = 0; i < num_partitions; i++) {
        auto partition = con.Query(
            "SELECT * FROM " + table_name + "_train "
            "ORDER BY rowid "  // Ensure consistent ordering
            "LIMIT " + std::to_string(rows_per_partition) + 
            " OFFSET " + std::to_string(i * rows_per_partition)
        );
        
        partitions.push_back(std::move(partition));
        
        std::cout << "Partition " << i+1 << " created with " 
                  << partitions.back()->RowCount() << " rows" << std::endl;
    }
    
    return partitions;
}

void QueryRunner::updateTestDataset(Connection& con, const DatasetConfig& dataset, Appender& test_appender, std::vector<std::tuple<int, std::vector<float>, std::vector<size_t>>>& ground_truth_indices, int test_vectors_count) {
    // Truncate test table to remove old test vectors and replace w ground truth
    con.Query("DELETE FROM " + dataset.name + "_test;");
    auto test_res = con.Query("select * from " + dataset.name + "_test;");
    assert(test_res->RowCount() == 0);

    std::vector<Value> gt_neighbor_ids;
    gt_neighbor_ids.reserve(std::get<2>(ground_truth_indices[0]).size());
    std::vector<Value> query_vec_val;
    query_vec_val.reserve(dataset.dimensions);

    // Add ground truth results to test table
    for (const auto& result : ground_truth_indices) {
        try {
            // Create result value
            Value result_list_value;
            std::vector<Value> id_values;
            id_values.reserve(std::get<2>(result).size());
            for (std::size_t j = 0; j < std::get<2>(result).size(); ++j) {
                size_t key = static_cast<size_t>(std::get<2>(result)[j]);
                id_values.push_back(Value::INTEGER(key));
            }
            result_list_value =  Value::LIST(LogicalType::INTEGER, std::move(id_values));
            // Create vector value
            Value vector_list_value;
            std::vector<Value> vec_values;
            vec_values.reserve(std::get<1>(result).size());
            for (std::size_t j = 0; j < std::get<1>(result).size(); ++j) {
                vec_values.push_back(Value::FLOAT(std::get<1>(result)[j]));
            }
            vector_list_value =  Value::LIST(LogicalType::FLOAT, std::move(vec_values));

            test_appender.AppendRow(
                Value(std::get<0>(result)),
                vector_list_value,
                result_list_value
            );
        } catch (const std::exception& e) {
            std::cerr << "Error appending ground truth result to test table: " << e.what() << std::endl;
        }
    }

    test_appender.Flush();

    auto test_res_after = con.Query("select * from " + dataset.name + "_test;");
    assert(test_res_after->RowCount() == test_vectors_count);
}
