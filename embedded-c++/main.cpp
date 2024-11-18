#include "duckdb.hpp"
#include <iostream>
using namespace duckdb;

int main() {

	DuckDB db(nullptr);

	Connection con(db);
 
 	// con.Query("SET threads TO 1");
 
 	// con.Query("CREATE TABLE my_vector_table (vec FLOAT[3])");
 	// con.Query("INSERT INTO my_vector_table SELECT array_value(a,b,c) FROM range(1,10) ra(a), range(1,10) rb(b), range(1,10) rc(c)");
 	// // Create centroid HNSW index + cluster HNSW indexes
 	// con.Query("CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec)");
 	// // Get the index names
 	// auto result_duckdb_indexes =  con.Query("SELECT index_name FROM duckdb_indexes()");
 	// result_duckdb_indexes->Print();
 	// // Perform two-tiered vector similarity search
 	// auto result = con.Query("SELECT * FROM my_vector_table ORDER BY array_distance(vec, [9,9,9]::FLOAT[3]) LIMIT 3");
 	// result->Print();

	// DuckDB db(nullptr);
	// Connection con(db);

	con.Query("SET enable_progress_bar = true;");
	// // con.Query("SET hnsw_enable_experimental_persistence = true;");

	con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");
	std::cout << "Attached raw.db" << std::endl;

	con.Query("CREATE OR REPLACE TABLE memory.fashion_mnist_train (vec FLOAT[784])");
    con.Query("INSERT INTO memory.fashion_mnist_train SELECT * FROM raw.fashion_mnist_train;");
	std::cout << "Copied from raw to memory" << std::endl;

	con.Query("DETACH raw;");
	std::cout << "Detached raw.db" << std::endl;

    // con.Query("SET threads TO 1;");

	con.Query("CREATE INDEX clustered_hnsw_index ON memory.fashion_mnist_train USING HNSW (vec);");

	// Example query vector
    std::vector<float> query_vector(784, 0.1f); 

    // Start building the query string
    std::ostringstream query_stream;
    query_stream << "SELECT * FROM memory.fashion_mnist_train ORDER BY array_distance(vec, [";

    // Append each element of query_vector to the query string
    for (size_t i = 0; i < query_vector.size(); ++i) {
        query_stream << query_vector[i];
        if (i < query_vector.size() - 1) {
            // If not the last element, add a comma
            query_stream << ",";
        }
    }

    // Finish the query string
    query_stream << "]::FLOAT[" << query_vector.size() << "]) LIMIT 1";

    // Convert the stream into a string and execute the query
    auto result = con.Query(query_stream.str());
    result->Print();
}
