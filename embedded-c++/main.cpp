#include "duckdb.hpp"

using namespace duckdb;

int main() {
	DuckDB db(nullptr);

	Connection con(db);

	con.Query("SET threads TO 1");

	con.Query("CREATE TABLE my_vector_table (vec FLOAT[3])");
	con.Query("INSERT INTO my_vector_table SELECT array_value(a,b,c) FROM range(1,10) ra(a), range(1,10) rb(b), range(1,10) rc(c)");
	con.Query("CREATE INDEX my_hnsw_index ON my_vector_table USING HNSW (vec)");
	auto result = con.Query("SELECT * FROM my_vector_table ORDER BY array_distance(vec, [9,9,9]::FLOAT[3]) LIMIT 3");
	result->Print();
	// con.Query("SELECT * FROM my_vector_table ORDER BY array_distance(vec, [1,2,3]::FLOAT[3]) LIMIT 3");
	// auto result_duckdb_indexes =  con.Query("SELECT index_name FROM duckdb_indexes()");
	// result_duckdb_indexes->Print();
}
