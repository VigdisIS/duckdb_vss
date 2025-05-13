.PHONY: duckdb-vss duckdb-vss-debug clean prepare-build build run-create-usearch-indexes run-usearch-random run-usearch-full-coverage run-usearch-new-data run-usearch-unreachable-points run-all-usearch-experiments run-hnswlib-random run-hnswlib-full-coverage run-hnswlib-new-data run-all-hnswlib-experiments run-all run-all-experiments

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

clean:
	rm -rf build

prepare-build:
	mkdir -p build

build: clean prepare-build
	cd build && cmake .. && make


# usearch experiments

run-create-usearch-indexes: build
	./build/usearch_create_indexes
	
run-usearch-random: build
	./build/usearch_random
run-usearch-full-coverage: build
	./build/usearch_full_coverage
run-usearch-new-data: build
	./build/usearch_new_data
run-usearch-unreachable-points-sampl-all: build
	./build/usearch_unreachable_points_sampl_all
run-usearch-unreachable-points-sampl-reachable: build
	./build/usearch_unreachable_points_sampl_reachable

run-all-usearch-experiments: build
	@echo "Running all usearch experiments..."
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points_sampl_all
	./build/usearch_unreachable_points_sampl_reachable
	@echo "All usearch experiments completed."
	

# hnswlib experiments

run-hnswlib-create-indexes: build
	./build/hnswlib_create_indexes

run-hnswlib-random: build
	./build/hnswlib_random
run-hnswlib-full-coverage: build
	./build/hnswlib_full_coverage
run-hnswlib-new-data: build
	./build/hnswlib_new_data
run-hnswlib-unreachable-points-sampl-all: build
	./build/hnswlib_unreachable_points_sampl_all
run-hnswlib-unreachable-points-sampl-reachable: build
	./build/hnswlib_unreachable_points_sampl_reachable

run-all-hnswlib-experiments: build
	@echo "Running all hnswlib experiments..."
	./build/hnswlib_random
	./build/hnswlib_full_coverage
	./build/hnswlib_new_data
	./build/hnswlib_unreachable_points_sampl_all
	./build/hnswlib_unreachable_points_sampl_reachable
	@echo "All hnswlib experiments completed."


# repl_cand_hnswlib experiments

run-repl-cand-hnswlib-random: build
	./build/repl_cand_hnswlib_random
run-repl-cand-hnswlib-full-coverage: build
	./build/repl_cand_hnswlib_full_coverage
run-repl-cand-hnswlib-new-data: build
	./build/repl_cand_hnswlib_new_data

run-all-repl-cand-hnswlib-experiments: build
	@echo "Running all repl_cand_hnswlib experiments..."
	./build/repl_cand_hnswlib_random
	./build/repl_cand_hnswlib_full_coverage
	./build/repl_cand_hnswlib_new_data
	@echo "All repl_cand_hnswlib experiments completed."


# only experiments in order (no setup)
run-all-experiments: build
	@echo "Running all experiments..."
	@echo "Running USearch experiments..."
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points_sampl_all
	./build/usearch_unreachable_points_sampl_reachable
	@echo "USearch experiments completed."
	@echo "Running HNSWLib experiments..."
	./build/hnswlib_random
	./build/hnswlib_full_coverage
	./build/hnswlib_new_data
	./build/hnswlib_unreachable_points_sampl_all
	./build/hnswlib_unreachable_points_sampl_reachable
	@echo "HNSWLib experiments completed."
	@echo "All experiments completed."
