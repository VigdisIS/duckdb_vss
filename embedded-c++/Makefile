.PHONY: duckdb-vss duckdb-vss-debug clean prepare-build build build-usearch build-hnswlib build-repl-cand-hnswlib run-create-usearch-indexes run-usearch-random run-usearch-full-coverage run-usearch-new-data run-usearch-unreachable-points run-all-usearch-experiments run-hnswlib-random run-hnswlib-full-coverage run-hnswlib-new-data run-all-hnswlib-experiments run-all run-all-experiments

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

clean:
	rm -rf build

prepare-build:
	mkdir -p build

# Build all
build: clean prepare-build
	cd build && cmake .. && make

# Build specific subdirectories
build-usearch: prepare-build
	cd build && cmake .. && make -C usearch

build-hnswlib: prepare-build
	cd build && cmake .. && make -C hnswlib

build-repl-cand-hnswlib: prepare-build
	cd build && cmake .. && make -C repl_cand_hnswlib


# usearch experiments
run-create-usearch-indexes: build-usearch
	./build/usearch/create_indexes
	
run-usearch-random: build-usearch
	./build/usearch/random

run-usearch-full-coverage: build-usearch
	./build/usearch/full_coverage

run-usearch-new-data: build-usearch
	./build/usearch/new_data

run-usearch-unreachable-points-sampl-all: build-usearch
	./build/usearch/unreachable_points_sampl_all

run-usearch-unreachable-points-sampl-reachable: build-usearch
	./build/usearch/unreachable_points_sampl_reachable

run-all-usearch-experiments: build-usearch
	@echo "Running all usearch experiments..."
	./build/usearch/random
	./build/usearch/full_coverage
	./build/usearch/new_data
	./build/usearch/unreachable_points_sampl_all
	./build/usearch/unreachable_points_sampl_reachable
	@echo "All usearch experiments completed."
	

# hnswlib experiments
run-hnswlib-create-indexes: build-hnswlib
	./build/hnswlib/create_indexes

run-hnswlib-random: build-hnswlib
	./build/hnswlib/random

run-hnswlib-full-coverage: build-hnswlib
	./build/hnswlib/full_coverage

run-hnswlib-new-data: build-hnswlib
	./build/hnswlib/new_data

run-hnswlib-unreachable-points-sampl-all: build-hnswlib
	./build/hnswlib/unreachable_points_sampl_all

run-hnswlib-unreachable-points-sampl-reachable: build-hnswlib
	./build/hnswlib/unreachable_points_sampl_reachable

run-all-hnswlib-experiments: build-hnswlib
	@echo "Running all hnswlib experiments..."
	./build/hnswlib/random
	./build/hnswlib/full_coverage
	./build/hnswlib/new_data
	./build/hnswlib/unreachable_points_sampl_all
	./build/hnswlib/unreachable_points_sampl_reachable
	@echo "All hnswlib experiments completed."


# repl_cand_hnswlib experiments
run-repl-cand-hnswlib-random: build-repl-cand-hnswlib
	./build/repl_cand_hnswlib/random

run-repl-cand-hnswlib-full-coverage: build-repl-cand-hnswlib
	./build/repl_cand_hnswlib/full_coverage

run-repl-cand-hnswlib-new-data: build-repl-cand-hnswlib
	./build/repl_cand_hnswlib/new_data

run-repl-cand-hnswlib-unreachable-points-sampl-all: build-repl-cand-hnswlib
	./build/repl_cand_hnswlib/unreachable_points_sampl_all

run-repl-cand-hnswlib-unreachable-points-sampl-reachable: build-repl-cand-hnswlib
	./build/repl_cand_hnswlib/unreachable_points_sampl_reachable

run-all-repl-cand-hnswlib-experiments: build-repl-cand-hnswlib
	@echo "Running all repl_cand_hnswlib experiments..."
	./build/repl_cand_hnswlib/full_coverage
	./build/repl_cand_hnswlib/new_data
	./build/repl_cand_hnswlib/random
	./build/repl_cand_hnswlib/unreachable_points_sampl_all
	./build/repl_cand_hnswlib/unreachable_points_sampl_reachable
	@echo "All repl_cand_hnswlib experiments completed."


# current
run-repl-cand-and-hnswlib: build-repl-cand-hnswlib build-hnswlib
	./build/repl_cand_hnswlib/full_coverage
	./build/repl_cand_hnswlib/new_data
	./build/repl_cand_hnswlib/random
	./build/repl_cand_hnswlib/unreachable_points_sampl_reachable
	./build/repl_cand_hnswlib/unreachable_points_sampl_all
	./build/hnswlib/unreachable_points_sampl_reachable
	./build/hnswlib/unreachable_points_sampl_all
	./build/hnswlib/random
	./build/hnswlib/full_coverage
	./build/hnswlib/new_data
	

# only experiments in order (no setup)
run-all-experiments: build
	@echo "Running all experiments..."
	@echo "Running USearch experiments..."
	./build/usearch/random
	./build/usearch/full_coverage
	./build/usearch/new_data
	./build/usearch/unreachable_points_sampl_all
	./build/usearch/unreachable_points_sampl_reachable
	@echo "USearch experiments completed."
	@echo "Running HNSWLib experiments..."
	./build/hnswlib/random
	./build/hnswlib/full_coverage
	./build/hnswlib/new_data
	./build/hnswlib/unreachable_points_sampl_all
	./build/hnswlib/unreachable_points_sampl_reachable
	@echo "HNSWLib experiments completed."
	@echo "All experiments completed."