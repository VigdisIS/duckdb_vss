.PHONY: clean duckdb-vss duckdb-vss-debug bm_performance bm_memory results

all: duckdb-vss bm_performance

all-bm_memory: duckdb-vss bm_memory

all-main: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

prepare-build:
	mkdir -p build

bm_performance: clean prepare-build
	cd build && cmake .. && make
	build/benchmark_search --benchmark_out=bm_clustering_search_time.json --benchmark_out_format=json
	build/benchmark_idx_creation --benchmark_out=bm_clustering_idx_creation_time.json --benchmark_out_format=json

# Uncomment JSON output in physical operator before running this
bm_memory: clean prepare-build
	cd build && cmake .. && make
	build/benchmark_memory

# Uncomment cluster index output to txt file in scan operator before running this
results: clean prepare-build
	cd build && cmake .. && make
	build/results

main: clean prepare-build
	cd build && cmake .. && make
	build/example
