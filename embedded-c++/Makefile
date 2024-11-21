

.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

main:
	mkdir -p build
	cd build && cmake .. && make
	build/results
	# build/example --benchmark_out=bm_clustering_idx_search_controlled.txt --benchmark_out_format=console
	# build/example --benchmark_out=bm_clustering_idx_search_controlled.json --benchmark_out_format=json


