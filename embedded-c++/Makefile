

.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

main:
	mkdir -p build
	cd build && cmake .. && make
	build/example --benchmark_out=clustered_vss_benchmarks.txt --benchmark_out_format=console
	build/example --benchmark_out=clustered_vss_benchmarks.json --benchmark_out_format=json


