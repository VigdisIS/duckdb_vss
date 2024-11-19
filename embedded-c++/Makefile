
.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make release

main:
	mkdir -p build
	cd build && cmake .. && make
	build/example --benchmark_out=vss_benchmarks.json --benchmark_out_format=json
	build/example --benchmark_out=vss_benchmarks.txt --benchmark_out_format=console
