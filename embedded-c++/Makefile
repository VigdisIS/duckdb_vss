
.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make release

main:
	mkdir -p build
	cd build && cmake .. && make
	build/example --benchmark_out=bm_vss_all.json --benchmark_out_format=json
	# build/results

