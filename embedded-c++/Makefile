

.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

main:
	mkdir -p build
	cd build && cmake .. && make
	# build/results
	build/example --benchmark_out=bm_gist_5-20c.json --benchmark_out_format=json
	# build/example


