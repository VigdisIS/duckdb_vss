

.PHONY: duckdb clean main

all: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

main:
	mkdir -p build
	cd build && cmake .. && make
	build/example


