.PHONY: clean prepare-build duckdb-vss hnswlib

all: hnswlib

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

prepare-build:
	mkdir -p build

hnswlib: clean prepare-build
	cd build && cmake .. && make
	build/hnswlib_test

usearch: clean prepare-build
	cd build && cmake .. && make
	build/usearch_test
