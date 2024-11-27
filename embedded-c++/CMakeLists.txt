cmake_minimum_required(VERSION 2.8.12...3.29)
project(example-c++)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-O3)

include_directories(../duckdb/src/include include)
link_directories(../build/release/src)

set(BENCHMARK_LIB_PATH "${CMAKE_SOURCE_DIR}/include/benchmark")
find_library(BENCHMARK_LIB NAMES benchmark PATHS ${BENCHMARK_LIB_PATH})
if(NOT BENCHMARK_LIB)
    message(FATAL_ERROR "Benchmark library not found.")
else()
    message(STATUS "Benchmark library found: ${BENCHMARK_LIB}")
endif()

# add_executable(example main.cpp)
add_executable(benchmark_idx_creation benchmark_idx_creation.cpp)
add_executable(benchmark_search benchmark_search.cpp)
add_executable(benchmark_memory benchmark_memory.cpp)
add_executable(results get_results.cpp)

# Linking libraries
target_link_libraries(benchmark_idx_creation duckdb ${BENCHMARK_LIB})
target_link_libraries(benchmark_search duckdb ${BENCHMARK_LIB})
target_link_libraries(benchmark_memory duckdb ${BENCHMARK_LIB})
target_link_libraries(results duckdb ${BENCHMARK_LIB})
