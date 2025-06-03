cmake_minimum_required(VERSION 2.8.12...3.29)
project(vss-experiments)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -ffast-math)
# add_compile_options(-g) # for debug -- SLOW

# Common include directories
include_directories(
    ../duckdb/src/include
    include
    ../src/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/usearch/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib/helpers
)

# link_directories(../build/debug/src) # for debug
link_directories(../build/release/src)

# List all implementation files explicitly
set(UTILS_SOURCES
    usearch/helpers/index_operations.cpp
    usearch/helpers/database_setup.cpp
    usearch/helpers/file_operations.cpp
    usearch/helpers/query_runner.cpp
)
set(HNSWLIB_UTILS_SOURCES
    hnswlib/helpers/thread_pool.cpp
    hnswlib/helpers/util.cpp
    hnswlib/helpers/hnswlib_index_operations.cpp
)

# Shared helper classes - built as STATIC library
add_library(usearch_utils STATIC ${UTILS_SOURCES})
add_library(hnswlib_utils STATIC ${HNSWLIB_UTILS_SOURCES})

target_link_libraries(usearch_utils duckdb)
target_link_libraries(hnswlib_utils duckdb)

# Add subdirectories for each experiment type
add_subdirectory(usearch)
add_subdirectory(hnswlib)
add_subdirectory(repl_cand_hnswlib)
add_subdirectory(reset_first_candidate)
add_subdirectory(MN_RBC)