cmake_minimum_required(VERSION 2.8.12...3.29)
project(example-c++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -ffast-math)
# add_compile_options(-g) # for debug -- SLOW

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

# usearch executables
add_executable(usearch_random usearch/random.cpp)
add_executable(usearch_full_coverage usearch/full_coverage.cpp)
add_executable(usearch_new_data usearch/new_data.cpp)
add_executable(usearch_unreachable_points_sampl_all usearch/unreachable_points_sampl_all.cpp)
add_executable(usearch_unreachable_points_sampl_reachable usearch/unreachable_points_sampl_reachable.cpp)
add_executable(usearch_create_indexes usearch/create_index.cpp)

#hnswlib executables
add_executable(hnswlib_random hnswlib/random.cpp)
add_executable(hnswlib_full_coverage hnswlib/full_coverage.cpp)
add_executable(hnswlib_new_data hnswlib/new_data.cpp)
add_executable(hnswlib_unreachable_points_sampl_all hnswlib/unreachable_points_sampl_all.cpp)
add_executable(hnswlib_unreachable_points_sampl_reachable hnswlib/unreachable_points_sampl_reachable.cpp)
add_executable(hnswlib_create_indexes hnswlib/create_index.cpp)

# repl_cand_hnswlib executables
add_executable(repl_cand_hnswlib_random repl_cand_hnswlib/random.cpp)
add_executable(repl_cand_hnswlib_new_data repl_cand_hnswlib/new_data.cpp)
add_executable(repl_cand_hnswlib_full_coverage repl_cand_hnswlib/full_coverage.cpp)

# usearch link libraries
target_link_libraries(usearch_random usearch_utils duckdb)
target_link_libraries(usearch_full_coverage usearch_utils duckdb)
target_link_libraries(usearch_new_data usearch_utils duckdb)
target_link_libraries(usearch_unreachable_points_sampl_all usearch_utils duckdb)
target_link_libraries(usearch_unreachable_points_sampl_reachable usearch_utils duckdb)
target_link_libraries(usearch_create_indexes usearch_utils duckdb)

# hnswlib link libraries
target_link_libraries(hnswlib_random hnswlib_utils usearch_utils duckdb)
target_link_libraries(hnswlib_full_coverage hnswlib_utils usearch_utils duckdb)
target_link_libraries(hnswlib_new_data hnswlib_utils usearch_utils duckdb)
target_link_libraries(hnswlib_unreachable_points_sampl_all hnswlib_utils usearch_utils duckdb)
target_link_libraries(hnswlib_unreachable_points_sampl_reachable hnswlib_utils usearch_utils duckdb)
target_link_libraries(hnswlib_create_indexes hnswlib_utils usearch_utils duckdb)

# repl_cand_hnswlib link libraries
target_link_libraries(repl_cand_hnswlib_random hnswlib_utils usearch_utils duckdb)
target_link_libraries(repl_cand_hnswlib_full_coverage hnswlib_utils usearch_utils duckdb)
target_link_libraries(repl_cand_hnswlib_new_data hnswlib_utils usearch_utils duckdb)