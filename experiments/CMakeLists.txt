cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

include_directories(../duckdb/src/include include)
# link_directories(../build/debug/src) # for debug
link_directories(../build/release/src)




add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

add_library(usearch INTERFACE)
add_library(usearch::usearch ALIAS usearch)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(usearch INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/usearch
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS usearch
    EXPORT usearchTargets)

install(TARGETS hnswlib
    EXPORT hnswlibTargets)


install(EXPORT usearchTargets
    FILE usearchConfig.cmake
    NAMESPACE usearch::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/usearch)


install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

add_executable(hnswlib_test hnswlib_test.cpp)
add_executable(usearch_test usearch_test.cpp)

target_link_libraries(hnswlib_test duckdb hnswlib)    
target_link_libraries(usearch_test duckdb usearch)
