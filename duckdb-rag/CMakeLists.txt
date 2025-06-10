cmake_minimum_required(VERSION 3.16)
project(duckdb_rag_onnx)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime path (in the current directory)
set(ONNXRUNTIME_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

# Include directories
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

# Link directories  
if(APPLE)
    link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
elseif(UNIX)
    link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
endif()

# Create executable
add_executable(onnx_text onnx_text.cpp)

# Link ONNX Runtime
if(APPLE)
    target_link_libraries(onnx_text onnxruntime)
elseif(UNIX)
    target_link_libraries(onnx_text onnxruntime)
endif()