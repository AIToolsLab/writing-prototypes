cmake_minimum_required(VERSION 3.16)
project(writing-engine CXX)
set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# llama.cpp, pinned. To use an existing checkout: -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=/path/to/llama.cpp
include(FetchContent)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)   # chat templates
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG f7b886f)
FetchContent_MakeAvailable(llama_cpp)

add_executable(writing-engine server.cpp)
target_link_libraries(writing-engine PRIVATE llama llama-common)
target_include_directories(writing-engine PRIVATE ${llama_cpp_SOURCE_DIR}/vendor)
