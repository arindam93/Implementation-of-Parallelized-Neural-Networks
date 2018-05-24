cmake_minimum_required(VERSION 3.9.6)
project(APT_Project)

set(CMAKE_CXX_STANDARD 11)

include_directories(src)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(APT_Project
        ../src/neural_network.cpp
        ../src/test.cpp
        ../src/mnist_helper.cpp
        ../src/convolutional_neural_network.cpp
        )
