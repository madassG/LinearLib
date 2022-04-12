cmake_minimum_required(VERSION 3.21)
project(Linear)

set(CMAKE_CXX_STANDARD 17)

add_executable(Linear main.cpp linear.cpp linear.h exception.cpp exception.h operator.cpp operator.h test_runner.h linear_test.h PCA.cpp PCA.h)
