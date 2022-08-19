cmake_minimum_required(VERSION 3.23)

# Set the project name and version
project(OpenMpSampleProject LANGUAGES CXX)
set(PROJECT_VERSION 0.1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set source files
set(SOURCE
    src/Multiprocessor.h
    src/Multiprocessor.cpp
    src/Main.cpp
)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Targets
add_executable(OpenMpSample ${SOURCE})