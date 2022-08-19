cmake_minimum_required(VERSION 3.23)

# Set the project name and version
project(OpenAccSampleProject LANGUAGES CXX)
set(PROJECT_VERSION 0.1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set source files
set(SOURCE
    src/Accelerator.h
    src/Accelerator.cpp
    src/Main.cpp
)

# OpenACC
find_package(OpenACC REQUIRED)
if(OpenACC_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc -Minfo=acc")
endif()

# Targets
add_executable(OpenAccSample ${SOURCE})