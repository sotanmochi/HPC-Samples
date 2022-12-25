cmake_minimum_required(VERSION 3.23)

# Set the project name and version
project(CudaSample LANGUAGES CXX)
set(PROJECT_VERSION 0.1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set source files
file(GLOB HEADERS src/*.h)
file(GLOB CXX_SOURCES src/*.cpp)
file(GLOB CUDA_SOURCES src/*.cu)
set(SOURCES ${HEADERS} ${CXX_SOURCES} ${CUDA_SOURCES})

# CUDA
# set(CUDAToolkit_LIBRARIES "cufft curand")
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Targets
add_executable(${PROJECT_NAME} ${SOURCES})

# Logs
message(STATUS "CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
# message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
# message(STATUS "SOURCES: ${SOURCES}")