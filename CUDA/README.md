# CUDA Code Samples

## Tested Environment
- Windows 11 Pro (Version: 21H2, OS Build: 22000.856)
- Ubuntu 20.04.4 LTS (GNU/Linux 5.10.102.1-microsoft-standard-WSL2 x86_64)
- NVIDIA HPC SDK 22.7
- CUDA Toolkit 11.7
- CMake 3.23.3
- Build tool: Ninja 1.11.0
- Compiler: nvc++ 22.7-0 64-bit target (included in NVIDIA HPC SDK)

## Build and Run
```
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_BUILD_TYPE=Release -GNinja ..
$ ninja
```
```
$ ./CudaSample
```
