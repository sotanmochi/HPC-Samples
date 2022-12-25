#include <cfloat>
#include <chrono>
#include <iostream>

#include "Accelerator.h"
#include "Kernel.cuh"

CudaSamples::Accelerator::Accelerator(uint16_t n)
{
    std::cout << "Matrix size: " << n << "x" << n << std::endl;

    _n = n;
    _size = n*n;

    _matrixA = new float[_size];
    _matrixB = new float[_size];
    _matrixC = new float[_size];

	MallocCudaMemory((void **)&_deviceMatrixA, _size);
	MallocCudaMemory((void **)&_deviceMatrixB, _size);
	MallocCudaMemory((void **)&_deviceMatrixC, _size);
}

CudaSamples::Accelerator::~Accelerator()
{
    delete[] _matrixA;
    delete[] _matrixB;
    delete[] _matrixC;

    FreeCudaMemory(_deviceMatrixA);
    FreeCudaMemory(_deviceMatrixB);
    FreeCudaMemory(_deviceMatrixC);
}

void CudaSamples::Accelerator::Initialize()
{
    for (int y = 0; y < _n; y++) // Row
    {
        for (int x = 0; x < _n; x++) // Column
        {
            _matrixA[x + _n * y] = 1.0f;
            _matrixB[x + _n * y] = 0.1f;
        }
    }
}

bool CudaSamples::Accelerator::CheckResult()
{
    float expected = 0.1f * _n;

    for (int j = 0; j < _n; j++) // Row
    {
        for (int i = 0; i < _n; i++) // Column
        {
            float diff = abs(expected - _matrixC[i + _n * j]);
            if (diff > FLT_EPSILON) return false;
        }
    }

    return true;
}

void CudaSamples::Accelerator::RunOnAccelerator()
{
    std::cout << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "Matrix Multiplication on Accelerator (using CUDA)" << std::endl;

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    CopyCudaMemoryHostToDevice(_deviceMatrixA, _matrixA, _size);
	CopyCudaMemoryHostToDevice(_deviceMatrixB, _matrixB, _size);

    NaiveMatrixMultiply(_n, _deviceMatrixA, _deviceMatrixB, _deviceMatrixC);

	CopyCudaMemoryDeviceToHost(_matrixC, _deviceMatrixC, _size);

    end = std::chrono::system_clock::now();
    double elapsedTimeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
    std::cout << "CheckResult: " << (CheckResult() ? "OK" : "NG") << std::endl;
    std::cout << "----------" << std::endl;
}

void CudaSamples::Accelerator::RunOnCpu()
{
    std::cout << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "Matrix Multiplication on CPU (Single Thread)" << std::endl;

    std::chrono::system_clock::time_point start, end;

    int N = _n;
    start = std::chrono::system_clock::now();

    // Matrix Multiplication
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float cij = 0.0f;
            for (int k = 0; k < N; k++)
            {
                cij += _matrixA[i + N * k] * _matrixB[k + N * j];
            }
            _matrixC[i + N * j] = cij;
        }
    }

    end = std::chrono::system_clock::now();
    double elapsedTimeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
    std::cout << "CheckResult: " << (CheckResult() ? "OK" : "NG") << std::endl;
    std::cout << "----------" << std::endl;
}
