#include <cfloat>
#include <chrono>
#include <cmath>
#include <iostream>

#include "Accelerator.h"
#include "Kernel.cuh"

CudaSamples::Accelerator::Accelerator(uint16_t n)
{
    std::cout << "Matrix size: " << n << "x" << n << std::endl;

    _n = n;
    _matrixSize = n*n;

    _matrixA = new float[_matrixSize];
    _matrixB = new float[_matrixSize];
    _matrixC = new float[_matrixSize];

    _matrixMemorySize = _matrixSize * sizeof(float);

	MallocCudaMemory((void **)&_deviceMatrixA, _matrixMemorySize);
	MallocCudaMemory((void **)&_deviceMatrixB, _matrixMemorySize);
	MallocCudaMemory((void **)&_deviceMatrixC, _matrixMemorySize);
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

float CudaSamples::Accelerator::CheckResult()
{
    float diffSum = 0.0f;
    float expected = 0.1f * _n;

    for (int j = 0; j < _n; j++) // Row
    {
        for (int i = 0; i < _n; i++) // Column
        {
            float value = _matrixC[i + _n * j];

            //
            // Compares two floating point values if they are similar.
            // References:
            //   - https://github.com/Unity-Technologies/UnityCsReference/blob/2022.2/Runtime/Export/Math/Mathf.cs#L280
            //
            float diff = fabsf(expected - value);
            bool approximately = diff < fmax(0.000001f * fmax(fabsf(expected), fabsf(value)), FLT_EPSILON * 8);
            if (!approximately)
            {
                diffSum += diff;
                // printf("----------\n");
                // printf("Expected[%d][%d]: %-16g\n", i, j, expected);
                // printf("Actual[%d][%d]: %-16g\n", i, j, value);
                // printf("Diff[%d][%d]: %-16g\n", i, j, fabsf(expected - value));
            }
        }
    }

    return diffSum;
}

void CudaSamples::Accelerator::RunOnAccelerator()
{
    std::cout << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "Matrix Multiplication on Accelerator (using CUDA)" << std::endl;
    std::cout << "----------" << std::endl;

    std::chrono::system_clock::time_point start, end;
    start = std::chrono::system_clock::now();

    CopyCudaMemoryHostToDevice(_deviceMatrixA, _matrixA, _matrixMemorySize);
	CopyCudaMemoryHostToDevice(_deviceMatrixB, _matrixB, _matrixMemorySize);

    NaiveMatrixMultiply(_n, _deviceMatrixA, _deviceMatrixB, _deviceMatrixC);

	CopyCudaMemoryDeviceToHost(_matrixC, _deviceMatrixC, _matrixMemorySize);

    end = std::chrono::system_clock::now();
    double elapsedTimeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    float diffSum = CheckResult();

    std::cout << "----------" << std::endl;
    std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
    std::cout << "CheckResult: " << (diffSum ? "NG" : "OK") << std::endl;
    std::cout << "DiffSum: " << diffSum << std::endl;
    std::cout << "----------" << std::endl;
}

void CudaSamples::Accelerator::RunOnCpu()
{
    std::cout << std::endl;
    std::cout << "----------" << std::endl;
    std::cout << "Matrix Multiplication on CPU (Single Thread)" << std::endl;
    std::cout << "----------" << std::endl;

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

    float diffSum = CheckResult();

    std::cout << "----------" << std::endl;
    std::cout << "Elapsed time: " << elapsedTimeMilliseconds << " [ms]" << std::endl;
    std::cout << "CheckResult: " << (diffSum ? "NG" : "OK") << std::endl;
    std::cout << "DiffSum: " << diffSum << std::endl;
    std::cout << "----------" << std::endl;
}
