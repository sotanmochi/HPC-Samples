#include <iostream>
#include "Kernel.cuh"

#define BLOCK_SIZE 32

namespace CudaSamples
{

__global__ void CalculateMachineEpsilonKernel()
{
	// printf("CalculateMachineEpsilonKernel\n");

	float feps = 1.0f;
	for (float ftmp = feps + 1.0f; ftmp > 1; ftmp = feps + 1.0f)
	{
		feps /= 2.0f;
	}

	printf("Machine epsilon CUDA: %-16g\n", feps * 2.0f);
}

__global__ void NaiveMatrixMultiplyKernel(int n, float *matrixA, float *matrixB, float *matrixC)
{
	unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int a_index = row * n;
	unsigned int b_index = col;
	unsigned int c_index = row * n + col;

	float sum = 0.0f;

	if (row >= n || col >= n)
	{
		// printf("Skiped for (%d, %d)\n", row, col);
		return;
	}

	for (int i = 0; i < n; i++)
	{
		sum += matrixA[a_index] * matrixB[b_index];
		a_index += 1;
		b_index += n;
	}

	matrixC[c_index] = sum;
}

void CalculateMachineEpsilon()
{
	CalculateMachineEpsilonKernel<<<1,1>>>();
	cudaDeviceSynchronize();
}

void MallocCudaMemory(void **pointer, size_t size)
{
	cudaError_t err = cudaMalloc(pointer, size);
    std::cout << "CudaError@MallocCudaMemory: " << err << std::endl;
}

void FreeCudaMemory(void *pointer)
{
	cudaError_t err = cudaFree(pointer);
    std::cout << "CudaError@FreeCudaMemory: " << err << std::endl;
}

void CopyCudaMemoryHostToDevice(void *dst, void *src, size_t nbytes)
{
	cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    std::cout << "CudaError@CopyCudaMemoryHostToDevice: " << err << std::endl;
}

void CopyCudaMemoryDeviceToHost(void *dst, void *src, size_t nbytes)
{
	cudaError_t err = cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
    std::cout << "CudaError@CopyCudaMemoryDeviceToHost: " << err << std::endl;
}

void NaiveMatrixMultiply(int n, float *deviceMatrixA, float *deviceMatrixB, float *deviceMatrixC)
{
	int blockWidth = BLOCK_SIZE;
	int blockHeight = BLOCK_SIZE;
	int gridWidth  = ceil((float)n/blockWidth);
	int gridHeight = ceil((float)n/blockHeight);

	dim3 blockSize(blockWidth, blockHeight, 1);
	dim3 gridSize(gridWidth, gridHeight, 1);

    NaiveMatrixMultiplyKernel<<<gridSize, blockSize>>>(n, deviceMatrixA, deviceMatrixB, deviceMatrixC);

	cudaError_t err = cudaGetLastError();
    std::cout << "CudaError@NaiveMatrixMultiply: " << err << std::endl;
}

}
