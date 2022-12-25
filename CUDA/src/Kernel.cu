#include <iostream>
#include "Kernel.cuh"

#define BLOCK_SIZE 32

namespace CudaSamples
{

__global__ void NaiveMatrixMultiplyKernel(float *matrixA, float *matrixB, float *matrixC)
{
	unsigned int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int indexY = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned int width = gridDim.x * blockDim.x;
	unsigned int height = gridDim.y * blockDim.y;

	// ToDo

	matrixC[indexY * width + indexX] = 1;
}

void MallocCudaMemory(void **pointer, size_t size)
{
	cudaMalloc(pointer, size);
}

void FreeCudaMemory(void *pointer)
{
	cudaFree(pointer);
}

void CopyCudaMemoryHostToDevice(void *dst, void *src, size_t nbytes)
{
	cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
}

void CopyCudaMemoryDeviceToHost(void *dst, void *src, size_t nbytes)
{
	cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost);
}

void NaiveMatrixMultiply(int n, float *deviceMatrixA, float *deviceMatrixB, float *deviceMatrixC)
{
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 gridSize(n/BLOCK_SIZE, n/BLOCK_SIZE, 1);
    NaiveMatrixMultiplyKernel<<<gridSize, blockSize>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC);
}

}
