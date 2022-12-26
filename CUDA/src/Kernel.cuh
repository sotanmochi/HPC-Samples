#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

namespace CudaSamples
{
void CalculateMachineEpsilon();
void MallocCudaMemory(void **pointer, size_t size);
void FreeCudaMemory(void *pointer);
void CopyCudaMemoryHostToDevice(void *dst, void *src, size_t nbytes);
void CopyCudaMemoryDeviceToHost(void *dst, void *src, size_t nbytes);
void NaiveMatrixMultiply(int n, float *deviceMatrixA, float *deviceMatrixB, float *deviceMatrixC);
}

#endif // _KERNEL_CUH_