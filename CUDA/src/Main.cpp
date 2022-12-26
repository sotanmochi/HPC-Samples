#include <cfloat>
#include <iostream>
#include "Accelerator.h"
#include "Kernel.cuh"

int main()
{
    std::cout << "====================" << std::endl;
    std::cout << "     CUDA Sample    " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    CudaSamples::CalculateMachineEpsilon();
    std::cout << "FLT_EPSILON: " << FLT_EPSILON << std::endl;
    std::cout << std::endl;

    int n = 128;
    CudaSamples::Accelerator accelerator(n);
    accelerator.Initialize();

    std::cout << std::endl;
    for (int i = 0; i < 3; i++)
    {
        std::cout << "====================" << std::endl;
        std::cout << "Count: " << i << std::endl;
        accelerator.RunOnAccelerator();
        accelerator.RunOnCpu();
        std::cout << "====================" << std::endl;
        std::cout << std::endl;
    }
}