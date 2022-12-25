#include <iostream>
#include "Accelerator.h"

int main()
{
    std::cout << "====================" << std::endl;
    std::cout << "     CUDA Sample    " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    int n = 1024;

    CudaSamples::Accelerator accelerator(n);
    std::cout << std::endl;

    accelerator.Initialize();

    for (int i = 0; i < 3; i++)
    {
        std::cout << "====================" << std::endl;
        std::cout << "Count: " << i << std::endl;
        accelerator.RunOnAccelerator();
        // accelerator.RunOnCpu();
        std::cout << "====================" << std::endl;
    }
}