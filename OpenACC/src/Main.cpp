#include "Accelerator.h"
#include <iostream>

int main()
{
    std::cout << "====================" << std::endl;
    std::cout << "   OpenACC Sample   " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    HpcSamples::Accelerator accelerator(1024);
    std::cout << std::endl;

    for (int i = 0; i < 10; i++)
    {
        std::cout << "====================" << std::endl;
        std::cout << "Count: " << i << std::endl;
        accelerator.RunOnAccelerator();
        accelerator.RunOnCpu();
        std::cout << "====================" << std::endl;
    }
}