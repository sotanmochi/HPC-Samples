#include "Multiprocessor.h"
#include <iostream>

int main()
{
    std::cout << "====================" << std::endl;
    std::cout << "    OpenMP Sample   " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;

    HpcSamples::Multiprocessor multiprocessor(1024 + 512);
    std::cout << std::endl;

    for (int i = 0; i < 3; i++)
    {
        std::cout << "====================" << std::endl;
        std::cout << "Count: " << i << std::endl;
        multiprocessor.RunOnMultiprocessor();
        multiprocessor.RunOnCpu();
        std::cout << "====================" << std::endl;
    }
}