#pragma once

#include <cstdint>

namespace CudaSamples
{
    class Accelerator
    {
        public:
            Accelerator(uint16_t n);
            ~Accelerator();

            void Initialize();
            float CheckResult();
            void RunOnAccelerator();
            void RunOnCpu();

        private:
            uint16_t _n;
            uint32_t _matrixSize;
            size_t _matrixMemorySize;
            float *_matrixA;
            float *_matrixB;
            float *_matrixC;
            float *_deviceMatrixA;
            float *_deviceMatrixB;
            float *_deviceMatrixC;
    };
}