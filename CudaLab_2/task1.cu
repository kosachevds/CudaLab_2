#include "common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

__global__ void task1_Coalescing(unsigned const* a, unsigned const* b, unsigned* result, size_t size)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    result[index] = a[index] * b[index];
}

__global__ void task1_NoCoalescing(unsigned const* a, unsigned const* b, unsigned* result, size_t size)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x + 7;
    if (index > size + 6) {
        return;
    }
    if (index >= size) {
        index -= 7;
    }
    result[index] = a[index] * b[index];
}

void Task1()
{
    const auto COUNT = 50u;
    const auto SIZE = 10 * 1024u;
    //const auto SIZE = 100 * 1024u;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<unsigned> host_a(COUNT * SIZE), host_b(COUNT * SIZE), host_c(COUNT * SIZE);
    unsigned *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, COUNT * SIZE * sizeof(unsigned));
    cudaMalloc(&dev_b, COUNT * SIZE * sizeof(unsigned));
    cudaMalloc(&dev_c, COUNT * SIZE * sizeof(unsigned));

    std::vector<float> times_coalescing(COUNT), times_no_coalescing(COUNT);
    for (auto i = 1u; i <= COUNT; ++i) {
        system("cls");
        std::cout << i << " : " << COUNT << std::endl;
        auto current_size = i * SIZE;
        for (auto j = 0u; j < current_size; ++j) {
            host_a[j] = rand() % current_size + 1;
            host_b[j] = rand() % current_size + 1;
        }
        cudaMemcpy(dev_a, host_a.data(), sizeof(unsigned) * current_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, host_b.data(), sizeof(unsigned) * current_size, cudaMemcpyHostToDevice);
        unsigned block_size;
        if (current_size <= MAX_BLOCK_SIZE) {
            block_size = current_size;
        } else {
            block_size = MAX_BLOCK_SIZE;
        }
        unsigned block_count = ceil(static_cast<double>(current_size) / MAX_BLOCK_SIZE);
        cudaEventRecord(start);
        task1_Coalescing<<<block_count, block_size>>>(dev_a, dev_b, dev_c, current_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_coalescing[i - 1], start, stop);
        cudaMemcpy(host_c.data(), dev_c, current_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (!GoodMiltiplication(host_a, host_b, host_c)) {
            times_coalescing[i - 1] = -1;
        }

        cudaEventRecord(start);
        task1_NoCoalescing<<<block_count, block_size>>>(dev_a, dev_b, dev_c, current_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_no_coalescing[i - 1], start, stop);
        cudaMemcpy(host_c.data(), dev_c, current_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (!GoodMiltiplication(host_a, host_b, host_c)) {
            times_no_coalescing[i - 1] = -1;
        }
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto out_file = std::ofstream("times.txt");
    WriteVector(times_coalescing, out_file);
    out_file << ";\n";
    WriteVector(times_no_coalescing, out_file);
    out_file.close();
}
