#include "common.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <ctime>
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
    if (index > size) {
        index -= 7;
    }
    result[index] = a[index] * b[index];
}

void task1()
{
    const auto COUNT = 100u;
    const auto SIZE = 100000u;
    srand(time(nullptr));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<unsigned> host_a(COUNT * SIZE), host_b(COUNT * SIZE), host_c(COUNT * SIZE);
    unsigned *dev_a, *dev_b, *dev_c;
    cudaMalloc(reinterpret_cast<void**>(&dev_a), COUNT * SIZE * sizeof(unsigned));
    cudaMalloc(reinterpret_cast<void**>(&dev_b), COUNT * SIZE * sizeof(unsigned));
    cudaMalloc(reinterpret_cast<void**>(&dev_c), COUNT * SIZE * sizeof(unsigned));
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
        unsigned block_count, block_size;
        //block_size = 16;
        //block_count = current_size / block_size + 1;
        if (current_size <= MAX_BLOCK_SIZE) {
            block_size = current_size;
            block_count = 1;
        } else {
            block_size = MAX_BLOCK_SIZE;
            block_count = current_size / MAX_BLOCK_SIZE + 1;
        }

        cudaEventRecord(start);
        task1_Coalescing<<<block_count, block_size>>>(dev_a, dev_b, dev_c, current_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_coalescing[i - 1], start, stop);
        cudaMemcpy(host_c.data(), dev_c, current_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (!goodMiltiplication(host_a.data(), host_b.data(), host_c.data(), current_size)) {
            times_coalescing[i - 1] = -1;
        }

        cudaEventRecord(start);
        task1_NoCoalescing<<<block_count, block_size>>>(dev_a, dev_b, dev_c, current_size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times_no_coalescing[i - 1], start, stop);
        cudaMemcpy(host_c.data(), dev_c, current_size * sizeof(unsigned), cudaMemcpyDeviceToHost);
        if (!goodMiltiplication(host_a.data(), host_b.data(), host_c.data(), current_size)) {
            times_no_coalescing[i - 1] = -1;
        }
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto out_file = std::ofstream("times.txt");
    writeVector(times_coalescing, out_file);
    out_file << ";\n";
    writeVector(times_no_coalescing, out_file);
    out_file.close();
}
