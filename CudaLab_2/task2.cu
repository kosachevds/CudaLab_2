#include "common.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <ctime>

__global__ void task2Kernel(unsigned const* a, unsigned const* b, unsigned* result, size_t size)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    result[index] = a[index] * b[index];
}

void task2()
{
    const auto SIZE = 8 * 1024 * 1024;
    const auto STREAM_COUNT = 8;
    srand(time(nullptr));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // TODO: matrix[3][SIZE];
    std::vector<unsigned> host_a(SIZE), host_b(SIZE), host_c(SIZE);
    unsigned *dev_a, *dev_b, *dev_c;
    cudaMalloc(reinterpret_cast<void**>(&dev_a), SIZE * sizeof(unsigned));
    cudaMalloc(reinterpret_cast<void**>(&dev_b), SIZE * sizeof(unsigned));
    cudaMalloc(reinterpret_cast<void**>(&dev_c), SIZE * sizeof(unsigned));
    std::vector<float> times;
    cudaStream_t streams[STREAM_COUNT];
    for (auto count = 1; count <= STREAM_COUNT; ++count) {
        for (auto i = 0; i < count; ++i) {
            cudaStreamCreate(&streams[i]);
        }
        for (auto i = 0; i < SIZE; ++i) {
            host_a[i] = rand() % 1024 + 1;
            host_b[i] = rand() % 1024 + 1;
        }
        auto chunk_size = SIZE / count;
        //cudaEventRecord(start);
        for (auto i = 0; i < count; ++i) {
            auto begin = i * chunk_size;
            int size;
            if (i != count - 1) {
                size = SIZE - begin;
            } else {
                size = chunk_size;
            }
            auto bytes = size * sizeof(unsigned);
            cudaEventRecord(start, streams[i]);
            cudaMemcpyAsync(dev_a + begin, host_a.data() + begin, bytes, cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(dev_b + begin, host_b.data() + begin, bytes, cudaMemcpyHostToDevice, streams[i]);
            task2Kernel<<<SIZE / MAX_BLOCK_SIZE, MAX_BLOCK_SIZE, 0, streams[i]>>>(
                dev_a + begin, dev_b + begin, dev_c + begin, size);
            cudaMemcpyAsync(host_c.data() + begin, dev_c + begin, bytes, cudaMemcpyDeviceToHost, streams[i]);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, streams[i]);
        //cudaEventSynchronize(stop);
        }
        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        for (auto i = 0; i < count; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::ofstream out("times.txt");
    writeVector(times, out);
    out.close();
}