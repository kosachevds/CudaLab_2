#include "common.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <chrono>

static const auto BLOCK_GLOBAL = 32u; // sqrt(MAX_BLOCK_SIZE);
static const auto BLOCK_SHARED = 16u;

static float multiplyOnCpu(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
static float multiplyGpuGlobal(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
static float multiplyGpuShared(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
static void copyMatrixToGpu(unsigned const* const* matrix, size_t size, unsigned* out_array);
static void copyMatrixFromGpu(unsigned const* gpu_array, size_t one_dim_size, unsigned*const* matrix);
static unsigned** createMatrix(size_t size);
static unsigned** createRandomMatrix(size_t size);
static void deleteMatrix(unsigned** matrix, size_t size);
static void printMatrix(unsigned const* const* matrix, size_t size);
static bool equalsMatrices(unsigned const* const* m1, unsigned const* const* m2, size_t size);

static __global__ void multiplyGlobal(unsigned const* left, unsigned const* right, unsigned* result, size_t size);

static __global__ void multiplyShared(unsigned const* left, unsigned const* right, unsigned* result, int size);

void Task3()
{
    const auto MIN_SIZE = 8;
    const auto STEP = MIN_SIZE;
    const auto MAX_SIZE = 32 * MIN_SIZE;

    std::vector<size_t> sizes;
    std::vector<float> times_cpu, times_gpu_g, times_gpu_s;
    for (auto size = MIN_SIZE; size <= MAX_SIZE; size += STEP) {
        sizes.push_back(size);
        system("cls");
        std::cout << size << ": " << MAX_SIZE << std::endl;
        auto left = createRandomMatrix(size);
        auto right = createRandomMatrix(size);
        auto result_cpu = createMatrix(size);
        auto result_gpu = createMatrix(size);
        auto ms = multiplyOnCpu(left, right, result_cpu, size);
        times_cpu.push_back(ms);
        ms = multiplyGpuGlobal(left, right, result_gpu, size);
        if (!equalsMatrices(result_cpu, result_gpu, size)) {
             ms = -1;
        }
        times_gpu_g.push_back(ms);
        ms = multiplyGpuShared(left, right, result_gpu, size);
        if (!equalsMatrices(result_cpu, result_gpu, size)) {
             ms = -1;
        }
        times_gpu_s.push_back(ms);
        deleteMatrix(left, size);
        deleteMatrix(right, size);
        deleteMatrix(result_cpu, size);
        deleteMatrix(result_gpu, size);
    }
    std::ofstream out("times3.txt");
    WriteVector(sizes, out);
    out << ";\n";
    WriteVector(times_cpu, out);
    out << ";\n";
    WriteVector(times_gpu_g, out);
    out << ";\n";
    WriteVector(times_gpu_s, out);
}

float multiplyOnCpu(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (auto i = 0u; i < size; ++i) {
        for (auto j = 0; j < size; ++j) {
            result[i][j] = 0u;
            for (auto k = 0; k < size; ++k) {
                result[i][j] += left[i][k] * right[k][j];
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0F;
}

float multiplyGpuGlobal(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size)
{
    const auto SIZE = size * size;
    unsigned *gpu_left, *gpu_right, *gpu_result;
    cudaMalloc(&gpu_left, SIZE * sizeof(unsigned));
    copyMatrixToGpu(left, size, gpu_left);
    cudaMalloc(&gpu_right, SIZE * sizeof(unsigned));
    copyMatrixToGpu(right, size, gpu_right);
    cudaMalloc(&gpu_result, SIZE * sizeof(unsigned));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_size = BLOCK_GLOBAL;
    unsigned block_count = ceil(size / static_cast<double>(block_size));
    cudaEventRecord(start);
    multiplyGlobal<<<dim3(block_count, block_count), dim3(block_size, block_size)>>>
        (gpu_left, gpu_right, gpu_result, size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    copyMatrixFromGpu(gpu_result, size, result);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(gpu_left);
    cudaFree(gpu_right);
    cudaFree(gpu_result);
    return ms;
}

float multiplyGpuShared(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size)
{
    const auto SIZE = size * size;
    unsigned *gpu_left, *gpu_right, *gpu_result;
    cudaMalloc(&gpu_left, SIZE * sizeof(unsigned));
    copyMatrixToGpu(left, size, gpu_left);
    cudaMalloc(&gpu_right, SIZE * sizeof(unsigned));
    copyMatrixToGpu(right, size, gpu_right);
    cudaMalloc(&gpu_result, SIZE * sizeof(unsigned));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_size = BLOCK_SHARED;
    unsigned block_count = ceil(size / static_cast<double>(block_size));
    cudaEventRecord(start);
    multiplyShared<<<dim3(block_count, block_count), dim3(block_size, block_size)>>>
        (gpu_left, gpu_right, gpu_result, size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    copyMatrixFromGpu(gpu_result, size, result);

    cudaFree(gpu_left);
    cudaFree(gpu_right);
    cudaFree(gpu_result);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return ms;
}

void copyMatrixToGpu(unsigned const* const* matrix, size_t size, unsigned* out_array)
{
    std::vector<unsigned> buffer;
    buffer.reserve(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            buffer.push_back(matrix[i][j]);
        }
    }
    cudaMemcpy(out_array, buffer.data(), buffer.size() * sizeof(unsigned), cudaMemcpyHostToDevice);
}

void copyMatrixFromGpu(unsigned const* gpu_array, size_t one_dim_size, unsigned*const* matrix)
{
    std::vector<unsigned> buffer(one_dim_size * one_dim_size);
    cudaMemcpy(buffer.data(), gpu_array, buffer.size() * sizeof(unsigned), cudaMemcpyDeviceToHost);
    auto index = 0;
    for (int i = 0; i < one_dim_size; ++i) {
        for (int j = 0; j < one_dim_size; ++j) {
            matrix[i][j] = buffer[index];
            ++index;
        }
    }
}

unsigned** createMatrix(size_t size)
{
    auto matrix = new unsigned*[size];
    for (auto i = 0; i < size; ++i) {
        matrix[i] = new unsigned[size];
    }
    return matrix;
}

unsigned** createRandomMatrix(size_t size)
{
    auto matrix = createMatrix(size);
    for (auto i = 0; i < size; ++i) {
        for (auto j = 0; j < size; ++j) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
    return matrix;
}

void deleteMatrix(unsigned** matrix, size_t size)
{
    for (auto i = 0; i < size; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void printMatrix(unsigned const* const* matrix, size_t size)
{
    for (auto i = 0; i < size; ++i) {
        for (auto j = 0; j < size; ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << std::endl;
    }
}

bool equalsMatrices(unsigned const* const* m1, unsigned const* const* m2, size_t size)
{
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (m1[i][j] != m2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

__global__ void multiplyGlobal(unsigned const* left, unsigned const* right, unsigned* result, size_t size)
{
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        auto sum = 0u;
        for (int k = 0; k < size; k++) {
            sum += left[row * size + k] * right[k * size + col];
        }
        result[row * size + col] = sum;
    }
}

__global__ void multiplyShared(unsigned const* left, unsigned const* right, unsigned* result, int size)
{
    __shared__ unsigned as [BLOCK_SHARED][BLOCK_SHARED];
    __shared__ unsigned bs [BLOCK_SHARED][BLOCK_SHARED];
    int tx = threadIdx.x, ty = threadIdx.y;
    auto row = blockIdx.y * BLOCK_SHARED + ty;
    auto col = blockIdx.x * BLOCK_SHARED + tx;
    auto sum = 0u;
    for (int m = 0; m < (size - 1) / BLOCK_SHARED + 1; ++m) {
        if (row < size && m * BLOCK_SHARED + tx < size) {
            as[ty][tx] = left[row * size + m * BLOCK_SHARED + tx];
        } else {
            as[ty][tx] = 0;
        }
        if (col < size && m * BLOCK_SHARED + ty < size) {
            bs[ty][tx] = right[(m * BLOCK_SHARED + ty) * size + col];
        } else {
            bs[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SHARED; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }
        __syncthreads();
    }
    if (row < size && col < size) {
        result[row * size + col] = sum;
    }
}
