#include "common.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <chrono>

const auto MAX_SQUARE_BLOCK_WIDTH = 32.0; // sqrt(MAX_BLOCK_SIZE);

float multiplyOnCpu(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
float multiplyGpuGlobal(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
float multiplyGpuShared(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size);
void copyMatrixToGpu(unsigned const* const* matrix, size_t size, unsigned* out_array);
void copyMatrixFromGpu(unsigned const* gpu_array, size_t one_dim_size, unsigned*const* matrix);
unsigned** createMatrix(size_t size);
unsigned** createRandomMatrix(size_t size);
void deleteMatrix(unsigned** matrix, size_t size);
void printMatrix(unsigned const* const* matrix, size_t size);
bool equalsMatrices(unsigned const* const* m1, unsigned const* const* m2, size_t size);

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
        //result[row * size + col] += left[row * size + col] + right[col * size + col];
    }
}

__global__ void multiplyShared(unsigned const* a, unsigned const* b, unsigned* c, int size)
{
    const auto BLOCK_SIZE = 16;
    int bx = blockIdx.x; // индексы блока
    int by = blockIdx.y; //
    int tx = threadIdx.x; // индексы нити внутри блока
    int ty = threadIdx.y; //
    int aBegin = size * BLOCK_SIZE * by;
    int aEnd = aBegin + size - 1;
    int aStep = BLOCK_SIZE;
    int bBegin = bx * BLOCK_SIZE;
    int bStep = BLOCK_SIZE * size;
    float sum = 0.0f;
    for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
        __shared__ unsigned as [BLOCK_SIZE][BLOCK_SIZE];
        __shared__ unsigned bs [BLOCK_SIZE][BLOCK_SIZE];
        as[ty][tx] = a[ia + size * ty + tx];
        bs[ty][tx] = b[ib + size * ty + tx];
        __syncthreads(); // Убедимся, что подматрицы полностью загружены
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as[ty][k] * bs[k][tx];
        __syncthreads(); // Убедимся, что подматрицы никому больше не нужны
    }
    c[size * BLOCK_SIZE * by + BLOCK_SIZE * bx + size * ty + tx] = sum;
}


void task3()
{
    const auto MIN_SIZE = 8;
    const auto STEP = MIN_SIZE;
    const auto MAX_SIZE =  32 * MIN_SIZE;
    std::vector<float> times_cpu, times_gpu_g, times_gpu_s;
    for (auto size = MIN_SIZE; size <= MAX_SIZE ; size += STEP) {
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
        ms = multiplyGpuShared(left, right, result_cpu, size);
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
    writeVector(times_cpu, out);
    out << ";\n";
    writeVector(times_gpu_g, out);
    out << ";\n";
    writeVector(times_gpu_s, out);
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
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}

float multiplyGpuGlobal(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size)
{
    const auto SIZE = size * size;
    unsigned *left_gpu, *right_gpu, *result_gpu;
    cudaMalloc(reinterpret_cast<void**>(&left_gpu), SIZE * sizeof(unsigned));
    copyMatrixToGpu(left, size, left_gpu);
    cudaMalloc(reinterpret_cast<void**>(&right_gpu), SIZE * sizeof(unsigned));
    copyMatrixToGpu(right, size, right_gpu);
    cudaMalloc(reinterpret_cast<void**>(&result_gpu), SIZE * sizeof(unsigned));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_count = ceil(size / MAX_SQUARE_BLOCK_WIDTH);
    unsigned block_size;
    if (block_count > 1) {
        block_size = MAX_SQUARE_BLOCK_WIDTH;
    } else {
        block_size = size;
    }
    cudaEventRecord(start);
    multiplyGlobal<<<dim3(block_count, block_count), dim3(block_size, block_size)>>>
        (left_gpu, right_gpu, result_gpu, size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    copyMatrixFromGpu(result_gpu, size, result);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(left_gpu);
    cudaFree(right_gpu);
    cudaFree(result_gpu);
    return ms;
}

float multiplyGpuShared(unsigned const* const* left, unsigned const* const* right, unsigned* const* result, size_t size)
{
    const auto SIZE = size * size;
    unsigned *gpu_left, *gpu_right, *gpu_result;
    cudaMalloc(reinterpret_cast<void**>(&gpu_left), SIZE * sizeof(unsigned));
    copyMatrixToGpu(left, size, gpu_left);
    cudaMalloc(reinterpret_cast<void**>(&gpu_right), SIZE * sizeof(unsigned));
    copyMatrixToGpu(right, size, gpu_right);
    //cudaMalloc(&gpu_result, SIZE * sizeof(&gpu_result));
    cudaMalloc(reinterpret_cast<void**>(&gpu_result), SIZE * sizeof(unsigned));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    unsigned block_count = ceil(size / MAX_SQUARE_BLOCK_WIDTH);
    unsigned block_size;
    if (block_count > 1) {
        block_size = MAX_SQUARE_BLOCK_WIDTH;
    } else {
        block_size = size;
    }
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
