#include <cuda_runtime.h>
#include <iostream>


// run this first to get Visual Studio Developer Command Prompt
// from a remote vscode terminal (must start in command prompt - not powershell)
// "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"   ---> 32bit
// "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
// then compile using
// nvcc main.cu -o main
// then to run
// main

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    if (index < n) {
        c[index] = a[index] + b[index];
    }
    printf("Hello from the GPU! Block %d, Thread %d (Global Thread ID: %d)\n",
           blockIdx.x, threadIdx.x, index);
}

int main() {
    const int n = 1000;              // Number of elements
    const int size = n * sizeof(int); // Total memory size

    // Allocate memory on the host
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Initialize input arrays
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on the device
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int threadsPerBlock = 256; // 256 threads per block
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Ceiling division

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %d != %d\n", i, h_c[i], h_a[i] + h_b[i]);
            return -1;
        }
    }
    printf("Vector addition completed successfully!\n");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}