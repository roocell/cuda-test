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


__global__ void helloFromGPU() {
    printf("Hello from the GPU!\n");
}

int main() {
    helloFromGPU<<<1, 10>>>();  // Launch 10 GPU threads
    cudaDeviceSynchronize();     // Wait for GPU to finish
    return 0;
}