#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" void cudaFunction(float* posX, float* posY, float* posZ, int N, int cube, float offset);

extern "C" void cudaMovement(float* posX, float* posY, float* posZ, int N, int cube);