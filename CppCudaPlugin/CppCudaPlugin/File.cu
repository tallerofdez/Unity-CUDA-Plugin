#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>


#include "File.cuh"

__global__ void instanceParticles(float* outX, float* outY, float* outZ, float offset, int N ,int WIDTH, int HEIGHT, int DEPTH) {

	unsigned int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int Zidx = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int idx = Xidx + Yidx * WIDTH + Zidx * WIDTH * HEIGHT ;

	if (Xidx < WIDTH && Yidx < HEIGHT && Zidx < DEPTH) {
	outX[idx] = Xidx * offset;
	outY[idx] = Yidx * offset;
	outZ[idx] = Zidx * offset;
	}

}

__global__ void moveParticles(float* outX, float* outY, float* outZ, int N, int WIDTH, int HEIGHT, int DEPTH) {
	
	unsigned int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int Zidx = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int idx = Xidx + Yidx * WIDTH + Zidx * WIDTH * HEIGHT;

	if (Xidx < WIDTH && Yidx < HEIGHT && Zidx < DEPTH) {
		outX[idx] = 1;
		outY[idx] = 1;
		outZ[idx] = 1;
	}
}


extern "C" void cudaFunction(float* posX, float* posY, float* posZ, int N, int cube, float offset) {

	const int THREAD_SIZE = cube;
	const int BLOCK_SIZE = floor(cbrt(1024));


	float* d_X;
	float* d_Y;
	float* d_Z;


	cudaMalloc(&d_X, sizeof(float)*N);
	cudaMalloc(&d_Y, sizeof(float)*N);
	cudaMalloc(&d_Z, sizeof(float)*N);


	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(ceil((double)cube / blockDim.x), ceil((double)cube / blockDim.y), ceil((double)cube / blockDim.z));


	instanceParticles << <gridDim, blockDim >> > (d_X, d_Y, d_Z, offset, N, cube, cube, cube);
	cudaDeviceSynchronize();


	cudaMemcpy(posX, d_X, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(posY, d_Y, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(posZ, d_Z, sizeof(float) * N, cudaMemcpyDeviceToHost);


	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);

}

extern "C" void cudaMovement(float* posX, float* posY, float* posZ, int N, int cube)
{
	
	const int THREAD_SIZE = cube;
	const int BLOCK_SIZE = floor(cbrt(1024));


	float* d_X;
	float* d_Y;
	float* d_Z;


	cudaMalloc(&d_X, sizeof(float) * N);
	cudaMalloc(&d_Y, sizeof(float) * N);
	cudaMalloc(&d_Z, sizeof(float) * N);

	/**/
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim(ceil((double)cube / blockDim.x), ceil((double)cube / blockDim.y), ceil((double)cube / blockDim.z));
	/**/

	moveParticles << <gridDim, blockDim >> > (d_X, d_Y, d_Z, N, cube, cube, cube);
	cudaDeviceSynchronize();


	cudaMemcpy(posX, d_X, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(posY, d_Y, sizeof(float) * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(posZ, d_Z, sizeof(float) * N, cudaMemcpyDeviceToHost);


	cudaFree(d_X);
	cudaFree(d_Y);
	cudaFree(d_Z);

	
}
