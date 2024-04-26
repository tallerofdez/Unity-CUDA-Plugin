#define _USE_MATH_DEFINES

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <string>



#pragma warning(2:4235)

#define DllExport __declspec (dllexport)

# define M_PI           3.14159265358979323846  /* pi */



__global__ void instanceParticles(float* outX, float* outY, float* outZ, float offset, int N, int WIDTH, int HEIGHT, int DEPTH) {

	unsigned int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int Zidx = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int idx = Xidx + Yidx * WIDTH + Zidx * WIDTH * HEIGHT;

	if (Xidx < WIDTH && Yidx < HEIGHT && Zidx < DEPTH) {
		outX[idx] = Xidx * offset;
		outY[idx] = Yidx * offset;
		outZ[idx] = Zidx * offset;
	}

}

__global__ void moveParticles(float* outX, float* outY, float* outZ, float* initialX, float* initialY, float* initialZ, float movement, int N, int WIDTH, int HEIGHT, int DEPTH) {

	unsigned int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int Zidx = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int idx = Xidx + Yidx * WIDTH + Zidx * WIDTH * HEIGHT;

	if (Xidx < WIDTH && Yidx < HEIGHT && Zidx < DEPTH) {

		outX[idx] = initialX[idx];
		outY[idx] = initialY[idx] + movement;
		outZ[idx] = initialZ[idx];

	}
}

__global__ void setInitialPos(float* inX, float* inY, float* inZ, float* outX, float* outY, float* outZ, int N, int WIDTH, int HEIGHT, int DEPTH) {
	unsigned int Xidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int Yidx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int Zidx = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int idx = Xidx + Yidx * WIDTH + Zidx * WIDTH * HEIGHT;

	if (Xidx < WIDTH && Yidx < HEIGHT && Zidx < DEPTH) {

		outX[idx] = inX[idx];
		outY[idx] = inY[idx];
		outZ[idx] = inZ[idx];

	}
}




extern "C" {



	DllExport const char*  CUDA_device_name()
	{
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, 0);
		char* label = new char[256];
		strcpy_s(label, 256, device.name);
		return label;
	}

	DllExport void Clear(float* host)
	{
		free(host);
	}

	DllExport const void  cubeFormation(float* posX, float* posY, float* posZ, int N, int cube, float offset)
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


		instanceParticles << <gridDim, blockDim >> > (d_X, d_Y, d_Z, offset, N, cube, cube, cube);
		cudaDeviceSynchronize();

		cudaMemcpy(posX, d_X, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(posY, d_Y, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(posZ, d_Z, sizeof(float) * N, cudaMemcpyDeviceToHost);

		cudaFree(d_X);
		cudaFree(d_Y);
		cudaFree(d_Z);

		

	}


	DllExport const void InitialPos (float* posX, float* posY, float* posZ, float* initialPosX, float* initialPosY, float* initialPosZ, int N, int cube) {
		
		
		const int THREAD_SIZE = cube;
		const int BLOCK_SIZE = floor(cbrt(1024));

		float* d_inX;
		float* d_inY;
		float* d_inZ;

		float* d_outX;
		float* d_outY;
		float* d_outZ;

		cudaMalloc(&d_inX, sizeof(float) * N);
		cudaMalloc(&d_inY, sizeof(float) * N);
		cudaMalloc(&d_inZ, sizeof(float) * N);

		cudaMalloc(&d_outX, sizeof(float) * N);
		cudaMalloc(&d_outY, sizeof(float) * N);
		cudaMalloc(&d_outZ, sizeof(float) * N);

		cudaMemcpy(d_inX, posX, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_inY, posY, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_inZ, posZ, sizeof(float) * N, cudaMemcpyHostToDevice);

		dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridDim(ceil((double)cube / blockDim.x), ceil((double)cube / blockDim.y), ceil((double)cube / blockDim.z));

		setInitialPos << < gridDim, blockDim >> > (d_inX, d_inY, d_inZ, d_outX, d_outY, d_outZ, N, cube, cube, cube);

		cudaMemcpy(initialPosX, d_outX, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(initialPosY, d_outY, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(initialPosZ, d_outZ, sizeof(float) * N, cudaMemcpyDeviceToHost);

		cudaFree(d_outX);
		cudaFree(d_outY);
		cudaFree(d_outZ);

		cudaFree(d_inX);
		cudaFree(d_inY);
		cudaFree(d_inZ);

	}


	DllExport const void   cubeMovement(float* posX, float* posY, float* posZ, float* initialPosX, float* initialPosY, float* initialPosZ, int N, int cube, float ciclos) {


		float tau = M_PI * 2;
		float sinFunction = std::sin(tau * ciclos);
		float movement = (sinFunction / 2) + 0.5f;
		/**
		FILE* f = fopen("log_cuda.txt", "a");
		fprintf(f, "movement: %f , sin : %f  \n", movement, sinFunction);
		fprintf(f, "cudaMemcpy. posX[0] = %f, posY[0] = %f , posZ[0] = %f\n", posX[0], posY[0], posZ[0]);
		fprintf(f, "cudaMemcpy. INiPosX[0] = %f, iniPosY[0] = %f , iniPosZ[0] = %f\n", initialPosX[0], initialPosY[0], initialPosZ[0]);
		fclose(f);
		/**/
		const int THREAD_SIZE = cube;
		const int BLOCK_SIZE = floor(cbrt(1024));

		float* d_X;
		float* d_Y;
		float* d_Z;

		float* initialX;
		float* initialY;
		float* initialZ;

		cudaMalloc(&d_X, sizeof(float) * N);
		cudaMalloc(&d_Y, sizeof(float) * N);
		cudaMalloc(&d_Z, sizeof(float) * N);

		cudaMalloc(&initialX, sizeof(float) * N);
		cudaMalloc(&initialY, sizeof(float) * N);
		cudaMalloc(&initialZ, sizeof(float) * N);

		cudaMemcpy(initialX, initialPosX, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(initialY, initialPosY, sizeof(float) * N, cudaMemcpyHostToDevice);
		cudaMemcpy(initialZ, initialPosZ, sizeof(float) * N, cudaMemcpyHostToDevice);

		/**/
		dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
		dim3 gridDim(ceil((double)cube / blockDim.x), ceil((double)cube / blockDim.y), ceil((double)cube / blockDim.z));

		moveParticles << <gridDim, blockDim >> > (d_X, d_Y, d_Z, initialX, initialY, initialZ, movement, N, cube, cube, cube);
		cudaDeviceSynchronize();


		cudaMemcpy(posX, d_X, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(posY, d_Y, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(posZ, d_Z, sizeof(float) * N, cudaMemcpyDeviceToHost);
		/**
		f = fopen("log_cuda.txt", "a");
		fprintf(f, "cudaMemcpy. posX[0] = %f, posY[0] = %f , posZ[0] = %f\n", posX[0], posY[0], posZ[0]);
		fclose(f);
		/**/
		cudaFree(d_X);
		cudaFree(d_Y);
		cudaFree(d_Z);

		cudaFree(initialX);
		cudaFree(initialY);
		cudaFree(initialZ);

	}


}