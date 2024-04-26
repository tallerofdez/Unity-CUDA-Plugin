//#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "IUnityGraphics.h"
#include "IUnityInterface.h"
#include <string.h>
#include <iostream>
#include <cmath>
#include <string>

#include "File.cuh"



#pragma warning(2:4235)

#define DllExport __declspec (dllexport)



extern "C" {



	DllExport const char* UNITY_INTERFACE_API UNITY_INTERFACE_API CUDA_device_name() 
	{ 
		cudaDeviceProp device;
		cudaGetDeviceProperties(&device, 0);
		char* label = new char[256];
		strcpy_s(label, 256, device.name);
		return label;
	}
	
	DllExport const void UNITY_INTERFACE_API UNITY_INTERFACE_API cubeFormation(float* posX, float* posY, float* posZ, int numParticles, int cube, float offset)
	{
		cudaFunction(posX, posY, posZ, numParticles, cube, offset);
	}
	

	DllExport const void UNITY_INTERFACE_API UNITY_INTERFACE_API cubeMovement(float* posX, float* posY, float* posZ, int N, int cube) {
		cudaMovement(posX, posY, posZ, N, cube);
	}


}