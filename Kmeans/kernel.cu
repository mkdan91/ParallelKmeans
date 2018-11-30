#pragma once
#include "Device.h"
#ifndef __CUDACC__   
#define __CUDACC__
#endif
#include <device_functions.h>


int getThreadsPerBlock()
{
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	return properties.maxThreadsPerBlock;
}

void checkCuda(const cudaError_t *const status, int line)
{
	if (*status != cudaSuccess)
	{
		printf("CUDA ERROR [line: %d]: %s\n", line - 1, cudaGetErrorString(*status));
	}
}

// -- Find optimal number of Blocks& threads 
void cuda_config(unsigned int * gridSize, unsigned int *blockSize, unsigned int dataSize)
{
	unsigned int remainder ;
	unsigned int nThreadPerBlock = getThreadsPerBlock();

	*blockSize = nThreadPerBlock;
	*gridSize = dataSize / nThreadPerBlock;
	remainder = dataSize % nThreadPerBlock;
	if (remainder != 0)
		*gridSize=*gridSize+1;
}

//-- Find the largest diameter between all distances in the current cluster: activate only half of threads -- 
__global__ void get_max(double * distances, unsigned int size) {

	double dist;
	unsigned int index = threadIdx.x + (blockDim.x * blockIdx.x);
	unsigned int totalSize = size;

	while (totalSize > 1)
	{
		unsigned int halfPoint = totalSize / 2;
		unsigned int remained = totalSize % 2;

		if (index < halfPoint || index == halfPoint && remained)
		{
			dist = distances[index + halfPoint];
			if (dist > distances[index]) {
				distances[index] = dist;
			}
		}
		__syncthreads();
		totalSize = totalSize / 2 + remained;
	}
}

//-- Reset all points by time -- 
__global__ void movePoints(Point* points, int N, double dT)
{
	unsigned index = threadIdx.x + blockDim.x *blockIdx.x;
	if (index < N)
	{
		points[index].ID = index;
		points[index].x += points[index].Vx * dT;
		points[index].y += points[index].Vy * dT;
		points[index].z += points[index].Vz * dT;
		points[index].dist = 0;
	}
}

cudaError cuda_calcDiameter(double* host_distances, unsigned int size)
{
	cudaError_t status;
	double* dev_distances;
	unsigned int gridSize, blockSize; 
	
	//-- Allocate device memmory--
	status= cudaMalloc((void**)&dev_distances, size * sizeof(double));
	checkCuda(&status, __LINE__);

	//--Copy from host to device memmory --
	status=cudaMemcpy(dev_distances, host_distances, size * sizeof(double), cudaMemcpyHostToDevice);
	checkCuda(&status, __LINE__);
	
	//-- Set cuda blocks & threads configuration for kernal call
	cuda_config(&gridSize, &blockSize, size);
	get_max<<<gridSize, blockSize>>>(dev_distances, size);
	
	status = cudaGetLastError();
	checkCuda(&status, __LINE__);

	//-- Check if any error caused during the kernal launch
	status = cudaDeviceSynchronize();
	checkCuda(&status, __LINE__);

	//-- Copy back from device to host memmory -- 
	status = cudaMemcpy(host_distances, dev_distances, size * sizeof(double), cudaMemcpyDeviceToHost);
	checkCuda(&status, __LINE__);

	//-- Free device memmory
	status= cudaFree(dev_distances);
	checkCuda(&status, __LINE__);

	return status;
}

cudaError_t cuda_resetPoints(Point* points, unsigned int N, double dT)
{
	Point* dev_points;
	cudaError_t status;
	unsigned int gridSize, blockSize;

	status = cudaSetDevice(0);
	checkCuda(&status, __LINE__);

	//-- Allocate device memmory--
	status = cudaMalloc((void**)&dev_points, N * sizeof(Point));
	checkCuda(&status, __LINE__);

	//--Copy from host to device memmory --
	status = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
	checkCuda(&status, __LINE__);
	
	//-- Set cuda blocks & threads configuration for kernal call
	cuda_config(&gridSize, &blockSize, N);
	movePoints <<<gridSize, blockSize >>>(dev_points, N, dT);

	status = cudaGetLastError();
	checkCuda(&status, __LINE__);
	
	//-- Check if any error caused during the kernal launch
	status = cudaDeviceSynchronize();
	checkCuda(&status, __LINE__);

	//-- Copy back from device to host memmory --  
	status = cudaMemcpy(points, dev_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
	checkCuda(&status, __LINE__);

	//-- Free device memmory
	cudaFree(dev_points);
	return status;
}
