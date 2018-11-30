#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"
#include "Host.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int getThreadsPerBlock();
void checkCuda(const cudaError_t *const status, int line);
void cuda_config(unsigned int * gridSize, unsigned int *blockSize, unsigned int dataSize);
cudaError cuda_calcDiameter(double* host_distances, unsigned int size);
cudaError_t cuda_resetPoints(Point* points, unsigned int N, double dT);
__global__ void movePoints(Point* points, int N, double dT);
__global__ void get_max(double * distances, unsigned int size);

