#pragma once

#define FINPUT "C:\\Users\\mkdan\\Desktop\\ParallelKmeans\\Kmeans\\input.txt"
#define FOUTPUT "C:\\Users\\mkdan\\Desktop\\ParallelKmeans\\Kmeans\\output.txt"
#define MASTER 0
#define POINT_DIM 8
#define CENTROID_DIM 9
#define SLICE_SIZE 1000
#define TAG0 0
#define TAG1 1
#define TAG2 2
#define TAG3 3
#define TAG4 4
#define TAG5 5

struct Centroid
{
	int ID;
	int numPoints;
	double x, y, z;
	double sumX, sumY, sumZ;
	double diameter;
};

struct Point
{
	int ID;
	double dist;
	double x, y, z;
	double Vx, Vy, Vz;
};

