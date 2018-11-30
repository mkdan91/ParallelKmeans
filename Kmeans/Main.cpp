#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>
#include "Host.h"
#include "Device.h"
using namespace std;

template<typename T1, typename T2> double distance(T1 p1, T2 p2);
double getRand();
void deleteDataFromFile();
void displayStatus(double qm, double QM);
void initDataFile(int N, int K, double T, double dT, int limit, double qm);
int sortByCentroid(const void* p1, const void* p2);
void loadFromFile(int *N, int* K, double *T, double* dT, int *Limit, double *QM, Point *& points);
void saveToFile(Centroid *centroids, int K, double moment, double qm, int iter);
void SetMpiDataTypes(MPI_Datatype * point, MPI_Datatype* centroid);
void resetPoints(int N, Point * points, double dT);
void resetKCneters(int K, Centroid* centroids, Point* points);
void resetBuffer(Point *points, Point** buffer, int bufferSize, int sizeOfSlice);
void setClusterBuffer(Point* sortedPoints, Point** cudaBuffer, Centroid* centroids, int N, int K);
void UpdatePoints(Point* const points, Point* const data, int nslice, int* unConverged);
void UpdateMeansStage(Centroid*  centroids, Point*  points, int K, int N);
void AssignmentStage(Point* dataSlice, Centroid* centroids, int K);
void QMstage(Point* points, Centroid*  centroids, int K, int N, double *qm);


double getRand()
{
	double f, r;
	double min = -100;
	double max = 100;
	f = (double)rand() / RAND_MAX;
	r = min + f * (max - min);
	return r;
}

void deleteDataFromFile()
{
	ofstream ofs;
	ofs.open(FINPUT, ofstream::out | ofstream::trunc);
	ofs.close();
}

void initDataFile(int N, int K, double T, double dT, int limit, double qm)
{
	deleteDataFromFile();
	srand(time(0));
	FILE* f = fopen(FINPUT, "w");
	if (f == NULL)
	{
		printf("Failed open file");
		MPI_Finalize();
		exit(1);
	}

	fprintf(f, "%d %d %lf %lf %d %lf\n", N, K, T, dT, limit, qm);
	for (int i = 0; i < N; i++) {
		fprintf(f, "%lf %lf %lf %lf %lf %lf\n", getRand(), getRand(), getRand(), getRand(), getRand(), getRand());
	}
	fclose(f);
}

void displayStatus(double qm, double QM)
{
	printf("Current QM : %.4f , Target: %.4f \n", qm, QM);
	fflush(stdout);
	printf("...............................\n\n");
	fflush(stdout);
}

// --- calc distance between two type of points (point/cluster) -- 
template<typename T1, typename T2>  double distance(T1 p1, T2 p2) {

	double diffx = 0, diffy = 0, diffz = 0, res = 0;
	diffx = p1.x - p2.x;
	diffy = p1.y - p2.y;
	diffz = p1.z - p2.z;
	res = sqrt((diffx * diffx) + (diffy * diffy) + (diffz * diffz));
	return res;
}

// -- Compare function for qsort :according to the association of each point to each cluster --
int sortByCentroid(const void* p1, const void* p2)
{
	int id1 = ((Point*)p1)->ID;
	int id2 = ((Point*)p2)->ID;
	return id1 - id2;
}

// -- Load all data points& configuration from file
void loadFromFile(int *N, int* K, double *T, double* dT, int *Limit, double *QM, Point *& points)
{
	FILE * f = fopen(FINPUT, "r");
	int i = 0;

	if (f == NULL)
	{
		printf("Failed open file");
		MPI_Finalize();
		exit(1);
	}
	fscanf(f, "%d %d %lf %lf %d %lf", N, K, T, dT, Limit, QM);
	points = (Point*)calloc(*N, sizeof(Point));

#pragma omp parallel for ordered private(i)
	for (i = 0; i < *N; i++)
	{
#pragma omp ordered
		fscanf(f, "%lf %lf %lf %lf %lf %lf", &points[i].x, &points[i].y, &points[i].z, &points[i].Vx, &points[i].Vy, &points[i].Vz);
		points[i].ID = i;
		points[i].dist = 0;
	}
	fclose(f);
}

// -- Save QM & Clusters centers result
void saveToFile(Centroid *centroids, int K, double moment, double qm)
{
	FILE *out = fopen(FOUTPUT, "w");
	int i;

	fprintf(out, "First occurness moment= %lf, QM = %lf\n", moment, qm);
	fprintf(out, "Centroids:\n");

#pragma omp parallel for ordered private(i)
	for (i = 0; i < K; i++) {
#pragma omp ordered
		fprintf(out, "Centroid  %d : [ x : %lf , y : %lf , z : %lf ]\n", centroids[i].ID, centroids[i].x, centroids[i].y, centroids[i].z);
	}
	fclose(out);
}

//------- create MPI point & Centroid data type --- 
void SetMpiDataTypes(MPI_Datatype * mpi_point, MPI_Datatype* mpi_centroid)
{
	int pointMembers[POINT_DIM] = { 1, 1, 1 , 1 , 1 , 1 ,1 ,1 };
	int CentroidMembers[CENTROID_DIM] = { 1 , 1 , 1 , 1 ,1 ,1 , 1 , 1 , 1 };

	MPI_Aint pointAddress[POINT_DIM] = {
		offsetof(Point, ID),
		offsetof(Point, x),
		offsetof(Point, y),
		offsetof(Point, z),
		offsetof(Point, Vx),
		offsetof(Point, Vy),
		offsetof(Point, Vz),
		offsetof(Point, dist) };

	MPI_Aint CentroidAddress[CENTROID_DIM] = {
		offsetof(Centroid, ID),
		offsetof(Centroid, numPoints),
		offsetof(Centroid, x),
		offsetof(Centroid, y),
		offsetof(Centroid, z),
		offsetof(Centroid, sumX),
		offsetof(Centroid, sumY),
		offsetof(Centroid, sumZ),
		offsetof(Centroid, diameter) };


	MPI_Datatype pointMemTypes[POINT_DIM] =
	{
		MPI_INT,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE };

	MPI_Datatype CentroidMemTypes[CENTROID_DIM] = {
		MPI_INT,
		MPI_INT,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE,
		MPI_DOUBLE };

	MPI_Type_struct(POINT_DIM, pointMembers, pointAddress, pointMemTypes, mpi_point);
	MPI_Type_commit(mpi_point);
	MPI_Type_struct(CENTROID_DIM, CentroidMembers, CentroidAddress, CentroidMemTypes, mpi_centroid);
	MPI_Type_commit(mpi_centroid);

}

//--- Move points by given time-period (i.e dT) ---
void resetPoints(int N, Point* points, double dT,double moment, double T)
{
	printf("--- Moment [%.3f / %.3f]--- \n", moment, T);
	printf("K-means: Move points in time...\n");
	fflush(stdout);
	cudaError_t status;
	status = cuda_resetPoints(points, N, dT);
	checkCuda(&status, __LINE__);
	status = cudaDeviceReset();
	checkCuda(&status, __LINE__);
}

// -- Reset centroids with first K points --
void resetKCneters(int K, Centroid* centroids, Point* points)
{
	int i;
#pragma omp parallel for private(i)  
	for (i = 0; i < K; i++)
	{
		centroids[i].ID = i;
		centroids[i].numPoints = 0;
		centroids[i].x = points[i].x;
		centroids[i].y = points[i].y;
		centroids[i].z = points[i].z;

		centroids[i].sumX = 0;
		centroids[i].sumY = 0;
		centroids[i].sumZ = 0;
		centroids[i].diameter = 0;
	}
}

// -- Assign array-of-arrays for send-receive procedure --  
void resetBuffer(Point *points, Point** buffer, int bufferSize, int sizeOfSlice)
{
	int i, j;
#pragma omp parallel for private(i,j) 
	for (i = 0; i < bufferSize; i++)
	{
		if (buffer[i])
			free(buffer[i]);
		buffer[i] = (Point*)calloc(sizeOfSlice, sizeof(Point));
		for (j = 0; j < sizeOfSlice; j++)
		{
			buffer[i][j] = points[j + i * sizeOfSlice];
		}
	}
}

// --  Assign array-of-arrays for Quality measure procedure --  
void setClusterBuffer(Point* sortedPoints, Point** cudaBuffer, Centroid* centroids, int N, int K)
{
	int i, start, numPoints;
	start = 0;
	for (i = 0; i < K; i++)
	{
		numPoints = centroids[i].numPoints;
		if (numPoints == 0)
			cudaBuffer[i] = NULL;
		else {
			cudaBuffer[i] = (Point*)calloc(numPoints, sizeof(Point));
			memcpy(cudaBuffer[i], &sortedPoints[start], sizeof(Point) * numPoints);
		}
		start += numPoints;
	}
}

// -- Update source points array with the values of the data-pieces received from each slave --
void UpdatePoints(Point* const points, Point* const dataSlice, int nslice, int* unConverged)
{
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i< SLICE_SIZE; i++)
	{
		if (points[i + nslice * SLICE_SIZE].ID != dataSlice[i].ID)
		{
			*unConverged = 1;
		}
		points[i + nslice * SLICE_SIZE].ID = dataSlice[i].ID;
		points[i + nslice * SLICE_SIZE].dist = dataSlice[i].dist;
	}
}

//-- Quality measuring stage: CUDA & OMP
void QMstage(Point* points, Centroid*  centroids, int K, int N, double *qm)
{
	cudaError status;
	Point** buffer;
	Point* sortedPoints;
	int j, i, k, numPoints;
	double * distances;
	double dist = 0;

	//-- Allocate & copy & sort all points according to the clusters - association ---  
	buffer = (Point**)calloc(K, sizeof(Point*));
	sortedPoints = (Point*)calloc(N, sizeof(Point));
	memcpy(sortedPoints, points, sizeof(Point) * N);
	qsort(sortedPoints, N, sizeof(Point), sortByCentroid);
	setClusterBuffer(sortedPoints, buffer, centroids, N, K);

	printf("K-means: Start quality measuring..\n");
	printf(".................................\n");
	fflush(stdout);

	for (j = 0; j < K; j++)
	{
		numPoints = centroids[j].numPoints;
		if (numPoints != 0)
		{
			distances = (double*)calloc(numPoints, sizeof(double));
			distances[0] = 0;
			
#pragma omp parallel for private(i,k)
			for (i = 0; i < numPoints; i++)
			{
				distances[i] = distance<Point, Point>(buffer[j][i], buffer[j][0]);
				for (k = 1; k < numPoints; k++)
				{
					dist = distance<Point, Point>(buffer[j][i], buffer[j][k]);
					
					if (dist > distances[i])
					{
						distances[i] = dist;
					}
				}
			}

			status = cuda_calcDiameter(distances, numPoints);
			checkCuda(&status, __LINE__);
			status = cudaDeviceReset();
			checkCuda(&status, __LINE__);

			centroids[j].diameter = distances[0];
			for (i = 0; i < K; i++)
			{
				if (i != j)
				{
					dist = distance <Centroid, Centroid>(centroids[i], centroids[j]);
					*qm += (centroids[j].diameter / dist);
				}
			}
			free(distances);
		}
	}
	free(sortedPoints);
	for (i = 0; i < K; i++)
		if (buffer[i])
			free(buffer[i]);
	free(buffer);
	*qm = *qm / (K * (K - 1));
}

// -- Calculate the new means to be the centroids of the observations in the new clusters -- 
void UpdateMeansStage(Centroid* centroids, Point*  points, int K, int N)
{
	int i, j, idx;
#pragma omp parallel for private(i)
	for (i = 0; i < N; i++)
	{
#pragma omp critical
		{
			idx = points[i].ID;
			centroids[idx].numPoints += 1;
			centroids[idx].sumX += points[i].x;
			centroids[idx].sumY += points[i].y;
			centroids[idx].sumZ += points[i].z;
		}
	}
#pragma omp parallel for private(j)
	for (j = 0; j < K; j++)
	{
		if (centroids[j].numPoints != 0)
		{
			centroids[j].x = centroids[j].sumX / centroids[j].numPoints;
			centroids[j].y = centroids[j].sumY / centroids[j].numPoints;
			centroids[j].z = centroids[j].sumZ / centroids[j].numPoints;
		}
	}
}

//-- Assign each observation to the cluster whose mean has the least squared Euclidean distance --
void AssignmentStage(Point* dataSlice, Centroid* centroids, int K)
{
	int i, j;
	double dist;

#pragma omp parallel for private(j,dist)  
	for (i = 0; i < SLICE_SIZE; i++) {
		j = 0;
		dataSlice[i].dist = distance<Point, Centroid>(dataSlice[i], centroids[j]);
		dataSlice[i].ID = j;
		for (j = 1; j < K; j++)
		{
#pragma omp critical 
			dist = distance<Point, Centroid>(dataSlice[i], centroids[j]);
			if (dist< dataSlice[i].dist)
			{
				dataSlice[i].dist = dist;
				dataSlice[i].ID = j;
			}
		}
	}
}

int main(int argc, char *argv[])
{
	//initDataFile(200000, 80, 30, 0.5, 200, 0.06);
	MPI_Datatype MPI_point, MPI_Centroid;
	Point* points, *dataSlice;
	Point** buffer;
	Centroid* centroids;
	MPI_Status status;

	double QM, qm , dT ,T, moment=0;
	int i, rank, numProces, N, K, Limit, iteration;
	int slice, recvSlice, bufferSize;
	int terminate, unConverged, achivedQM;

	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		printf("MPI Error: Failed to initialize MPI!\n");
		return -1;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProces);
	SetMpiDataTypes(&MPI_point, &MPI_Centroid);

	if (numProces < 3)
	{
		printf("MPI Error: Required at least 3 process\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	if (rank == MASTER)
	{
		loadFromFile(&N, &K, &T, &dT, &Limit, &QM, points);
		bufferSize = N / SLICE_SIZE;
		centroids = (Centroid*)calloc(K, sizeof(Centroid));
		buffer = (Point**)calloc(bufferSize, sizeof(Point*));
		dataSlice = (Point*)calloc(SLICE_SIZE, sizeof(Point));
		
		terminate = 0;
		achivedQM = 0;
		resetPoints(N, points, 0 ,moment,T);
		for (moment = 0; moment < T && !achivedQM ; moment += dT)
		{
			qm = 0;  unConverged = 1;
			if (moment != 0)
				resetPoints(N, points, dT, moment,T);
			resetKCneters(K, centroids, points);
			resetBuffer(points, buffer, bufferSize, SLICE_SIZE);

			for (iteration = 0; iteration < Limit && unConverged; iteration++)
			{
				slice = 0;
				recvSlice = 0;
				unConverged = 0;
				if (iteration == 0)
				{
					printf("K-means: Send-Receive procedure..\n");
					fflush(stdout);
					for (i = 1; i < numProces; i++)
					{
						MPI_Send(&K, 1, MPI_INT, i, TAG0, MPI_COMM_WORLD);
					}
				}
				// -- Master send to each slave piece of data-points
				for (i = 1; i < numProces; i++)
				{
					dataSlice = buffer[slice];
					MPI_Send(&terminate, 1, MPI_INT, i, TAG1, MPI_COMM_WORLD);
					MPI_Send(&slice, 1, MPI_INT, i, TAG2, MPI_COMM_WORLD);
					MPI_Send(dataSlice, SLICE_SIZE, MPI_point, i, TAG3, MPI_COMM_WORLD);
					MPI_Send(centroids, K, MPI_Centroid, i, TAG4, MPI_COMM_WORLD);
					slice++;
				}
				//-- Master receive a processed piece of data from slave and send it a new one
				while (slice < bufferSize)
				{
					MPI_Recv(&recvSlice, 1, MPI_INT, MPI_ANY_SOURCE, TAG5, MPI_COMM_WORLD, &status);
					MPI_Recv(dataSlice, SLICE_SIZE, MPI_point, status.MPI_SOURCE, TAG3, MPI_COMM_WORLD, &status);
					UpdatePoints(points, dataSlice, recvSlice, &unConverged);
					dataSlice = buffer[slice];

					MPI_Send(&terminate, 1, MPI_INT, status.MPI_SOURCE, TAG1, MPI_COMM_WORLD);
					MPI_Send(&slice, 1, MPI_INT, status.MPI_SOURCE, TAG2, MPI_COMM_WORLD);
					MPI_Send(dataSlice, SLICE_SIZE, MPI_point, status.MPI_SOURCE, TAG3, MPI_COMM_WORLD);
					MPI_Send(centroids, K, MPI_Centroid, status.MPI_SOURCE, TAG4, MPI_COMM_WORLD);
					slice++;
				}
				//-- Master receive the last processed pieces of data 
				for (i = 1; i < numProces; i++)
				{
					MPI_Recv(&recvSlice, 1, MPI_INT, MPI_ANY_SOURCE, TAG5, MPI_COMM_WORLD, &status);
					MPI_Recv(dataSlice, SLICE_SIZE, MPI_point, status.MPI_SOURCE, TAG3, MPI_COMM_WORLD, &status);
					UpdatePoints(points, dataSlice, recvSlice, &unConverged);
				}
				resetKCneters(K, centroids, points);
				UpdateMeansStage(centroids, points, K, N);
				resetBuffer(points, buffer, bufferSize, SLICE_SIZE);
			}

			QMstage(points, centroids, K, N, &qm);
			displayStatus(qm, QM);
			
			if (qm > QM)
				achivedQM = FALSE;
			else
				achivedQM = TRUE;
		}
		saveToFile(centroids, K, moment - dT, qm);
		free(points);
		free(centroids);
		free(dataSlice);
		free(buffer);

		for (i = 1; i < numProces; i++)
		{
			terminate = 1;
			MPI_Send(&terminate, 1, MPI_INT, i, TAG1, MPI_COMM_WORLD);
		}
	}
	else //------------------- SLAVE JOB --------------------
	{
	
		MPI_Recv(&K, 1, MPI_INT, MASTER, TAG0, MPI_COMM_WORLD, &status);
		MPI_Recv(&terminate, 1, MPI_INT, MASTER, TAG1, MPI_COMM_WORLD, &status);
		dataSlice = (Point*)calloc(SLICE_SIZE, sizeof(Point));
		centroids = (Centroid*)calloc(K, sizeof(Centroid));

		//-- until it won't be ordered to stop- slave receive a piece of data from the master and start the assignment procedure
		while (terminate == 0)
		{
			MPI_Recv(&recvSlice, 1, MPI_INT, MASTER, TAG2, MPI_COMM_WORLD, &status);
			MPI_Recv(dataSlice, SLICE_SIZE, MPI_point, MASTER, TAG3, MPI_COMM_WORLD, &status);
			MPI_Recv(centroids, K, MPI_Centroid, MASTER, TAG4, MPI_COMM_WORLD, &status);
			AssignmentStage(dataSlice, centroids, K);
			MPI_Send(&recvSlice, 1, MPI_INT, MASTER, TAG5, MPI_COMM_WORLD);
			MPI_Send(dataSlice, SLICE_SIZE, MPI_point, MASTER, TAG3, MPI_COMM_WORLD);
			MPI_Recv(&terminate, 1, MPI_INT, MASTER, TAG1, MPI_COMM_WORLD, &status);
		}

		free(centroids);
		free(dataSlice);
	}

	printf("Task  %d is finihed \n", rank);
	fflush(stdout);
	MPI_Type_free(&MPI_point);
	MPI_Type_free(&MPI_Centroid);
	MPI_Finalize();
	return 0;
}