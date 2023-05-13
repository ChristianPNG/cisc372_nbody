#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

__global__ void computeAccels(vector3 *accels, vector3 *hPos, vector3 *hVel, double *mass){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < NUMENTITIES){
		if(j < NUMENTITIES){
			if (i==j) {
				FILL_VECTOR(accels[i*NUMENTITIES+j],0,0,0);
			}
			else{
				vector3 distance;
				distance[0]=hPos[i][0]-hPos[j][0];
				distance[1]=hPos[i][1]-hPos[j][1];
				distance[2]=hPos[i][2]-hPos[j][2];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i*NUMENTITIES+j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}
}

__global__ void sumAccels(vector3 *accels, vector3 *hPos, vector3 *hVel){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i < NUMENTITIES){
        vector3 accel_sum={0,0,0};
		if(j < NUMENTITIES){
			for (int k=0;k<3;k++)
				accel_sum[k]+=accels[i*NUMENTITIES+j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (int k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]=hVel[i][k]*INTERVAL;
		}
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
extern "C" void compute(){
	double *d_mass;
	vector3 *d_accels;

    cudaMalloc(&d_accels, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&d_hVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&d_mass, sizeof(double)*NUMENTITIES);

	cudaMemcpy(d_hPos, hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_hVel, hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mass, mass, sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);

	dim3 blockSize(16,16);
	int num = (NUMENTITIES + 16 - 1)/16;
    dim3 numBlocks(num, num);

    computeAccels<<<numBlocks, blockSize>>>(d_accels, d_hPos, d_hVel, d_mass);
	sumAccels<<<numBlocks, blockSize>>>(d_accels, d_hPos, d_hVel);
	cudaDeviceSynchronize();

	cudaMemcpy(hPos, d_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);

	cudaFree(d_hPos);
	cudaFree(d_hVel);
	cudaFree(d_mass);
}
