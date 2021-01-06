// ONLY MODIFY THIS FILE

#include "scan2.h"
#include "gpuerrors.h"
#include <cmath>
#include <cuda.h>

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

//#define Block_Size 1024

// This GPU kernel does blockwise in-place scan 
__global__ void Blelloch_Exclusive_Scan(float *d_in, float* d_out, int Block_Size)
{
    __shared__ float sh_array[1024];
    int id = bx * Block_Size + tx;
    // Copying data from global to shared memory
    sh_array[tx] = d_in[id];
    __syncthreads();
	
    /** Performing block-wise in-place Blelloch scan **/
    // First step of Blelloch scan : REDUCTION
    for(int k=2; k <= Block_Size; k *= 2)
    {
        if((tx+1) % k == 0)
        {
            sh_array[tx] = sh_array[tx - (k/2)] + sh_array[tx];
        }
        __syncthreads();
    }
	
	
    // At the end of reduction, the last element of each block conatins the sum of all elements in that block
    // We store these block-wise sums in d_out
    if(tx == (Block_Size - 1))
    {
        d_out[bx] = sh_array[tx];
        sh_array[tx] = 0;
    }
    __syncthreads();

    // Second step of Blelloch scan : DOWNSWEEP 
    // This is structurally the exact reverse of the reduction step
    for(int k = Block_Size; k >= 2; k /= 2)
    {
        if((tx+1) % k == 0)
        {
            float temp = sh_array[tx - (k/2)];
            sh_array[tx - (k/2)] = sh_array[tx];
            sh_array[tx] = temp + sh_array[tx];
        }
        __syncthreads();
    }
	
	
    // Copying the scan result back into global memory
    d_in[id] = sh_array[tx];
	
    // d_in now contains blockwise scan result
    __syncthreads();
}

// This GPU kernel adds the value d_out[id] to all values in the (id)th block of d_in
__global__ void Max(float* d_in, float* d_out, int Block_Size)
{
    int id = bx * Block_Size + tx;
    d_in[id] = d_out[bx] + d_in[id];
    __syncthreads();
}



void gpuKernel(float* a, float* c,int n) {
	
	if(n <= 33554432) //n <= 2^25
	{
		float* ad;
		float* cd;
		float* cdd;
		float* d_max;
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, (n / 1024) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cdd, (n / (1024*1024)) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_max, sizeof(float)));
		
		HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c, ad+1, (n-1) * sizeof(float), cudaMemcpyDeviceToHost));
		c[n-1] = c[n-2] + a[n-1];
		
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));
		HANDLE_ERROR(cudaFree(cdd));
		HANDLE_ERROR(cudaFree(d_max));
	}
	else if(n == 67108864) // 2^26
	{
		n = n/2;
		float* ad;
		float* cd;
		float* cdd;
		float* d_max;
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, (n / 1024) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cdd, (n / (1024*1024)) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_max, sizeof(float)));
		
		// first half
		HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c, ad+1, (n-1) * sizeof(float), cudaMemcpyDeviceToHost));
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		// second half
		HANDLE_ERROR(cudaMemcpy(ad, a+n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));
		HANDLE_ERROR(cudaFree(cdd));
		HANDLE_ERROR(cudaFree(d_max));
	}
	else if(n == 134217728) // 2^27
	{
		n = n/4;
		float* ad;
		float* cd;
		float* cdd;
		float* d_max;
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, (n / 1024) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cdd, (n / (1024*1024)) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_max, sizeof(float)));
		
		// first 1/4
		HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c, ad+1, (n-1) * sizeof(float), cudaMemcpyDeviceToHost));
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		// second 1/4
		HANDLE_ERROR(cudaMemcpy(ad, a+n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		// Third 1/4
		HANDLE_ERROR(cudaMemcpy(ad, a+2*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+2*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];

		// Fourth 1/4
		HANDLE_ERROR(cudaMemcpy(ad, a+3*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+3*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));
		HANDLE_ERROR(cudaFree(cdd));
		HANDLE_ERROR(cudaFree(d_max));
	}
	else if(n == 268435456) // 2^28
	{
		n = n/8;
		float* ad;
		float* cd;
		float* cdd;
		float* d_max;
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, (n / 1024) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cdd, (n / (1024*1024)) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_max, sizeof(float)));
		
		// first 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c, ad+1, (n-1) * sizeof(float), cudaMemcpyDeviceToHost));
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		// second 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		// Third 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+2*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+2*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];

		// Fourth 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+3*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+3*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		a[4*n-1] = c[4*n-1];
		
		// 5th 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+4*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+4*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[5*n-2] = c[5*n-3] + a[5*n-2];
		c[5*n-1] = c[5*n-2] + a[5*n-1];
		a[5*n-1] = c[5*n-1];
		
		// 6th 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+5*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+5*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[6*n-2] = c[6*n-3] + a[6*n-2];
		c[6*n-1] = c[6*n-2] + a[6*n-1];
		a[6*n-1] = c[6*n-1];
		
		// 7th 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+6*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+6*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[7*n-2] = c[7*n-3] + a[7*n-2];
		c[7*n-1] = c[7*n-2] + a[7*n-1];
		a[7*n-1] = c[7*n-1];
		
		// 8th 1/8
		HANDLE_ERROR(cudaMemcpy(ad, a+7*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+7*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[8*n-2] = c[8*n-3] + a[8*n-2];
		c[8*n-1] = c[8*n-2] + a[8*n-1];
		
		
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));
		HANDLE_ERROR(cudaFree(cdd));
		HANDLE_ERROR(cudaFree(d_max));
	}
	else if(n == 536870912) // 2^29
	{
		n = n/16;
		float* ad;
		float* cd;
		float* cdd;
		float* d_max;
		
		HANDLE_ERROR(cudaMalloc((void**)&ad, n * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cd, (n / 1024) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&cdd, (n / (1024*1024)) * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&d_max, sizeof(float)));
		
		// first 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c, ad+1, (n-1) * sizeof(float), cudaMemcpyDeviceToHost));
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		// second 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		// Third 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+2*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+2*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];

		// Fourth 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+3*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+3*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		a[4*n-1] = c[4*n-1];
		
		// 5th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+4*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+4*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[5*n-2] = c[5*n-3] + a[5*n-2];
		c[5*n-1] = c[5*n-2] + a[5*n-1];
		a[5*n-1] = c[5*n-1];
		
		// 6th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+5*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+5*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[6*n-2] = c[6*n-3] + a[6*n-2];
		c[6*n-1] = c[6*n-2] + a[6*n-1];
		a[6*n-1] = c[6*n-1];
		
		// 7th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+6*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+6*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[7*n-2] = c[7*n-3] + a[7*n-2];
		c[7*n-1] = c[7*n-2] + a[7*n-1];
		a[7*n-1] = c[7*n-1];
		
		// 8th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+7*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+7*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[8*n-2] = c[8*n-3] + a[8*n-2];
		c[8*n-1] = c[8*n-2] + a[8*n-1];
		a[8*n-1] = c[8*n-1];
		
		// 9th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+8*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+8*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[9*n-2] = c[9*n-3] + a[9*n-2];
		c[9*n-1] = c[9*n-2] + a[9*n-1];
		a[9*n-1] = c[9*n-1];
		
		// 10th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+9*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+9*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[10*n-2] = c[10*n-3] + a[10*n-2];
		c[10*n-1] = c[10*n-2] + a[10*n-1];
		a[10*n-1] = c[10*n-1];
		
		// 11th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+10*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+10*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[11*n-2] = c[11*n-3] + a[11*n-2];
		c[11*n-1] = c[11*n-2] + a[11*n-1];
		a[11*n-1] = c[11*n-1];
		
		// 12th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+11*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+11*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[12*n-2] = c[12*n-3] + a[12*n-2];
		c[12*n-1] = c[12*n-2] + a[12*n-1];
		a[12*n-1] = c[12*n-1];
		
		// 13th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+12*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+12*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[13*n-2] = c[13*n-3] + a[13*n-2];
		c[13*n-1] = c[13*n-2] + a[13*n-1];
		a[13*n-1] = c[13*n-1];
		
		// 14th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+13*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+13*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[14*n-2] = c[14*n-3] + a[14*n-2];
		c[14*n-1] = c[14*n-2] + a[14*n-1];
		a[14*n-1] = c[14*n-1];
		
		// 15th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+14*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+14*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[15*n-2] = c[15*n-3] + a[15*n-2];
		c[15*n-1] = c[15*n-2] + a[15*n-1];
		a[15*n-1] = c[15*n-1];
		
		// 16th 1/16
		HANDLE_ERROR(cudaMemcpy(ad, a+15*n-1, n * sizeof(float), cudaMemcpyHostToDevice));
		
		Blelloch_Exclusive_Scan<<< n/1024 , 1024 >>>(ad,cd,1024);
		Blelloch_Exclusive_Scan<<< n/(1024*1024) , 1024 >>>(cd,cdd,1024);
		Blelloch_Exclusive_Scan<<< 1 , n/(1024*1024) >>>(cdd,d_max,n/(1024*1024));
		Max <<< n/(1024*1024) , 1024>>> (cd,cdd,1024);
		Max <<< n/1024 , 1024>>> (ad,cd,1024);	

		HANDLE_ERROR(cudaMemcpy(c+15*n, ad+2, (n-2) * sizeof(float), cudaMemcpyDeviceToHost));
		c[16*n-2] = c[16*n-3] + a[16*n-2];
		c[16*n-1] = c[16*n-2] + a[16*n-1];
		
		HANDLE_ERROR(cudaFree(ad));
		HANDLE_ERROR(cudaFree(cd));
		HANDLE_ERROR(cudaFree(cdd));
		HANDLE_ERROR(cudaFree(d_max));
	}
}