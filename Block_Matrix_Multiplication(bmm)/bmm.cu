//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block

// Best TILEX : 32 and Best TILEY : 16 
#define TILEX 32
#define TILEY 16

// Best L = min(TILEX,TILEY)
#define L (TILEX <= TILEY ? TILEX : TILEY)

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}

dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {

	// Varibale definition needed for loops :
	int p;
	int k;	

	// Main index definition :
	int i = by * TILEY + ty;
	int j = bx * TILEX + tx;
	
	// C'sum Variable :
	float sum = 0;
	
	// Shared memory defnintion :
	__shared__ float as[TILEY][4*L];
	__shared__ float bs[4*L][TILEX];	

	for ( p = 0 ; p < (n/L) ; p = p + 4){
		
		// copy from ad to as (Shared memory) 
		// unrolled loop for best performance :

		if(tx < TILEY){
			as[ty][tx] = ad[((i)<<(m)) + (L*p+tx)];
		}
		
		if(tx < TILEY){
			as[ty][tx + L] = ad[((i)<<(m)) + (L*(p+1)+tx)];
		}
		
		if(tx < TILEY){
			as[ty][tx + 2*L] = ad[((i)<<(m)) + (L*(p+2)+tx)];
		}
		
		if(tx < TILEY){
			as[ty][tx + 3*L] = ad[((i)<<(m)) + (L*(p+3)+tx)];
		}
		
		// copy from bd to bs (Shared memory) 
		// unrolled loop for best performance :

		if (ty < TILEX){
			bs[ty][tx] = bd[((L*p + ty)<<(m)) + (j)];
		}

		if (ty < TILEX){
			bs[ty + L][tx] = bd[((L * (p+1) + ty) << (m)) + (j)];
		}
		
		if (ty < TILEX){
			bs[ty + 2*L][tx] = bd[((L * (p+2) + ty) << (m)) + (j)];
		}
		
		if (ty < TILEX){
			bs[ty + 3*L][tx] = bd[((L * (p+3) + ty) << (m)) + (j)];
		}
		
		// Sync all threads :
		__syncthreads();
		
		// multiplications :
		for (k = 0 ; k < 4 * L ; k++){ 
			sum += as[ ty ][ k ] * bs[ k ][ tx ];
		}
		__syncthreads();
	}
	
	cd [( (i) << (m) ) + ( j )] = sum;
}
