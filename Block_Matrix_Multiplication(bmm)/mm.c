#define tx threadIdx.x
#define bx blockIdx.x

serial:
for i=0..n-1
for j=0..n-1
for k=0..n-1
	c[i][j] += a[i][k]*b[k][j]

parallel:
mult(a,b,c,n){
	i = 32*by + ty;
	j = 32*bx + tx;
	for k=0..n-1
		c[i][j] += a[i][k] * b[k][j]
}
R =(n^2)thread * (2*n) = 2*n^3
W = n^3

n=32
mult<<<1,dim3(32,32,1)>>>(a,b,c,n);

n=2^20
mult<<<dim3(n/32,n/32,1),dim3(32,32,1)>>>(a,b,c,n);

mult(a,b,c,n){
	i = by;
	j = tx;
	k = tz;
	ab[k] = a[i][k] * b[k][j];
	...
}
mult<<<dim3(1,n,1), dim3(n,1,n)>>>(a,b,c,n);
R =(n^3)thread * (2) = 2*n^3
R =(n^2)thread * (2*n) = 2*n^3
W = n^3


bmm(a,b,c,n){
	i = T*by + ty;
	j = T*bx + tx;
	float sum =0;

	__shared__ float as[T][T];
	__shared__ float bs[T][T];	

	for p=0..(n/T -1) { //0..3
		//copy to shared memory:
		as[ty][tx] = a[i][T*p +tx]; 
		bs[ty][tx] = b[T*p +ty][j]; 
		__syncthreads();
		
		//computations:
		for k=0..T-1   //0..15
			//sum += a[i][ p*T + k ] * b[ p*T + k ][j];
			sum += as[ty][ k ] * bs[ k ][tx];
		__syncthreads();
	}
	c[i][j] = sum;
}
T=16
mult<<<dim3(n/T,n/T,1),dim3(T,T,1)>>>(a,b,c,n);














