Implementation of parallel scan algorithm (inclusive scan) on an input array “A” with N 32-bit “float” values using the Blelloch algorithm.
N=2^M and 20 <= M <= 29. The output array “C” also has N elements. 
Our program fills the array A with random float values -2.0 <= A[i] <= +2.0

``` cpp
Compile: nvcc scan2.cu scan2_main.cu -o scan2
Execute: ./scan2 M
```
