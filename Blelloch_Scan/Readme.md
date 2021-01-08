## Blelloch Scan - Prefix Sum (inclusive scan)

Implementation of parallel scan algorithm (inclusive scan) on an input array “A” with N 32-bit “float” values using the Blelloch algorithm.
N=2^M and 20 <= M <= 29. The output array “C” also has N elements.\
Our program fills the array A with random float values -2.0 <= A[i] <= +2.0

``` cpp
Compile: nvcc scan2.cu scan2_main.cu -o scan2
Execute: ./scan2 M
```

Results :

![blobk matrix multiplication](https://github.com/hoseinyavarzadeh/Parallel_Computing/blob/main/Blelloch_Scan/results_cudascan2.png)

## Problem Definition:
In computer science, the prefix sum, cumulative sum, inclusive scan, or simply scan of a sequence of numbers x0, x1, x2, ... is a second sequence of numbers y0, y1, y2, ..., the sums of prefixes (running totals) of the input sequence:

```
y0 = x0
y1 = x0 + x1
y2 = x0 + x1+ x2
...
```

For instance, the prefix sums of the natural numbers are the triangular numbers:

|input numbers|	 1	| 2	| 3	| 4	| 5	| 6	|
| ----- | -- | -- | -- | -- | -- | -- |
|prefix sums	|  1	| 3	| 6	| 10| 15 |	 21	 |
