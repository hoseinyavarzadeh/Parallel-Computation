### The implementation of the bmm with CUDA!

A and B are “float” matrices of size N × N, where N = 2M. We would like to calculate C = A * B.
Parameter M should be a command line argument to the main() function. Our program works correctly for any value 10 <= M <= 13. Note that larger arrays, for example M=14, may not fit into the GPU global memory. Our program fills A and B with random float values between -8.0f and +8.0f using srand() and rand() functions.

We use the block matrix multiply algorithm. Each block computes one
square sub-matrix Csub of size TILEX×TILEY. Each thread computes one element of Csub. See
the following figure (Chapter 5 in David Kirk’s book). Parameters TILEX and TILEY should
be defined on top of bmm.cu using two #define directives. Our program works correctly
for TILEX and/or TILEY equal to 4, 8, 16 and 32, but you should tune both of them in your
code in order to gain the best possible speed. 

Note: Please do not confuse TILEX and TILEY
with TX and TY which correspond to thread index in a CUDA block.

![alt text](https://github.com/hoseinyavarzadeh/Parallel_Computing/blob/main/Block_Matrix_Multiplication(bmm)/bmm.png)

