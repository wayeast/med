#ifndef CUDA_F_H
#define CUDA_F_H

#define CUDA_H2D 0
#define CUDA_D2H 1

void call_med_kernel(
        int *,
        int,
        int *, 
        int *, 
        int *,
        int *,
        int,
        int
        );

void call_cudaMalloc(int **, int);
void call_cudaMemcpy(int *, int *, int, int);
void call_cudaFree(int *);

#endif
