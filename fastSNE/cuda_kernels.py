kernel_minMax_reduction = """
#include <stdio.h>
__global__ void perform_reduction(float *Xld, int N, int M, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N * M) {
        float old_value = Xld[idx];
        float new_value = old_value;
        if(idx*2 < N/2){
            new_value = new_value + 0.00001;
        }
        else{
            new_value = new_value - 0.00001;
        }
        Xld[idx] = new_value;
    }
}
"""


""" __device__ __forceinline__ void warpReduce1d_sum_float(volatile float* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float to_add  = vector[i + stride];
            vector[i]     += to_add;
        }
    }
}

__device__ __forceinline__ void reduce1d_sum_float(float* vector, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float to_add  = vector[i + stride];
            vector[i]    += to_add;
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_sum_float(vector, i, prev_len, stride);}
    __syncthreads();
}

__global__ void move_points(float *Xld, int N, int M, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N * M) {
        Xld[idx] += 1.0;
    }
    if(idx == 0){
        printf("Xld: %f \\n", Xld[idx]);
    }
} """