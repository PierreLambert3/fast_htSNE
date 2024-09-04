
# gets the min and max value for each column of a 2D array of shape (M, N) (transposed X matrix)
kernel_minMax_reduction = """
// include for bool and uint32_t
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__ void warpReduce1d_minMax_float(volatile float* vector_mins, volatile float* vector_maxs, uint32_t i, uint32_t prev_len, uint32_t stride){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1_mins = vector_mins[i];
            float value2_mins = vector_mins[i + stride];
            float value1_maxs = vector_maxs[i];
            float value2_maxs = vector_maxs[i + stride];
            vector_mins[i]    = fminf(value1_mins, value2_mins);
            vector_maxs[i]    = fmaxf(value1_maxs, value2_maxs);
        }
    }
}

__device__ __forceinline__ void reduce1d_minMax_float(float* vector_mins, float* vector_maxs, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1_mins = vector_mins[i];
            float value2_mins = vector_mins[i + stride];
            float value1_maxs = vector_maxs[i];
            float value2_maxs = vector_maxs[i + stride];
            vector_mins[i]    = fminf(value1_mins, value2_mins);
            vector_maxs[i]    = fmaxf(value1_maxs, value2_maxs);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_minMax_float(vector_mins, vector_maxs, i, prev_len, stride);}
    __syncthreads();
}


// 2D grid of blocks. Grid dim1 ~ N, Grid dim2 ~ M
__global__ void perform_minMax_reduction(float *vec2d_temp_mins, float* vec2d_temp_maxs, float *vec2d_out_min, float *vec2d_out_max, uint32_t N, uint32_t M, uint32_t Nafter) {
    // init share memory
    extern __shared__ float shared_memory_all[];
    float* shared_mins = &shared_memory_all[0];
    float* shared_maxs = (float*)&shared_memory_all[blockDim.x];

    // compute indices, remove out of bounds threads
    uint32_t obs_i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t var_i    = blockIdx.y;
    uint32_t flat_i   = var_i * N + obs_i;
    if (obs_i >= N || var_i >= M) { return; }

    // 1. copy the data to shared memory for faster access
    shared_mins[threadIdx.x] = vec2d_temp_mins[flat_i];
    shared_maxs[threadIdx.x] = vec2d_temp_maxs[flat_i];
    __syncthreads();

    // 2. launch a block-wide reduction for the current variable
    reduce1d_minMax_float(shared_mins, shared_maxs, blockDim.x, threadIdx.x);

    // 3. write the result to global memory
    if(threadIdx.x == 0){
        uint32_t out_i = Nafter*var_i + blockIdx.x;
        vec2d_out_min[out_i] = shared_mins[0];
        vec2d_out_max[out_i] = shared_maxs[0];
    }
    return;
}
"""

kernel_X_to_transpose = """
__global__ void kernel_X_to_transpose(float* Xread, float* Xwrite, uint32_t N, uint32_t M){
    uint32_t obs_i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t var_i    = blockIdx.y;
    if (obs_i >= N || var_i >= M) { return; }
    uint32_t flat_i   = obs_i * M + var_i;
    float value = Xread[flat_i];
    uint32_t out_i = var_i * N + obs_i;
    Xwrite[out_i] = value;
}
"""

# mins and maxs are of shape (1, M)
# X is not transposed: shape (N, M)
kernel_scale_X = """
__global__ void kernel_scale_X(float* X, float* mins, float* maxs, uint32_t N, uint32_t M){
    extern __shared__ float shared_memory_scaling[];
    uint32_t obs_i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t var_i    = blockIdx.y;
    if (obs_i >= N || var_i >= M) { return; }

    // copy the scaling parameters to shared memory
    if(threadIdx.x == 0){
        float min_val = mins[var_i];
        float max_val = maxs[var_i];
        float range   = max_val - min_val;
        shared_memory_scaling[0] = min_val;
        shared_memory_scaling[1] = range + 0.0000001f;
    }
    __syncthreads();

    // scale the data: (-0.75 to 0.75)  and shift the data (y min = -1.0)
    float value   = X[obs_i*M + var_i];
    float min_val = shared_memory_scaling[0];
    float range   = shared_memory_scaling[1];
    float scaled  = (value - min_val) / range;
    scaled = 1.5f * scaled - 0.75f; // make it between -0.75 and 0.75
    X[obs_i*M + var_i] = scaled;
    return;
}
"""