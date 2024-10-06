all_the_cuda_code = """
#include <stdint.h>
#include <stdio.h>

// --------------------------------------------------------------------------------------------------
// ---------------------------------  compiler-visible constants  -----------------------------------
// --------------------------------------------------------------------------------------------------
#define MAX_PERPLEXITY (80.0f)
//#define KHD ((((uint32_t)MAX_PERPLEXITY)* 3u / 32u + 1u) * 32u)
#define KHD ((((uint32_t)MAX_PERPLEXITY)* 3u / 32u + 1u) * 32u)
#define KLD (32u * 1u) 
#define N_CAND_LD (32u)
#define N_CAND_HD (32u)
#define N_INTERACTIONS_FAR (32u)

/*
#define N_CAND_HD (8u) <--- il faut faire N_cand beaucoup plus faible? vu que cand_R devient < 4 tres rapidement

ca donne self.candidates_HD_generate_and_sort_euclidean_cu
--> probablement le strided index: esssayer  avec indices directs
*/

__global__ void get_constants(float* max_perplexity, uint32_t* khd, uint32_t* kld, uint32_t* n_cand_ld, uint32_t* n_cand_hd, uint32_t* n_interactions_far){
    *max_perplexity = MAX_PERPLEXITY;
    *khd            = KHD;
    *kld            = KLD;
    *n_cand_ld      = N_CAND_LD;
    *n_cand_hd      = N_CAND_HD;
    *n_interactions_far = N_INTERACTIONS_FAR;
}

// --------------------------------------------------------------------------------------------------
// -----------  a terrible chaotic function that should never be used  ------------------------------
// --------------------------------------------------------------------------------------------------
// the rand_state SHOULD NEVER BE ZERO, and it's better to have it use all the 32 bits (ie: dont seed it with small values)
__device__ __forceinline__ uint32_t random_uint32_t_xorshift32(uint32_t* rand_state){
    *rand_state ^= *rand_state << 13u;
    *rand_state ^= *rand_state >> 17u;
    *rand_state ^= *rand_state << 5u;
    return *rand_state;
}

// --------------------------------------------------------------------------------------------------
// -------------------------------------  distance metrics  -----------------------------------------
// --------------------------------------------------------------------------------------------------

__device__ __forceinline__ float squared_euclidean_distance_thresholed(float* X_i, float* X_j, uint32_t M, float threshold){
    const uint32_t step_size = 6u;
    uint32_t n_steps = (M + step_size - 1u) / step_size;
    float dist = 0.0f;
    for(uint32_t step = 0u; step < n_steps; step++){
        uint32_t L = step * step_size;
        uint32_t R = min(L + step_size, M);
        for(uint32_t m = L; m < R; m++){
            float Xi_m = X_i[m];
            float Xj_m = X_j[m];
            float diff = Xi_m - Xj_m;
            float diffdiff = diff * diff;
            dist = dist + diffdiff;
        }
        if(dist > threshold){
            float estimated_dist = dist * (float) M / (float) R;
            return 2.0f * estimated_dist;
        }
    }
    return dist;
}

__device__ __forceinline__ float squared_euclidean_distance(float* X_i, float* X_j, uint32_t M){
    float dist = 0.0f;
    for(uint32_t m = 0; m < M; m++){
        float Xi_m = X_i[m];
        float Xj_m = X_j[m];
        float diff = Xi_m - Xj_m;
        float diffdiff = diff * diff;
        dist = dist + diffdiff;
    }
    return dist;
}

__device__ __forceinline__ float manhattan_distance(float* X_i, float* X_j, uint32_t M){
    float dist = 0.0f;
    for(uint32_t m = 0; m < M; m++){
        float diff    = X_i[m] - X_j[m];
        float absdiff = fabs(diff);
        dist          = dist + absdiff;
    }
    return dist;
}

__device__ __forceinline__ float cosine_distance(float* X_i, float* X_j, uint32_t M){
    float dot_product = 0.0f;
    float norm_i      = 0.0f;
    float norm_j      = 0.0f;
    for(uint32_t m = 0; m < M; m++){
        float xi = X_i[m];
        float xj = X_j[m];
        float mult_ij = xi * xj;
        float mult_ii = xi * xi;
        float mult_jj = xj * xj;
        dot_product   = dot_product + mult_ij;
        norm_i        = norm_i + mult_ii;
        norm_j        = norm_j + mult_jj;
    }
    //float denom = sqrtf(norm_i * norm_j) + 1.0f/(65536.0f * 16.0f);
    //return 1.0f - dot_product / denom;
    float inv_denom = rsqrtf(norm_i * norm_j);
    return 1.0f - dot_product * inv_denom;
}

__device__ __forceinline__ float custom_distance(float* X_i, float* X_j, uint32_t M){
    float sum_diff = 0.0f;
    float sum_sum  = 1.0f/(65536.0f * 16.0f);
    for(uint32_t m = 0; m < M; m++){
        float adiff  = fabs(X_i[m] - X_j[m]);
        float asum   = fabs(X_i[m] + X_j[m]);
        sum_diff     = sum_diff + adiff;
        sum_sum      = sum_sum + asum;
    }
    return sum_diff / sum_sum;
}

// --------------------------------------------------------------------------------------------------
// ------------------------------------  parallel reductions  ---------------------------------------
// --------------------------------------------------------------------------------------------------

// ------------------------------------  mins and max of vector ------------------------------------
__device__ __forceinline__ void warpReduce1d_minMax_float(volatile float* vector_mins, volatile float* vector_maxs, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
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
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    //while(stride > 1u){
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
        warpReduce1d_minMax_float(vector_mins, vector_maxs, i, prev_len, stride);
    }
    __syncthreads();
}

__device__ __forceinline__ void reduce1d_minMax_float_noWarp(float* vector_mins, float* vector_maxs, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
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
        __syncthreads();
    }
    __syncthreads();
}

__device__ __forceinline__ void warpReduce1d_min_uint32(volatile uint32_t* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = min(value1, value2);
        }
    }
}

__device__ __forceinline__ void reduce1d_min_uint32(uint32_t* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    //while(stride > 1u){
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = min(value1, value2);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_min_uint32(vector, i, prev_len, stride);
    }
    __syncthreads();
}

__device__ __forceinline__ void warpReduce1d_max_uint32(volatile uint32_t* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = max(value1, value2);
        }
    }
}

__device__ __forceinline__ void reduce1d_max_uint32(uint32_t* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    //while(stride > 1u){
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = max(value1, value2);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_max_uint32(vector, i, prev_len, stride);
    }
    __syncthreads();
}

// ------------------------------------  sum of vector elements ------------------------------------
__device__ __forceinline__ void warpReduce1d_argmin_float(volatile float* vector, volatile uint32_t* perms, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            if(value1 > value2){
                vector[i]               = value2;
                vector[i + stride]      = value1;
                uint32_t temp = perms[i];
                perms[i] = perms[i + stride];
                perms[i + stride] = temp;
            }
        }
    }
}
__device__ __forceinline__ void reduce1d_argmin_float(float* vector, uint32_t* perms, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;  
    uint32_t stride   = n;     
    while(stride > 32u){ 
        prev_len = stride; 
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            if(value1 > value2){
                vector[i]               = value2;
                vector[i + stride]      = value1;
                uint32_t temp = perms[i];
                perms[i] = perms[i + stride];
                perms[i + stride] = temp;
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_argmin_float(vector, perms, i, prev_len, stride);
    }
    __syncthreads();
}



__device__ __forceinline__ void warpReduce1d_argmax_float(volatile float* vector, volatile uint32_t* perms, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            if(value1 < value2){
                vector[i]               = value2;
                vector[i + stride]      = value1;
                uint32_t temp = perms[i];
                perms[i] = perms[i + stride];
                perms[i + stride] = temp;
            }
        }
    }
}

__device__ __forceinline__ void reduce1d_argmax_float(float* vector, uint32_t* perms, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;  
    uint32_t stride   = n;     
    //while(stride > 32u){ 
    while(stride > 1u){ 
        prev_len = stride; 
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            if(value1 < value2){
                vector[i]               = value2;
                vector[i + stride]      = value1;
                // Swap the permutations
                uint32_t temp = perms[i];
                perms[i] = perms[i + stride];
                perms[i + stride] = temp;
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_argmax_float(vector, perms, i, prev_len, stride);
    }
    __syncthreads();
}

__device__ __forceinline__ void warpReduce1d_min_float(volatile float* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = fminf(value1, value2);
        }
    }
}

__device__ __forceinline__ void reduce1d_min_float(float* vector, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = fminf(value1, value2);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_min_float(vector, i, prev_len, stride);
    }
}

__device__ __forceinline__ void warpReduce1d_sum_float(volatile float* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
    }
}

__device__ __forceinline__ void reduce1d_sum_float(float* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_sum_float(vector, i, prev_len, stride);
    }
    __syncthreads();
}


__device__ __forceinline__ void warpReduce1d_max_float(volatile float* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = fmaxf(value1, value2);
        }
    }
}
__device__ __forceinline__ void reduce1d_max_float(float* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            vector[i]    = fmaxf(value1, value2);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_max_float(vector, i, prev_len, stride);
    }
    __syncthreads();
}


__device__ __forceinline__ void warpReduce1d_sum_double(volatile double* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            double value1 = vector[i];
            double value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
    }
}

__device__ __forceinline__ void reduce1d_sum_double(double* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            double value1 = vector[i];
            double value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_sum_double(vector, i, prev_len, stride);}
    __syncthreads();
}

__device__ __forceinline__ void warpReduce1d_sum_uint32_t(volatile uint32_t* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
    }
}

__device__ __forceinline__ void reduce1d_sum_uint32_t(uint32_t* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i]    = value1 + value2;
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_sum_uint32_t(vector, i, prev_len, stride);
    }
    __syncthreads();
}

__device__ __forceinline__ void warpReduce1d_max_uint32_t(volatile uint32_t* vector, uint32_t i, uint32_t prev_len, uint32_t stride){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    while(stride > 1u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = max(value1, value2);
        }
    }
}

__device__ __forceinline__ void reduce1d_max_uint32_t(uint32_t* vector, uint32_t n, uint32_t i){
    // IF 2-d BLOCKS: the 1st dimension must be the one that is reduced !!!
    __syncthreads();
    uint32_t prev_len = 2u * n;
    uint32_t stride   = n;
    while(stride > 32u){
        prev_len = stride;
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            uint32_t value1 = vector[i];
            uint32_t value2 = vector[i + stride];
            vector[i] = max(value1, value2);
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_max_uint32_t(vector, i, prev_len, stride);
    }
    __syncthreads();
}



// ------------------------------------  max and permutations ------------------------------------



// --------------------------------------------------------------------------------------------------
// -------   non-overlapping random swaps: fast & helps the incremental sorting of the array   ------
// --------------------------------------------------------------------------------------------------
// assumes _K_ divisible by 2 (should be the case by design)
__device__ __forceinline__ void magicSwaps_local(float* vector, uint32_t* perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, bool k_divisible_by_3){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = k+1;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_3 && k < _K_-3){
        uint32_t left  = k;
        uint32_t right = k+2;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_2 && k > 0){ 
        uint32_t left  = k-1;
        uint32_t right = k;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
}

// the seed MUST be assured to be significantly smaller than max_uint32_t else overflow is possible
__device__ __forceinline__ void magicSwaps_global(float* vector, uint32_t* perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, uint32_t seed){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = ((seed + k/2) * 2u + 1u) % _K_; 
        if(left > right){
            uint32_t tmpidx = right;
            right = left;
            left  = tmpidx;
        }
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = (((67239+seed/2) + k/2) * 2u + 1u) % _K_;
        if(left > right){
            uint32_t tmpidx = right;
            right = left;
            left  = tmpidx;
        }
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void magicSwaps_local_ascending(float* vector, uint32_t* perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, bool k_divisible_by_3){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = k+1;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_3 && k < _K_-3){
        uint32_t left  = k;
        uint32_t right = k+2;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_2 && k > 0){ 
        uint32_t left  = k-1;
        uint32_t right = k;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
}

// the seed MUST be assured to be significantly smaller than max_uint32_t else overflow is possible
__device__ __forceinline__ void magicSwaps_global_ascending(float* vector, uint32_t* perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, uint32_t seed){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = ((seed + k/2) * 2u + 1u) % _K_; 
        if(left > right){
            uint32_t tmpidx = right;
            right = left;
            left  = tmpidx;
        }
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = (((67239+seed/2) + k/2) * 2u + 1u) % _K_;
        if(left > right){
            uint32_t tmpidx = right;
            right = left;
            left  = tmpidx;
        }
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            uint32_t temp = perms[left];
            perms[left]  = perms[right];
            perms[right] = temp;
        }
    }
}



__global__ void kernel_floatMaxReduction_one_step(float* input_vector, float* output_vector, uint32_t input_size){
    extern __shared__ float smem_floatMaxReduction_one_step[];
    uint32_t i_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i_local = threadIdx.x;
    uint32_t globalmem_idx = i_global;
    if(globalmem_idx >= input_size){
        globalmem_idx = i_local;
        if(globalmem_idx >= input_size){
            globalmem_idx = globalmem_idx / 2u;
            if(globalmem_idx >= input_size){
                globalmem_idx = 0u;
            }
        }
    }
    //smem_floatMaxReduction_one_step[i_local] = input_vector[idx];
    smem_floatMaxReduction_one_step[i_local] = input_vector[globalmem_idx];
    //smem_floatMaxReduction_one_step[i_local] = (i_global < input_size) ? input_vector[i_global] : input_vector[input_size-1];
    __syncthreads();
    reduce1d_max_float(smem_floatMaxReduction_one_step, blockDim.x, i_local);
    if(i_local == 0){
        output_vector[blockIdx.x] = smem_floatMaxReduction_one_step[0];
    }
}

__global__ void kernel_floatMinReduction_one_step(float* input_vector, float* output_vector, uint32_t input_size){
    extern __shared__ float smem_floatMinReduction_one_step[];
    uint32_t i_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i_local = threadIdx.x;
    uint32_t globalmem_idx = i_global;
    if(globalmem_idx >= input_size){
        globalmem_idx = i_local;
        if(globalmem_idx >= input_size){
            globalmem_idx = globalmem_idx / 2u;
            if(globalmem_idx >= input_size){
                globalmem_idx = 0u;
            }
        }
    }
    smem_floatMinReduction_one_step[i_local] = input_vector[globalmem_idx];
    __syncthreads();
    reduce1d_min_float(smem_floatMinReduction_one_step, blockDim.x, i_local);
    if(i_local == 0){
        output_vector[blockIdx.x] = smem_floatMinReduction_one_step[0];
    }
}



__global__ void kernel_floatSumReduction_one_step(float* input_vector, float* output_vector, uint32_t input_size){
    extern __shared__ float smem_floatSumReduction_one_step[];
    uint32_t i_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i_local = threadIdx.x;
    smem_floatSumReduction_one_step[i_local] = (i_global < input_size) ? input_vector[i_global] : 0.0f;
    __syncthreads();
    reduce1d_sum_float(smem_floatSumReduction_one_step, blockDim.x, i_local);
    if(i_local == 0){
        output_vector[blockIdx.x] = smem_floatSumReduction_one_step[0];
    }
}

__global__ void kernel_doubleSumReduction_one_step(double* input_vector, double* output_vector, uint32_t input_size){
    extern __shared__ double smem_doubleSumReduction_one_step[];
    uint32_t i_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i_local = threadIdx.x;
    smem_doubleSumReduction_one_step[i_local] = (i_global < input_size) ? input_vector[i_global] : 0.0;
    __syncthreads();
    reduce1d_sum_double(smem_doubleSumReduction_one_step, blockDim.x, i_local);
    if(i_local == 0){
        output_vector[blockIdx.x] = smem_doubleSumReduction_one_step[0];
    }
}

__global__ void kernel_uint32_tSumReduction_one_step(uint32_t* input_vector, uint32_t* output_vector, uint32_t input_size){
    extern __shared__ uint32_t smem_uint32_tSumReduction_one_step[];
    uint32_t i_global = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i_local = threadIdx.x;
    smem_uint32_tSumReduction_one_step[i_local] = (i_global < input_size) ? input_vector[i_global] : 0.0;
    __syncthreads();
    reduce1d_sum_uint32_t(smem_uint32_tSumReduction_one_step, blockDim.x, i_local);
    if(i_local == 0){
        output_vector[blockIdx.x] = smem_uint32_tSumReduction_one_step[0];
    }
}

__global__ void kernel_gradients(float exag, uint32_t do_gradients, float grad_eps, uint32_t N, uint32_t Mhd, uint32_t Mld, float lr, float cauchy_alpha, uint32_t seed, float* grad_acc_global, double* sumSnorms_rand, double* sumSnorms_neighs, float* X_nest, uint32_t* knn_HD, float* Psym, uint32_t* knn_LD, float repuls_multiplier, float denominatorLD){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t khd            = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;
    }
    bool is_main = (khd == 0);
    uint32_t seed_obs  = seed + obs_i_global;
    random_uint32_t_xorshift32(&seed_obs);
    uint32_t seed_self = seed_obs + obs_i_global*KHD + khd;
    random_uint32_t_xorshift32(&seed_self);
    extern __shared__ float _smem_GeneralGradients[];
    
    // ~~~~~~~~ shared mem init, load Xi, reset  gradients ~~~~~~~~
    float* Xi_ld =  (float*) &_smem_GeneralGradients[obs_i_in_block*Mld];
    uint32_t idx0 = n_obs_in_block*Mld;
    
    if(Mld < KHD){
        uint32_t m = khd;
        if(m < Mld){
            Xi_ld[m] = X_nest[obs_i_global * Mld + m];
        }
    }
    else{ // slow: bad config. TODO: strided parallel accesses
        if(is_main){  
            for(uint32_t m = 0u; m < Mld; m++){ // very slow
                Xi_ld[m] = X_nest[obs_i_global * Mld + m];
            }
        }
    }
    __syncthreads();

    // ~~~~~~~~ type of interaction, j_1 and j_2 ~~~~~~~~
    // |----------------j1--- K HD --------------------|  
    // |--- K LD ---|---j2------- N FAR ---------|   
    bool has_other = khd < (KLD + N_INTERACTIONS_FAR);
    bool other_is_neigh = (has_other && khd < KLD);
    uint32_t j_1 = knn_HD[obs_i_global * KHD + khd]; 
    uint32_t j_2 = 0u;
    if(has_other){
        if(other_is_neigh){
            j_2 = knn_LD[obs_i_global * KLD + khd]; // Kld = Khd
        }
        else{
            j_2 = random_uint32_t_xorshift32(&seed_self) % N;
        }
    }
    __syncthreads();

    // ~~~~~~~~~~~~~~~~ 1. compute LD similarities for j1 and j2 ~~~~~~~~~~~~~~~~
    float* Xj1 = NULL; float* Xj2 = NULL;
    float wij1 = 0.0f; float qij1 = 0.0f;
    float wij2 = 0.0f; float qij2 = 0.0f;

    // ~~~~~~~~ 1.1    j1 (HD neighbour)   ~~~~~~~~
    Xj1 = &X_nest[j_1 * Mld];
    float sq_eucl1 = squared_euclidean_distance(Xi_ld, Xj1, Mld);
    wij1 = 1.0f / __powf(1.0f + sq_eucl1/cauchy_alpha, cauchy_alpha);
    qij1 = wij1 / denominatorLD;
    
    // ~~~~~~~~ 1.2    j2  (LD neigh or rand sample)  ~~~~~~~~
    float* smem_KLD_then_FAR = (float*) &_smem_GeneralGradients[idx0 + obs_i_in_block*(KLD + N_INTERACTIONS_FAR)];
    idx0 += n_obs_in_block*(KLD + N_INTERACTIONS_FAR);
    float sq_eucl2 = 0.0f;
    if(has_other){
        Xj2 = &X_nest[j_2 * Mld];
        sq_eucl2 = squared_euclidean_distance(Xi_ld, Xj2, Mld);
        wij2 = 1.0f / __powf(1.0f + sq_eucl2/cauchy_alpha, cauchy_alpha);
        qij2 = wij2 / denominatorLD;
        smem_KLD_then_FAR[khd] = wij2;
    }
    
    // ~~~~~~~~~~~~~~~~ 2. sum the wij2 & save results to global memory ~~~~~~~~~~~~~~~~
    uint32_t idx_far = khd;
    if(has_other){
        if(other_is_neigh){
            idx_far = N_INTERACTIONS_FAR+khd;
        }
        else{
            idx_far = khd - N_INTERACTIONS_FAR;
        }
    }

    reduce1d_sum_float(&smem_KLD_then_FAR[KLD], N_INTERACTIONS_FAR, idx_far); 
    reduce1d_sum_float(smem_KLD_then_FAR, KLD, khd);
    if(is_main){
        double sum_rands  = (double) smem_KLD_then_FAR[KLD];  // verified
        double sum_neighs = (double) smem_KLD_then_FAR[0];    // verified
        sumSnorms_rand[obs_i_global]   = sum_rands;  // slow
        sumSnorms_neighs[obs_i_global] = sum_neighs; // slow
    }
    __syncthreads();


    // ~~~~~ find the max LD dist value in the KLD neighbours ~~~~~
    if(has_other && other_is_neigh){
        smem_KLD_then_FAR[khd] = sq_eucl2;
    }
    reduce1d_max_float(smem_KLD_then_FAR, KLD, khd);
    float LD_neigh_dist_threashold = smem_KLD_then_FAR[0];
    __syncthreads();

    // ~~~~~ stop after sum of LD similarities if warmup ~~~~~
    if(do_gradients < 1){
        return;
    }

    // ~~~~~~~~~~~~~~~~ 3. gradients ~~~~~~~~~~~~~~~~
    float* gradients_khds = (float*) &_smem_GeneralGradients[idx0 + obs_i_in_block*KHD];
    gradients_khds[khd] = 0.0f;
    idx0 += n_obs_in_block*KHD;
    float Pij1 = Psym[obs_i_global * KHD + khd] * exag;
    float _A_1 = 0.0f, _A_2 = 0.0f, _B_1 = 0.0f, _B_2 = 0.0f;
    float grad_prefix_1 = 0.0f, grad_prefix_2 = 0.0f;
    // ~~~~~~~~ 2.1 precompute i <--> j1/2 gradient prefix  ~~~~~~~~

//perplexité???

//il fallait faire attraction x10 et repulsion x10



// retirer double comptage KLD et KHD en calculant la max dist LD avant les gradietns

// OLD 4 LINES:
    //float attrac_multiplier = 1.0f;
    //lr *= 10.0f;
    //Pij1               *= 2.0f;
    //repuls_multiplier  *= 10.0f;

    
// NEW 4 LINES:
    float attrac_multiplier = 1.0f - repuls_multiplier;
    //lr                *= 5.0f;
    //attrac_multiplier *= 4.0f;
    //repuls_multiplier *= 20.0f;


// new new
    lr *= 1.0f;


    /*
    1/ doublons
    2/ relire vieux code et comparer les gradients et denominator
    3/ faire autre slider: LR multiplier 
        et boutton reset embedding (set as linear projection to 1) a cote de explosion
        et boutton exaggerate
    4/ perp change: il faut mettre les flags pendant 2 iterations. dans la fonction: tester de faire genre PP*=-1 pour voir si il y a bien un effet, car wtf la PP
    */

   
    // _A_1 = Pij1 * attrac_multiplier  - (qij1 * repuls_multiplier);
    if(sq_eucl1 > LD_neigh_dist_threashold){
        _A_1 = Pij1 * attrac_multiplier  - (qij1 * repuls_multiplier);
    }
    else{
        _A_1 = Pij1 * attrac_multiplier;
    }


    _B_1 = grad_eps + powf(wij1,__frcp_rn(cauchy_alpha));
    grad_prefix_1 = 4.0f * _A_1 * _B_1;
    if(has_other){
        _A_2 = (0.0f - (qij2 * repuls_multiplier));
        _B_2 = grad_eps + powf(wij2, __frcp_rn(cauchy_alpha));
        grad_prefix_2 = 4.0f * _A_2 * _B_2;
        if(!other_is_neigh){ // far sample: do as if there were (N-LKD) forces instead of nb of far samples
            //float scaling_factor = 0.5f * (float) (N-1u) / (float) N_INTERACTIONS_FAR;
            float scaling_factor = (float) (N-KLD) / (float) N_INTERACTIONS_FAR;
            grad_prefix_2 *= scaling_factor;
        }
    }

    for(uint32_t m = 0; m < Mld; m++){
        __syncthreads();
        // compute gradient for each j1 (and j2), for m-th dimension
        float xi_m  = Xi_ld[m];
        float xj1_m = Xj1[m];
        float grad1 = grad_prefix_1 * (xi_m - xj1_m);
        float grad2 = 0.0f;
        if(has_other){
            float xj2_m = Xj2[m];
            grad2 = grad_prefix_2 * (xi_m - xj2_m);
        }
        
        // perplexity change: no effect. find why, could be the reason

        grad1 *= 0.5f;
        grad2 *= 0.5f;
        gradients_khds[khd] = grad1 + grad2;
        reduce1d_sum_float(gradients_khds, KHD, khd); // sum the gradients along the KHD dimension to collapse KLD on each i
        // update other's grad_acc and khd grad_acc (more or less non-conflicting accesses)
        atomicAdd(&grad_acc_global[j_1 * Mld + m], grad1);
        if(has_other){
            atomicAdd(&grad_acc_global[j_2 * Mld + m], grad2); // needs to be of opposite sign to the one in is_main
        }
        if(is_main){
            atomicAdd(&grad_acc_global[obs_i_global * Mld + m], -gradients_khds[0]);
        }
        __syncthreads();
    }


// |----------------j1--- K HD --------------------|  
// |--- K LD ---|---j2------- N FAR ---------|  
//                 OR
// |--------------------- K HD -----------------j1-|  
// |--- K LD ---|------------ N FAR ---------|  j2 
    return;
}

__global__ void receive_gradients(uint32_t N, uint32_t M, float* grad_acc_global, float* write_Xld, float* Xld_mmtm, float lr){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    uint32_t m              = threadIdx.x;
    if(obs_i_global >= N || m >= M) { return; }
    float grad_m     = -grad_acc_global[obs_i_global * M + m] * 0.5f;
    float xi_m       = write_Xld[obs_i_global * M + m];
    float mmtm_m     = Xld_mmtm[obs_i_global * M + m];

    // update momentum
    float mmtm_alpha = 0.9f;
    float new_mmtm_m = mmtm_m * mmtm_alpha - (grad_m) * (1.0f - mmtm_alpha) * 20.0f;
    Xld_mmtm[obs_i_global * M + m] = new_mmtm_m;

    
// ok tres bien. mtmt il me faut le boutton exag et le boutton lr control: il faut un mmtm/lr plus faible, mais avec possibilité de le rendre plus fort

    // update position
    float new_xi_m = xi_m + 0.5f * lr * new_mmtm_m;
    write_Xld[obs_i_global * M + m] = new_xi_m;
    return;
}

__global__ void kernel_make_Xnesterov(uint32_t N, uint32_t M, float* grad_acc_global, float* params, float* nest, float* momenta, float lr){
    // here need a 2D grid: N x M_LD that way all the accesses are at once
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    uint32_t m              = threadIdx.x; 
    if (obs_i_global >= N || m >= M) { return; }
    float param    = params[obs_i_global * M + m];
    float momentum = momenta[obs_i_global * M + m];
    // "future" position
    nest[obs_i_global * M + m] = param + lr * momentum;
    grad_acc_global[obs_i_global * M + m] = 0.0f; // reset gradient accumulators
    return;
}

// --------------------------------------------------------------------------------------------------
// -------------------------------------  candidate neighbours  ------------------------------------------
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ void set_Pasym_andGet_entropy(uint32_t k, float ivRad, float* dists, float* Pi, float* arr_floats){
    __syncthreads();
    // ~~~~~~~ asym Pij given ivRad  ~~~~~~~
    float sq_dist = dists[k];
    float Pnom    = __expf(-sq_dist * ivRad);    

    Pnom += ((float)(KHD - k) / (float)KHD) * 0.1f / (float)KHD;

    arr_floats[k] = Pnom;
    // ~~~~~~~ 1. compute sum of Pnoms  ~~~~~~~
    reduce1d_sum_float(arr_floats, KHD, k);
    float sumP = arr_floats[0];
    if(sumP < 1e-19f){
        sumP = 1e-19f;
    }
    /*

//peut etre dist au lieu de sq dists?


    arr_floats[k] = sq_dist * Pnom / sumP;
    reduce1d_sum_float(arr_floats, KHD, k);
    if(k == 0){
        float sumPxD = arr_floats[0];
        // ~~~~~~~ 3. compute entropy  ~~~~~~~
        float entropy = __logf(sumP) + ivRad * sumPxD;
        arr_floats[0] = entropy;
    }

    __syncthreads();
    
    */
    
    
    // ~~~~~~~ 2. compute sum of (asym Pijs) x (dist)  ~~~~~~~
    float Pij = Pi[k] / sumP;   NON : Pi is empty cf l appel. REECRIRE CETTE PARTIE
    Pi[k] = Pij;                NON : Pi is empty  REECRIRE CETTE PARTIE

il faut pas utiliser Pi (qui est vide), mais utiliser Pnom

    arr_floats[k] = Pij * dists[k];
    reduce1d_sum_float(arr_floats, KHD, k);
    if(k == 0){
        float sumPxD = arr_floats[0];
        // ~~~~~~~ 3. compute entropy  ~~~~~~~
        float entropy = __logf(sumP) + ivRad * sumPxD;
        arr_floats[0] = entropy;
    }
    __syncthreads();
}

__global__ void kernel_radii_P_part2(uint32_t N, uint32_t* cuda_has_new_HD_neighs_acc, uint32_t* knn_HD_write, uint32_t* P_knn,\
     float* sqdists_HD_write, float* invRadii_HD, float* Psym, float* Pasm, float* Pasym_sums){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;
    }
    bool is_main = (k == 0);
    extern __shared__ float _2smem_radiiP[];
    // ~~~~~~~~ check if the observation has new HD neighbours  ~~~~~~~~
    uint32_t* smem_temp_uint32_t = (uint32_t*) &_2smem_radiiP[obs_i_in_block];
    if(is_main){
        smem_temp_uint32_t[0] = cuda_has_new_HD_neighs_acc[obs_i_global];
    }
    __syncthreads();
    if(smem_temp_uint32_t[0] < 1u){
        return;
    }
    smem_temp_uint32_t = NULL;
    // ~~~~~~~~ load 3 values from global mem  ~~~~~~~~
    uint32_t j   = knn_HD_write[obs_i_global * KHD + k];
    float sq_dij = sqdists_HD_write[obs_i_global * KHD + k];
    float invRad_j   = invRadii_HD[j];
    // ~~~~~~~~ fetch Pasym_sum using shared memory  ~~~~~~~~
    float* smem_temp_float = (float*) &_2smem_radiiP[obs_i_in_block];
    if(is_main){ 
        smem_temp_float[0] = Pasym_sums[obs_i_global];
    }
    __syncthreads();
    float Pasm_i_sum = smem_temp_float[0];
    float Pasm_j_sum = Pasym_sums[j];
    Pasm_j_sum = fmaxf(Pasm_j_sum, 1e-16f);
    Pasm_i_sum = fmaxf(Pasm_i_sum, 1e-16f);
    // ~~~~~~~~  symmetrized Pij  ~~~~~~~~
    float sij = Pasm[obs_i_global * KHD + k];
    float sji = __expf(-sq_dij * invRad_j);

    float Pij = sij / Pasm_i_sum;
    float Pji = sji / Pasm_j_sum;
    Pji = fminf(Pji, 1.0f); // no guarantee that Pji is < 1.0 if not mutual neighbours
    // ~~~~~~~~ save to global  ~~~~~~~~
    Psym[obs_i_global * KHD + k]  = (Pij + Pji) / (2.0f * (float)N);
    P_knn[obs_i_global * KHD + k] = j;
}

__global__ void kernel_radii_P_part1(uint32_t N, float target_perplexity, uint32_t* cuda_has_new_HD_neighs_acc, float* sqdists_HD_write,\
                                float* invRadii_HD, float* Pasm, float* Pasym_sums, uint32_t seed){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;
    }
    bool is_main = (k == 0);
    extern __shared__ float _smem_radiiP[];
    uint32_t seed_obs_i = seed + obs_i_global;
    random_uint32_t_xorshift32(&seed_obs_i);

    // ~~~~~~~~ check if the observation has new HD neighbours  ~~~~~~~~
    uint32_t* smem_temp_uint32_t = (uint32_t*) &_smem_radiiP[obs_i_in_block];
    if(is_main){
        smem_temp_uint32_t[0] = cuda_has_new_HD_neighs_acc[obs_i_global];
    }
    __syncthreads();
    if(smem_temp_uint32_t[0] < 1u){
        return;
    }
    __syncthreads();
    

    // ~~~~~~~~ load init invRadii  ~~~~~~~~
    float* smem_temp_float = (float*) &_smem_radiiP[obs_i_in_block];
    if(is_main){
        smem_temp_float[0] = invRadii_HD[obs_i_global];
    }
    __syncthreads();
    float invRadii0 = smem_temp_float[0];
    __syncthreads();
    
    // ~~~~~~~~ desired_entropy  ~~~~~~~~
    if (target_perplexity < 1.5f){
        target_perplexity = 1.5f;
    }

    target_perplexity *= 0.06f; attention à ici avec la rectification de l erreur


    float perplexity_now = 0.0f;
    if(target_perplexity < 0.1f){
        target_perplexity = 0.1f;}
    float PP_tol = 0.01f*target_perplexity;
    if(PP_tol < 0.05){
        PP_tol = 0.05;}

    float max_entropy = __logf(target_perplexity + PP_tol);
    float min_entropy = __logf(target_perplexity - PP_tol);
    float desired_entropy = __logf(target_perplexity);

    /*
    if(obs_i_global == 0 && k == 0){
        printf("min, desired, max : %.3f %.3f %.3f\\n", min_entropy, desired_entropy, max_entropy);
        printf("PPmin, PPdesired, PPmax : %.3f %.3f %.3f\\n", expf(min_entropy), expf(desired_entropy), expf(max_entropy));
    }
    */


    // ~~~~~~~~ shared memory  ~~~~~~~~
    float* smem_sqDists = &_smem_radiiP[(n_obs_in_block * KHD)*0 + obs_i_in_block * KHD];
    float* temp_pijs    = &_smem_radiiP[(n_obs_in_block * KHD)*1 + obs_i_in_block * KHD];
    float* temp_floats  = &_smem_radiiP[(n_obs_in_block * KHD)*2 + obs_i_in_block * KHD];
    smem_sqDists[k]     = sqdists_HD_write[obs_i_global * KHD + k];
    temp_pijs[k]        = 0.0f;
    __syncthreads();

    // ~~~~~~~~ binary search params  ~~~~~~~~
    float ivRad   = invRadii0;
    float L_ivRad = 0.0f;           // will be initialized if needed
    float R_ivRad = 99999999999.0f; // will be initialized if needed
    
    // ~~~~~~~~ entropy now: decrease, stay, or grow?  ~~~~~~~~
    set_Pasym_andGet_entropy(k, ivRad, smem_sqDists, temp_pijs, temp_floats);
    float entropy = temp_floats[0];
    __syncthreads();
    bool stay = entropy > min_entropy && entropy < max_entropy;
    float direction = desired_entropy - entropy;
    bool grow = !stay && direction < 0.0f;
    // ~~~~~~~~ binary search: L & R initialisation  ~~~~~~~~
    if( stay ){
        ;
    }
    else{
        // ~~~~~~~~ find the undefined bound by growing in one direction  ~~~~~~~~
        uint32_t iter_stretch = 0;
        float multiplier = grow ? 1.1f : 0.9f;
        bool H_ok = entropy > min_entropy && entropy < max_entropy;
        bool direction_changed = false;
        bool iter_limit_reached = false;
        while(!direction_changed && !H_ok && !iter_limit_reached){
            iter_stretch++;
            if(grow){
                L_ivRad = ivRad;
                R_ivRad = ivRad * multiplier;
                ivRad   = L_ivRad + 0.6f * (R_ivRad - L_ivRad);
                multiplier = multiplier * multiplier;
            }
            else{
                L_ivRad = ivRad * multiplier;
                R_ivRad = ivRad;
                ivRad   = L_ivRad + 0.4f * (R_ivRad - L_ivRad);
                multiplier = multiplier * multiplier;
            }
            if(L_ivRad < 1e-12f){
                L_ivRad = 1e-12f;
                ivRad = L_ivRad + 0.5f * (R_ivRad - L_ivRad);
                
            }
            set_Pasym_andGet_entropy(k, ivRad, smem_sqDists, temp_pijs, temp_floats);
            entropy = temp_floats[0];
            __syncthreads();
            H_ok = entropy > min_entropy && entropy < max_entropy;
            float directionnow = desired_entropy - entropy;
            bool grownow = !H_ok && directionnow < 0.0f;
            direction_changed = grownow != grow;
            iter_limit_reached = iter_stretch >= 100;
        }
        // ~~~~~~~~  binary search using the bounds  ~~~~~~~~
        if(direction_changed && (!H_ok && !iter_limit_reached)){
            uint32_t iter_binSearch = 0;
            while(!H_ok && iter_binSearch < 100){
                iter_binSearch++;
                direction = desired_entropy - entropy;
                grow = direction < 0.0f;
                if(grow){
                    L_ivRad = ivRad;
                    ivRad   = L_ivRad + 0.5f * (R_ivRad - L_ivRad);
                }
                else{
                    R_ivRad = ivRad;
                    ivRad   = L_ivRad + 0.5f * (R_ivRad - L_ivRad);
                }
                set_Pasym_andGet_entropy(k, ivRad, smem_sqDists, temp_pijs, temp_floats);
                entropy = temp_floats[0];
                __syncthreads();
                H_ok = entropy > min_entropy && entropy < max_entropy;
            }
        }
        // save the ivRad
        __syncthreads();
        if(is_main){
            invRadii_HD[obs_i_global] = ivRad;
        }
    }
    __syncthreads();

    // ~~~~~~~~  compute Pasym & sumPasm for obs i   ~~~~~~~~	
    float eucl_sq = smem_sqDists[k];
    float p_asm   = __expf(-eucl_sq * ivRad);
    temp_floats[k] = p_asm;
    reduce1d_sum_float(temp_floats, KHD, k);
    float sumPasm = temp_floats[0];
    
    // ~~~~~~~~  save Pasym & sumPasm for obs i   ~~~~~~~~
    Pasm[obs_i_global * KHD + k] = p_asm;
    if(is_main){
        Pasym_sums[obs_i_global] = sumPasm;
    }

    set_Pasym_andGet_entropy(k, ivRad, smem_sqDists, temp_pijs, temp_floats);
    entropy = temp_floats[0];
    perplexity_now = __expf(entropy);
    if(perplexity_now < target_perplexity-0.1f || perplexity_now > target_perplexity+0.1f || (obs_i_global == 0 && k == 0)){
        printf("perplexity  %f    (target %.3f)\\n", perplexity_now, target_perplexity);
    }


    return;
}

__global__ void kernel_flag_all_newNeighs(uint32_t N, uint32_t* cuda_has_new_HD_neighs, uint32_t* cuda_has_new_HD_neighs_acc){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;}
    cuda_has_new_HD_neighs[i]     = 1u;
    cuda_has_new_HD_neighs_acc[i] = 1u;
}

__global__ void kernel_HD_redetermine_farthest_dists_and_sort(uint32_t N, uint32_t Mhd, uint32_t seed_shared, uint32_t* has_new_HD_neighs, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;
    }
    extern __shared__ float smem_HD_distancesforfar[];
    uint32_t* smem_temp_uint32_t = (uint32_t*) &smem_HD_distancesforfar[obs_i_in_block];
    if(k == 0){
        smem_temp_uint32_t[0] = has_new_HD_neighs[obs_i_global];
    }
    __syncthreads();
    bool has_new_HD_neighs_obs_i = smem_temp_uint32_t[0] > 0u;
    uint32_t seed_here = seed_shared + obs_i_global;
    random_uint32_t_xorshift32(&seed_here);
    
    bool force_sort = (seed_here % 50) == 0;
    if(!has_new_HD_neighs_obs_i && !force_sort){
        return;
    }
    
    // init shared memory
    float*    smem_sqdists           = &smem_HD_distancesforfar[obs_i_in_block * KHD];
    uint32_t* smem_idxs_neighs       = (uint32_t*) &smem_HD_distancesforfar[n_obs_in_block * KHD + obs_i_in_block * KHD];
    smem_sqdists[k]     = sqdists_HD_write[obs_i_global * KHD + k];
    smem_idxs_neighs[k] = knn_HD_write[obs_i_global * KHD + k];
    __syncthreads();
    
    // sort the distances approximately
    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_sqdists, smem_idxs_neighs, KHD, k);
    __syncthreads();
    if(k == 0){ // save the farthest distance before swaps
        farthest_dist_HD_write[obs_i_global] = smem_sqdists[0];
    }

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_sqdists, smem_idxs_neighs, k, KHD, k_divisible_by_2, seed_shared);
    magicSwaps_local(smem_sqdists, smem_idxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);
    __syncthreads();

    // --------  write dists and neigbours to global memory  --------
    float sq_eucl = smem_sqdists[k];
    uint32_t j    = smem_idxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    
    return;
}

__global__ void candidates_HD_generate(uint32_t N, uint32_t Mhd, uint32_t* has_new_HD_neighs, uint32_t* has_new_HD_neighs_acc, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t* knn_LD_read, uint32_t seed_shared){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t cand_number    = threadIdx.x; 
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;}
    
    extern __shared__ float smem_HD_candidates_eucl[];
    // ------- init  shared memory -------
    uint32_t offset_smem = 0u;
    float*   X_i         = &smem_HD_candidates_eucl[offset_smem + obs_i_in_block * Mhd];
    offset_smem         += n_obs_in_block * Mhd;
    float*   cand_dists  = &smem_HD_candidates_eucl[offset_smem + obs_i_in_block * N_CAND_HD];
    offset_smem         += n_obs_in_block * N_CAND_HD;
    uint32_t* cand_idxs  = (uint32_t*) &smem_HD_candidates_eucl[offset_smem + obs_i_in_block * N_CAND_HD];
    offset_smem         += n_obs_in_block * N_CAND_HD;
    float* farthest_dist_ptr = &smem_HD_candidates_eucl[offset_smem + obs_i_in_block];
    offset_smem         += n_obs_in_block;
    uint32_t* smem_perms_retained  = (uint32_t*) &smem_HD_candidates_eucl[offset_smem + obs_i_in_block * N_CAND_HD];
    uint32_t* smem_reductionslocal = (uint32_t*) &smem_perms_retained[0];
    offset_smem              += n_obs_in_block * N_CAND_HD;
    uint32_t* smem_collisions = (uint32_t*) &smem_HD_candidates_eucl[offset_smem + obs_i_in_block*N_CAND_HD];
    
    cand_dists[cand_number] = 142u;
    cand_idxs[cand_number]   = 42u;
    smem_reductionslocal[cand_number] = 0u;
    smem_collisions[cand_number]      = 0u;
    __syncthreads();
    
    // load Xi to shared memory
    if(N_CAND_HD >= Mhd){
        if(cand_number < Mhd){
            float Xi_m = Xhd[obs_i_global * Mhd + cand_number];
            X_i[cand_number] = Xi_m;
        }
    }
    else{  // verified: ok
        const uint32_t N_steps   = (Mhd + N_CAND_HD - 1u) / N_CAND_HD;
        const uint32_t step_size = N_CAND_HD;
        for(uint32_t step = 0u; step < N_steps; step++){
            uint32_t index = step * step_size + cand_number;
            if(index < Mhd){
                float Xi_m = Xhd[obs_i_global * Mhd + index];
                X_i[index] = Xi_m;
            }
            __syncthreads();
        }
    }
    if(cand_number == 0u){
        farthest_dist_ptr[0] = farthest_dist_HD_write[obs_i_global];
    }

    __syncthreads();  // sync for smem
    // ------- init  a local seed -------
    uint32_t seed_local = seed_shared + obs_i_global + cand_number;
    random_uint32_t_xorshift32(&seed_local);
    // ------- find the index of the candidate: random index, a random neighbour of a random neighbour in LD or HD -------
    const uint32_t lookat_rand_until     = 2u;
    uint32_t cand = 0u;
    if(cand_number < lookat_rand_until){ // random index
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }
    else{ 
        uint32_t r1     = random_uint32_t_xorshift32(&seed_local) % 1024;
        bool look_in_HD = (r1 < ((1024 * 3) / 4));
        const bool biased_Ks          = true;
        const bool jump_between_HD_LD = true;
        uint32_t depth_todo = 1u;
        uint32_t rqsd = random_uint32_t_xorshift32(&seed_local);
        if((rqsd % 2) == 0){
            depth_todo = 1u;
        }
        else{
            rqsd = random_uint32_t_xorshift32(&seed_local);
            if((rqsd % 100) < 75){
                depth_todo = 2u;
            }
            else{
                depth_todo = 3u;
            }
        }
        uint32_t first_j = 0u;
        if(look_in_HD){
            uint32_t rand_k = random_uint32_t_xorshift32(&seed_local) % KHD;
            first_j = knn_HD_read[obs_i_global * KHD + rand_k];
        }
        else{
            uint32_t rand_k = random_uint32_t_xorshift32(&seed_local) % KLD;
            first_j = knn_LD_read[obs_i_global * KLD  + rand_k];
        }
        uint32_t j_from = obs_i_global;
        uint32_t j_to   = first_j;
        uint32_t depth  = 0u;
        while(depth < depth_todo){
            j_from = j_to;
            j_to   = 0u;
            uint32_t k_to = 0u;
            if(jump_between_HD_LD){
                r1 = random_uint32_t_xorshift32(&seed_local) % 1024;
                look_in_HD = (r1 < ((1024 * 4) / 5));
            }
            uint32_t  K    = (look_in_HD) ? KHD : KLD;
            uint32_t* knns = (look_in_HD) ? knn_HD_read : knn_LD_read;
            if(biased_Ks){
                uint32_t k1 = random_uint32_t_xorshift32(&seed_local) % K;
                uint32_t k2 = random_uint32_t_xorshift32(&seed_local) % K;
                k_to = max(k1, k2);
                j_to = knns[j_from * K + k_to];
            }
            else{
                k_to = random_uint32_t_xorshift32(&seed_local) % K;
                j_to = knns[j_from * K + k_to];
            }
            depth = depth + 1u;
        }
        cand = j_to;    
    }
    // ------- find the index of the candidate: random index, a random neighbour of a random neighbour in LD or HD -------
    while(cand == obs_i_global){
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }
    cand_idxs[cand_number] = cand;
    // ------- compute the distance to the candidate -------
    float* X_cand = &Xhd[cand * Mhd];
    float cand_distance = squared_euclidean_distance(X_i, X_cand, Mhd);
    cand_dists[cand_number] = cand_distance;
    __syncthreads();

    // -------  approximate sorting, ascending this time (opposite to neighbour sort)  --------
    reduce1d_argmin_float(cand_dists, cand_idxs, N_CAND_HD, cand_number);
    bool cand_divisible_by_2 = (cand_number % 2) == 0;
    bool cand_divisible_by_3 = (cand_number % 3) == 0;
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, cand_divisible_by_3);
    seed_shared = random_uint32_t_xorshift32(&seed_shared);
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, cand_divisible_by_3);
    seed_shared = random_uint32_t_xorshift32(&seed_shared);
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, cand_divisible_by_3);
    seed_shared = random_uint32_t_xorshift32(&seed_shared);
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_HD, cand_divisible_by_2, cand_divisible_by_3);

    // -------  update from permutations after reduction  -------
    __syncthreads();
    cand          = cand_idxs[cand_number];
    cand_distance = cand_dists[cand_number];
    __syncthreads();

    // ------- find "cand_R": the number of cand that are retained (there are the N_leftmost in cand_idxs) -------
    __syncthreads();
    float farthest  = farthest_dist_ptr[0u];
    uint32_t here_value = 0u;
    float dist_here = cand_distance;
    if(dist_here < farthest){
        float to_beat_1to1 = sqdists_HD_write[obs_i_global*KHD + cand_number];
        if(dist_here < to_beat_1to1){
            here_value = cand_number + 1u;
        }
    }
    smem_perms_retained[cand_number] = here_value;
    __syncthreads();
        if(cand_number > 0u){
        if(cand_number > 1u && smem_perms_retained[cand_number - 2u] == 0u){
            here_value = 0u;} 
        if(smem_perms_retained[cand_number - 1u] == 0u){
            here_value = 0u;}
    }
    __syncthreads();
    smem_perms_retained[cand_number] = here_value;
    __syncthreads();
    if((cand_number > 0u && smem_perms_retained[cand_number - 1u] == 0u) || (cand_number > 1u && smem_perms_retained[cand_number - 2u] == 0u)){
        here_value = 0u;
    }
    __syncthreads();
    smem_perms_retained[cand_number] = here_value;

    reduce1d_max_uint32(smem_perms_retained, N_CAND_HD, cand_number);
    uint32_t cand_R = smem_perms_retained[0];
    __syncthreads();
    smem_collisions[cand_number] = 0u;
    __syncthreads();

    // filter candidates where cand_dist > neighdist_at_same_idx (the cand_R upper bound is not a strict guarantee for the candidate to be closer than its associated neighbour)
    if(cand_number < cand_R){
        uint32_t cand_idx = cand_idxs[cand_number];
        float cand_dist   = cand_dists[cand_number];
        float to_beat     = sqdists_HD_write[obs_i_global*KHD + cand_number];
        if(to_beat < cand_dist){
            smem_collisions[cand_number] = 1u;        
        }
    }
    __syncthreads();

    // -------------- Collision detection --------------
    // 1. pre-load in parallel parts of KHD neighbours for each observation
    // will have a loop with a strided iteration on the neighbours, with step = N_CAND_HD. 
    // determine the loop params
    const uint32_t stride  = N_CAND_HD;
    uint32_t n_steps = (KHD + stride - 1u) / stride;
    // pre load the first strides in parallel
    uint32_t strided_8knn_HD[8u];
    if(n_steps < 8u){
        for(uint32_t step = 0u; step < n_steps; step++){
            uint32_t idx = step * stride + cand_number;
            if(idx < KHD){
                strided_8knn_HD[step] = knn_HD_write[obs_i_global*KHD + idx];
            }
        }
    }
    else{
        for(uint32_t step = 0u; step < 8u; step++){
            uint32_t idx = step * stride + cand_number;
            if(idx < KHD){
                strided_8knn_HD[step] = knn_HD_write[obs_i_global*KHD + idx];
            }
        }
    }
    __syncthreads();
    // ~~~~~~~~~~~~  2. candidates and neigbours collision detection  ~~~~~~~~~~~~
    uint32_t self_cursor = cand_number;
    uint32_t self_j      = cand_idxs[cand_number];
    bool  self_is_winner = self_cursor < cand_R;
    for(uint32_t current_cursor = 0u; current_cursor < cand_R; current_cursor++){
        __syncthreads();
        smem_reductionslocal[self_cursor] = 0u;
        uint32_t current_j = cand_idxs[current_cursor];
        // ~~~~~~  2.1. collision with other candidates  ~~~~~~~
        if(self_is_winner && self_cursor != current_cursor){
            if(self_j == current_j && self_cursor < current_cursor){
            //if(self_j == current_j){
                smem_reductionslocal[self_cursor] = 1u;
            }
        }

        // ~~~~~~  2.2. collision with neighbours  ~~~~~~
        bool hit = false;
        for(uint32_t step = 0u; step < n_steps; step++){
            uint32_t idx = step * stride + cand_number;
            uint32_t step_neigh_j = 0u;
            if(step < 8u || idx >= KHD){
                step_neigh_j = strided_8knn_HD[step];
            }
            else{
                step_neigh_j = knn_HD_write[obs_i_global*KHD + idx]; //   VERY SLOW !!! perhaps tweak the number of saved strides 
            }
            if(step_neigh_j == current_j){
                hit = true;
            }
        }
        if(hit){
            smem_reductionslocal[self_cursor] = 1u;
        }
        reduce1d_max_uint32_t(smem_reductionslocal, N_CAND_HD, self_cursor);
        if(self_cursor == current_cursor){
            bool has_collision = smem_reductionslocal[0] > 0u;
            if(has_collision){
                smem_collisions[self_cursor] = 1u;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    
    //  ~~~~~~  save the candidates that are retained to global memory  ~~~~~~
    smem_reductionslocal[cand_number] = 0u;
    if(cand_number < cand_R){
        if(!(smem_collisions[cand_number]>0u)){
            knn_HD_write[obs_i_global*KHD + cand_number]     = cand_idxs[cand_number];
            sqdists_HD_write[obs_i_global*KHD + cand_number] = cand_dists[cand_number];
            smem_reductionslocal[cand_number] = 1u; 
            /*if(obs_i_global == N-1u){
                printf("-");
            }*/
        }
    }
    //  ~~~~~~ determine if the obs had a new neighbour  ~~~~~~
    reduce1d_max_uint32_t(smem_reductionslocal, cand_R, cand_number);
    if(cand_number == 0u ){
        uint32_t reduction = smem_reductionslocal[0];
        has_new_HD_neighs[obs_i_global] = reduction > 0u ? 1u : 0u;
        if(reduction > 0u){
            has_new_HD_neighs_acc[obs_i_global] = 1u;
        }
    }
    return;
}

// block x : candidate number   block y : observation number
__global__ void candidates_LD_generate_and_sort(uint32_t N, uint32_t Mld, float* Xld_read, uint32_t* knn_LD_read, uint32_t* knn_LD_write, float* sqdists_LD_write, float* farthest_dist_LD_write, uint32_t* knn_HD_read, uint32_t seed_shared){
    extern __shared__ float smem_LD_candidates[];
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t cand_number    = threadIdx.x; 
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if(obs_i_global >= N){
        return;}
    // ------- init  shared memory -------
    uint32_t offset_smem = 0u;
    float*    X_i             = &smem_LD_candidates[offset_smem + obs_i_in_block * Mld];
    offset_smem              += n_obs_in_block * Mld;
    float*    cand_dists      = &smem_LD_candidates[offset_smem + obs_i_in_block*N_CAND_LD];
    offset_smem              += n_obs_in_block * N_CAND_LD;
    uint32_t* cand_idxs       = (uint32_t*) &smem_LD_candidates[offset_smem + obs_i_in_block*N_CAND_LD];
    offset_smem              += n_obs_in_block * N_CAND_LD;
    float*    farthest_dist   = &smem_LD_candidates[offset_smem + obs_i_in_block];
    offset_smem              += n_obs_in_block;
    uint32_t* smem_perms_retained = (uint32_t*) &smem_LD_candidates[offset_smem + obs_i_in_block*N_CAND_LD];
    uint32_t* smem_reductionslocal = (uint32_t*) &smem_perms_retained[0];
    offset_smem              += n_obs_in_block * N_CAND_LD;
    uint32_t* smem_collisions = (uint32_t*) &smem_LD_candidates[offset_smem + obs_i_in_block*N_CAND_LD];
    
    if(N_CAND_LD >= Mld){
        if(cand_number < Mld){
            float Xi_m = Xld_read[obs_i_global * Mld + cand_number];
            X_i[cand_number] = Xi_m;
        }
    }
    else{  
        const uint32_t N_steps   = (Mld + N_CAND_LD - 1u) / N_CAND_LD;
        const uint32_t step_size = N_CAND_LD;
        for(uint32_t step = 0u; step < N_steps; step++){
            uint32_t index = step * step_size + cand_number;
            if(index < Mld){
                float Xi_m = Xld_read[obs_i_global * Mld + index];
                X_i[index] = Xi_m;
            }
            __syncthreads();
        }
    }
    if(cand_number == 0u){
        farthest_dist[0u] = farthest_dist_LD_write[obs_i_global];
    }
    __syncthreads();  // sync for smem
    // ------- init  a local seed -------
    uint32_t seed_local = seed_shared + (obs_i_global*N_CAND_LD)*2387u + cand_number;
    random_uint32_t_xorshift32(&seed_local);
    // ------- find the index of the candidate: random index, a random neighbour of a random neighbour in LD or HD -------
    
    

    uint32_t cand = 0u;
    const uint32_t lookat_rand_until = 2u;
    if(cand_number < lookat_rand_until){ // random index
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }
    else{
        uint32_t r1     = random_uint32_t_xorshift32(&seed_local) % 1024;
        bool look_in_HD = (r1 < 1024 / 6);
        const bool biased_Ks    = true;
        const uint32_t maxDepth = 1u;
        uint32_t random_depth   = random_uint32_t_xorshift32(&seed_local) % 1024;
        uint32_t depth_todo     = 1u + ((random_depth * maxDepth) / 1024);
        uint32_t depth          = 0u;
        uint32_t first_j = 0u;
        if(look_in_HD){
            uint32_t rand_k = random_uint32_t_xorshift32(&seed_local) % KHD;
            first_j = knn_HD_read[obs_i_global * KHD + rand_k];
        }
        else{
            uint32_t rand_k = random_uint32_t_xorshift32(&seed_local) % KLD;
            first_j = knn_LD_read[obs_i_global * KLD  + rand_k];
        }
        uint32_t j_from = obs_i_global;
        uint32_t j_to   = first_j;
        while(depth < depth_todo){
            j_from = j_to;
            j_to   = 0u;
            uint32_t k_to = 0u;
            uint32_t  K    = (look_in_HD) ? KHD : KLD;
            uint32_t* knns = (look_in_HD) ? knn_HD_read : knn_LD_read;
            uint32_t k1 = random_uint32_t_xorshift32(&seed_local) % K;
            uint32_t k2 = random_uint32_t_xorshift32(&seed_local) % K;
            k_to = max(k1, k2);
            j_to = knns[j_from * K + k_to];
            depth = depth + 1u;
        }
        cand = j_to;
    }
    // ------- find the index of the candidate: random index, a random neighbour of a random neighbour in LD or HD -------
    while(cand == obs_i_global){
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }

    
    /*
    uint32_t cand = 0u;
    if(cand_number < lookat_rand_until){ // random index
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }
    else if (cand_number < lookat_LDneighs_until){
        uint32_t r1 = random_uint32_t_xorshift32(&seed_local) % KLD;
        uint32_t r2 = random_uint32_t_xorshift32(&seed_local) % KLD;
        uint32_t neighbour = knn_LD_read[obs_i_global*KLD + r1];
        cand = knn_LD_read[neighbour*KLD + r2];
    } 
    else{
        uint32_t r1 = random_uint32_t_xorshift32(&seed_local) % KHD;
        uint32_t r2 = random_uint32_t_xorshift32(&seed_local) % KHD;
        uint32_t neighbour = knn_HD_read[obs_i_global*KHD + r1];
        cand = knn_HD_read[neighbour*KHD + r2];
    }
    while(cand == obs_i_global){
        cand = random_uint32_t_xorshift32(&seed_local) % N;
    }
    */

    cand_idxs[cand_number] = cand;
    // ------- compute the distance to the candidate -------
    float* X_cand = &Xld_read[cand * Mld];
    float cand_distance = squared_euclidean_distance(X_i, X_cand, Mld);
    cand_dists[cand_number] = cand_distance;
    __syncthreads();
    // -------  approximate sorting, ascending this time (opposite to neighbour sort)  --------
    reduce1d_argmin_float(cand_dists, cand_idxs, N_CAND_LD, cand_number);
    bool cand_divisible_by_2 = (cand_number % 2) == 0;
    bool cand_divisible_by_3 = (cand_number % 3) == 0;
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, cand_divisible_by_3);
    seed_shared = (seed_shared / 2u) + 209383u;
    magicSwaps_global_ascending(cand_dists, cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, cand_divisible_by_3);
    // -------  update from permutations after reduction  -------
    cand          = cand_idxs[cand_number];  
    cand_distance = cand_dists[cand_number];
    // ------- find "cand_R": the number of cand that are retained (there are the N_leftmost in cand_idxs) -------
    __syncthreads();
    
    uint32_t here_value = 0u;
    float farthest  = farthest_dist[0];
    float dist_here = cand_distance;
    if(dist_here < farthest){
        float to_beat_1to1 = sqdists_LD_write[obs_i_global*KLD + cand_number];
        if(dist_here < to_beat_1to1){
            here_value = cand_number + 1u;
        }
    }
    smem_perms_retained[cand_number] = here_value;
    __syncthreads();
    if(cand_number > 0u){
        if(cand_number > 1u && smem_perms_retained[cand_number - 2u] == 0u){
            here_value = 0u;} 
        if(smem_perms_retained[cand_number - 1u] == 0u){
            here_value = 0u;}
    }
    __syncthreads();
    smem_perms_retained[cand_number] = here_value;
    __syncthreads();
    if((cand_number > 0u && smem_perms_retained[cand_number - 1u] == 0u) || (cand_number > 1u && smem_perms_retained[cand_number - 2u] == 0u)){
        here_value = 0u;
    }
    __syncthreads();
    smem_perms_retained[cand_number] = here_value;
    reduce1d_max_uint32(smem_perms_retained, N_CAND_LD, cand_number);
    uint32_t cand_R = smem_perms_retained[0];
    __syncthreads();

    smem_reductionslocal[cand_number] = 0u;
    smem_collisions[cand_number] = 0u;
    cand          = cand_idxs[cand_number];  
    cand_distance = cand_dists[cand_number];

    // pre load the neighbours at the the first 8 steps
    const uint32_t n_neighs_div_N_cand = (KLD + N_CAND_LD - 1) / N_CAND_LD;
    uint32_t first_8_steps_neighs[8];
    uint32_t min8_n_neighs_div_N_cand = min(8u, n_neighs_div_N_cand);
    for(uint32_t step = 0u; step < min8_n_neighs_div_N_cand; step++){
        uint32_t idx_k = step*N_CAND_LD + cand_number;
        if(idx_k >= KLD){
            idx_k = cand_number;}
        uint32_t knn_j = knn_LD_write[obs_i_global*KLD + idx_k]; // !!  slow  !!
        first_8_steps_neighs[step] = knn_j;
    }

    for(uint32_t working_cand_nb = 0u; working_cand_nb < cand_R; working_cand_nb++){
        __syncthreads();
        uint32_t working_j = cand_idxs[working_cand_nb];


        


        // collision within candidates
        uint32_t collision_cand = 0u;
        if(cand_number < cand_R){
            if(working_cand_nb != cand_number){
                if(cand == working_j && cand_number < working_cand_nb){
                //if(cand == working_j){
                    collision_cand = 1u;
                }
            }
        }
        // collision with the neighbours in LD
        uint32_t collision_knn  = 0u;
        for(uint32_t step = 0u; step < n_neighs_div_N_cand; step++){
            uint32_t knn_j = 0u;
            if(step < min8_n_neighs_div_N_cand){
                knn_j = first_8_steps_neighs[step];
            }
            else{
                uint32_t idx_k = step*N_CAND_LD + cand_number;
                if(idx_k >= KLD){
                    idx_k = cand_number;}
                knn_j = knn_LD_write[obs_i_global*KLD + idx_k]; // !!  slow  !!
            }
            if(knn_j == working_j){
                collision_knn = 1u;
            }
        }
        smem_reductionslocal[cand_number] = collision_cand || collision_knn;
        reduce1d_max_uint32(smem_reductionslocal, N_CAND_LD, cand_number);
        if(cand_number == working_cand_nb){
            smem_collisions[working_cand_nb] = smem_reductionslocal[0];
        }
    }
    
    // -------  insert valid candidates as neighbours -------
    __syncthreads();
    if(cand_number >= cand_R){
        return;
    }
    if(smem_collisions[cand_number] == 0u){
        knn_LD_write[obs_i_global*KLD + cand_number]     = cand;
        sqdists_LD_write[obs_i_global*KLD + cand_number] = cand_distance;
    }
    return;
}


// --------------------------------------------------------------------------------------------------
// -------------------------------------  neighbour dists  ------------------------------------------
// --------------------------------------------------------------------------------------------------
__global__ void compute_all_LD_sqdists(uint32_t N, uint32_t Mld, float* Xld_read, uint32_t* knn_LD_read,  uint32_t* knn_LD_write, float* sqdists_LD_write, float* farthest_dist_LD_write,\
     uint32_t seed){
    extern __shared__ float smem_LD_sqdists[];
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;

    if(obs_i_global >= N){ //  no need to check for k: sizes are adjusted on CPU to be correct
        return;}
    bool is_0_thread = (k == 0u);
    uint32_t j = knn_LD_read[obs_i_global*KLD + k]; // <--------- SLOW (global memory read)

    // --------  shared memory partition  --------
    uint32_t idx0 = 0u;
    float* X_i                   = &smem_LD_sqdists[idx0 + obs_i_in_block * Mld];
    idx0 += n_obs_in_block * Mld;
    float* smem_dists            = &smem_LD_sqdists[idx0 + obs_i_in_block*KLD];
    idx0 += n_obs_in_block * KLD;
    uint32_t* smem_idxs_neighs   = (uint32_t*) &smem_LD_sqdists[idx0 + obs_i_in_block*KLD];

    // --------  Xi: load  Xi to shared memory (todo: make euclidean faster by prefetching to smem during loop?)  --------
    if(KLD >= Mld){ //  coalesced access to global memory: nice
        if(k < Mld){
            float Xi_m = Xld_read[obs_i_global * Mld + k]; // <--------- SLOW (global memory read)
            X_i[k]     = Xi_m;
        }
    }
    else{  // A loop to access global memory, blocking all other threads. very bad
        if(is_0_thread){
            float* Xi_globalmem = &Xld_read[obs_i_global * Mld]; // <--------- SLOW (global memory read)
            for(uint32_t m = 0; m < Mld; m++){
                float Xi_m = Xi_globalmem[m];
                X_i[m]     = Xi_m;
            }
        }
    }
    __syncthreads();  // sync for smem 

    // -------- Xj & euclidean distance: (todo: make this efficient, pre fetch part of Xj to smem alongside the distance loop!)  --------
    float* X_j     = &Xld_read[j * Mld];  // compute the global memory address of Xj
    float  sq_eucl = squared_euclidean_distance(X_i, X_j, Mld);
    smem_dists[k]  = sq_eucl;
    smem_idxs_neighs[k] =  j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_idxs_neighs, KLD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_idxs_neighs, k, KLD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_idxs_neighs, k, KLD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = smem_idxs_neighs[k]; // likely different j (during parallel reduction)
    knn_LD_write[obs_i_global*KLD + k] = j;
    sqdists_LD_write[obs_i_global*KLD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_LD_write[obs_i_global] = sq_eucl;
    }
    
    /*
    // --------  compute the similarity in LD and save it --------
    float* smems_snoms       = (float*) &smem_LD_sqdists[obs_i_in_block*KLD];
    // TODO  : further optimisation: since we divide by bigDenom, remove the 1/... from the nominator and modify bigDenom accordingly
    // TODO2 : use double precision for bigDenom
    // float simi_nominator  = 1.0f / powf(1.0f + sq_eucl/cauchy_alpha, cauchy_alpha);
    float  simi_nominator    = 1.0f / __powf(1.0f + sq_eucl/cauchy_alpha, cauchy_alpha); 
    __syncthreads();  // necessary because same location as smem_dists which is used just before
    smems_snoms[k] = simi_nominator;
    __syncthreads();

    // --------  compute the sum on all neighbours for each point --------
    reduce1d_sum_float(smems_snoms, KLD, k);
    
    // --------  write the partial sum of nominators to global memory  --------
    __syncthreads();
    forgot to write the nominators to global memory
    if(is_0_thread){
        lvl1_sumsSimiNominators_LD_write[obs_i_global] = (double) smems_snoms[0];
    }
    */
    return;
}

__global__ void compute_all_HD_sqdists_euclidean(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_euclidean[];
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;

    if(obs_i_global >= N){ //  no need to check for k: sizes are adjusted on CPU to be correct
        return;}
    bool is_0_thread = (k == 0u);
    uint32_t j = knn_HD_read[obs_i_global*KHD + k]; // <--------- SLOW (global memory read)

    // --------  shared memory partition  --------
    float* X_i                   = &smem_HD_sqdists_euclidean[obs_i_in_block * Mhd];
    float* smem_dists            = &smem_HD_sqdists_euclidean[n_obs_in_block * Mhd + obs_i_in_block*KHD];
    uint32_t* smem_idxs_neighs = (uint32_t*) &smem_HD_sqdists_euclidean[n_obs_in_block * Mhd + n_obs_in_block*KHD + obs_i_in_block*KHD]; // it's okay

    // --------  Xi: load  Xi to shared memory (todo: make euclidean faster by prefetching to smem during loop?)  --------
    if(KHD >= Mhd){ //  coalesced access to global memory: nice
        if(k < Mhd){
            float Xi_m = Xhd[obs_i_global * Mhd + k]; // <--------- SLOW (global memory read)
            X_i[k]     = Xi_m;
        }
    }
    else{  // A loop to access global memory, blocking all other threads. Absolutely disgusting.
        if(is_0_thread){
            float* Xi_globalmem = &Xhd[obs_i_global * Mhd]; // <--------- SLOW (global memory read)
            for(uint32_t m = 0; m < Mhd; m++){
                float Xi_m = Xi_globalmem[m];
                X_i[m]     = Xi_m;
            }
        }
    }
    __syncthreads();  // sync for smem 

    // -------- Xj & euclidean distance: (todo: make this efficient, pre fetch part of Xj to smem alongside the distance loop!)  --------
    float* X_j     = &Xhd[j * Mhd];  // compute the global memory address of Xj
    float  sq_eucl = squared_euclidean_distance(X_i, X_j, Mhd);
    smem_dists[k]  = sq_eucl;
    smem_idxs_neighs[k] = j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_idxs_neighs, KHD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);
    

    seed = seed + 302743u;
    magicSwaps_global(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);

    seed = seed + 102233u;
    magicSwaps_global(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);
    /*
    seed = seed + 302743u;
    magicSwaps_global(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_idxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);
    */

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_idxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_HD_write[obs_i_global] = sq_eucl;
    }
    return;
}
__global__ void compute_all_HD_sqdists_manhattan(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_manhattan[];
    printf("remove function and doo all in one function with a if calling the appropriate metric\\n");
    return;
}
__global__ void compute_all_HD_sqdists_cosine(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_cosine[];
    printf("remove function and doo all in one function with a if calling the appropriate metric\\n");
    return;
}

__global__ void compute_all_HD_sqdists_custom(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_custom[];
    printf("remove function and doo all in one function with a if calling the appropriate metric\\n");
    return;
}




// --------------------------------------------------------------------------------------------------
// -------------------------------------  kernels for GUI  ------------------------------------------
// --------------------------------------------------------------------------------------------------
__global__ void perform_minMax_reduction(float *vec2d_temp_mins, float* vec2d_temp_maxs, float *vec2d_out_min, float *vec2d_out_max, uint32_t N, uint32_t M, uint32_t Nafter) {
    // 2D grid of blocks. Grid dim1 ~ N, Grid dim2 ~ M
    // init share memory
    extern __shared__ float shared_memory_all[];
    float* shared_mins = &shared_memory_all[0];
    float* shared_maxs = (float*)&shared_memory_all[blockDim.x];

    
    printf(" BAD FUNCTION, DELETE\\n");


    // compute indices, remove out of bounds threads
    uint32_t obs_i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t var_i    = blockIdx.y;
    uint32_t flat_i   = var_i * N + obs_i;
    if (obs_i >= N || var_i >= M) { return; }

    // non: swap for warreduce

    // 1. copy the data to shared memory for faster access
    shared_mins[threadIdx.x] = vec2d_temp_mins[flat_i];
    shared_maxs[threadIdx.x] = vec2d_temp_maxs[flat_i];
    __syncthreads();

    // 2. launch a block-wide reduction for the current variable
    reduce1d_minMax_float_noWarp(shared_mins, shared_maxs, blockDim.x, threadIdx.x);

    // 3. write the result to global memory
    if(threadIdx.x == 0){
        uint32_t out_i = Nafter*var_i + blockIdx.x;
        vec2d_out_min[out_i] = shared_mins[0];
        vec2d_out_max[out_i] = shared_maxs[0];
    }
    return;
}

__global__ void kernel_X_to_transpose(float* Xread, float* Xwrite, uint32_t N, uint32_t M){
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t m              = threadIdx.x; 
    uint32_t obs_i          = obs_i_in_block + blockIdx.x * n_obs_in_block;
    if (obs_i >= N || m >= M) { return; }
    uint32_t inp_i = obs_i * M + m;
    uint32_t out_i = m * N + obs_i;
    float value   = Xread[inp_i];
    Xwrite[out_i] = value;
}

__global__ void kernel_update_EMA_LD(float* mins_EMA, float* maxs_EMA, float* mins, float* maxs , uint32_t M){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    uint32_t m              = threadIdx.x; 
    if (obs_i_global > 0 || m >= M) { return; }
    float momentum = 0.9f;
    float min_val = mins[m];
    float max_val = maxs[m];
    float min_EMA = mins_EMA[m];
    float max_EMA = maxs_EMA[m];
    min_EMA = momentum * min_EMA + (1.0f - momentum) * min_val;
    max_EMA = momentum * max_EMA + (1.0f - momentum) * max_val;
    mins_EMA[m] = min_EMA;
    maxs_EMA[m] = max_EMA;
    return;
}

__global__ void kernel_scale_X(float* X_in, float* X_out, float min_val, float max_val, uint32_t N, uint32_t M){
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;
    uint32_t m              = threadIdx.x; 
    if (obs_i_global >= N || m >= M) { return; }

    // scale the data: (-0.75 to 0.75)  and shift the data (y min = -1.0)
    float value   = X_in[obs_i_global*M + m];
    float scaled  = (value - min_val) / (max_val - min_val);
    scaled = ((scaled - 0.5f) * 2.0f) * 0.75f;


    // X_out[obs_i_global*M + m] = scaled ;
    X_out[obs_i_global*M + m] = scaled;


    return;
}

"""







