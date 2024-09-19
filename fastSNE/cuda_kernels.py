all_the_cuda_code = """
#include <stdint.h>
#include <stdio.h>

// --------------------------------------------------------------------------------------------------
// ---------------------------------  compiler-visible constants  -----------------------------------
// --------------------------------------------------------------------------------------------------
#define MAX_PERPLEXITY (80.0f)
#define KHD ((((uint32_t)MAX_PERPLEXITY)* 3u / 32u + 1u) * 32u)
#define KLD       (32u * 1u) 
#define N_CAND_LD (32u)
#define N_CAND_HD (32u)

__global__ void get_constants(float* max_perplexity, uint32_t* khd, uint32_t* kld, uint32_t* n_cand_ld, uint32_t* n_cand_hd){
    *max_perplexity = MAX_PERPLEXITY;
    *khd            = KHD;
    *kld            = KLD;
    *n_cand_ld      = N_CAND_LD;
    *n_cand_hd      = N_CAND_HD;
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
__device__ __forceinline__ float squared_euclidean_distance(float* X_i, float* X_j, uint32_t M){
    float dist = 0.0f;
    for(uint32_t m = 0; m < M; m++){
        float diff = X_i[m] - X_j[m];
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
    while(stride > 1u){
    //while(stride > 32u){
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

// ------------------------------------  sum of vector elements ------------------------------------
__device__ __forceinline__ void warpReduce1d_argmin_float(volatile float* vector, volatile float* float_perms, uint32_t i, uint32_t prev_len, uint32_t stride){
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
                float temp = float_perms[i];
                float_perms[i] = float_perms[i + stride];
                float_perms[i + stride] = temp;
            }
        }
    }
}
__device__ __forceinline__ void reduce1d_argmin_float(float* vector, float* float_perms, uint32_t n, uint32_t i){
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
                // Swap the permutations
                float temp = float_perms[i];
                float_perms[i] = float_perms[i + stride];
                float_perms[i + stride] = temp;
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_argmin_float(vector, float_perms, i, prev_len, stride);
    }
    __syncthreads();
}



__device__ __forceinline__ void warpReduce1d_argmax_float(volatile float* vector, volatile float* float_perms, uint32_t i, uint32_t prev_len, uint32_t stride){
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
                float temp = float_perms[i];
                float_perms[i] = float_perms[i + stride];
                float_perms[i + stride] = temp;
            }
        }
    }
}

__device__ __forceinline__ void reduce1d_argmax_float(float* vector, float* float_perms, uint32_t n, uint32_t i){
    __syncthreads();
    uint32_t prev_len = 2u * n;  
    uint32_t stride   = n;     
    while(stride > 32u){ 
        prev_len = stride; 
        stride   = (uint32_t) ceilf((float)prev_len * 0.5f);
        if(i + stride < prev_len){
            float value1 = vector[i];
            float value2 = vector[i + stride];
            if(value1 < value2){
                vector[i]               = value2;
                vector[i + stride]      = value1;
                // Swap the permutations
                float temp = float_perms[i];
                float_perms[i] = float_perms[i + stride];
                float_perms[i + stride] = temp;
            }
        }
        __syncthreads();
    }
    // one warp remaining: no need to sync anymore
    if(i + stride < prev_len){
        warpReduce1d_argmax_float(vector, float_perms, i, prev_len, stride);
    }
    __syncthreads();
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



// ------------------------------------  max and permutations ------------------------------------



// --------------------------------------------------------------------------------------------------
// -------   non-overlapping random swaps: fast & helps the incremental sorting of the array   ------
// --------------------------------------------------------------------------------------------------
// assumes _K_ divisible by 2 (should be the case by design)
__device__ __forceinline__ void magicSwaps_local(float* vector, float* float_perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, bool k_divisible_by_3){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = k+1;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 < value2){
            vector[left]  = value2;
            vector[right] = value1;
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
        }
    }
}

// the seed MUST be assured to be significantly smaller than max_uint32_t else overflow is possible
__device__ __forceinline__ void magicSwaps_global(float* vector, float* float_perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, uint32_t seed){
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
        }
    }
}


__device__ __forceinline__ void magicSwaps_local_ascending(float* vector, float* float_perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, bool k_divisible_by_3){
    __syncthreads();
    if(k_divisible_by_2){ 
        uint32_t left  = k;
        uint32_t right = k+1;
        float value1 = vector[left];
        float value2 = vector[right];
        if(value1 > value2){
            vector[left]  = value2;
            vector[right] = value1;
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
        }
    }
}

// the seed MUST be assured to be significantly smaller than max_uint32_t else overflow is possible
__device__ __forceinline__ void magicSwaps_global_ascending(float* vector, float* float_perms, uint32_t k, uint32_t _K_, bool k_divisible_by_2, uint32_t seed){
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
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
            float temp = float_perms[left];
            float_perms[left]  = float_perms[right];
            float_perms[right] = temp;
        }
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

// --------------------------------------------------------------------------------------------------
// -------------------------------------  candidate neighbours  ------------------------------------------
// --------------------------------------------------------------------------------------------------
// block x : candidate number   block y : observation number
__global__ void candidates_LD_generate_and_sort(uint32_t N, uint32_t Mld, float* Xld_read, uint32_t* knn_LD_read, uint32_t* knn_HD_read, float* candidate_dists_LD, uint32_t* candidate_idx_LD, uint32_t seed_shared){
    extern __shared__ float smem_LD_candidates[];
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t cand_number    = threadIdx.x; 
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * blockDim.x;
    if(obs_i_global >= N){
        return;}
    const uint32_t lookat_rand_until     = 6u;
    const uint32_t lookat_LDneighs_until = lookat_rand_until + ((N_CAND_LD - lookat_rand_until) / 2u);
    const uint32_t lookat_HDneighs_until = N_CAND_LD;
    // ------- init  shared memory -------
    float*    X_i             = &smem_LD_candidates[obs_i_in_block * Mld];
    float*    cand_dists      = &smem_LD_candidates[n_obs_in_block * Mld + obs_i_in_block*N_CAND_LD];
    float*    float_cand_idxs = &smem_LD_candidates[n_obs_in_block * Mld + n_obs_in_block*N_CAND_LD + obs_i_in_block*N_CAND_LD];
    if(N_CAND_LD >= Mld){
        if(cand_number < Mld){
            float Xi_m = Xld_read[obs_i_global * Mld + cand_number];
            X_i[cand_number] = Xi_m;
        }
    }
    else{  
        if(cand_number == 0u){
            float* Xi_globalmem = &Xld_read[obs_i_global * Mld];
            for(uint32_t m = 0; m < Mld; m++){
                float Xi_m = Xi_globalmem[m];
                X_i[m]     = Xi_m;
            }
        }
    }
    __syncthreads();  // sync for smem
    // ------- init  a local seed -------
    uint32_t seed_local = seed_shared + (obs_i_global*N_CAND_LD)*2387u + cand_number*2u;
    random_uint32_t_xorshift32(&seed_local);
    // ------- find the index of the candidate: random index, a random neighbour of a random neighbour in LD or HD -------
    uint32_t cand_i = 0u;
    if(cand_number < 6u){ // random index
        cand_i = random_uint32_t_xorshift32(&seed_local) % N;
    }
    else if (cand_number < lookat_LDneighs_until){
        uint32_t r1 = random_uint32_t_xorshift32(&seed_local) % KLD;
        uint32_t r2 = random_uint32_t_xorshift32(&seed_local) % KLD;
        uint32_t neighbour = knn_LD_read[obs_i_global*KLD + r1];
        cand_i = knn_LD_read[neighbour*KLD + r2];
    } 
    else{
        uint32_t r1 = random_uint32_t_xorshift32(&seed_local) % KHD;
        uint32_t r2 = random_uint32_t_xorshift32(&seed_local) % KHD;
        uint32_t neighbour = knn_HD_read[obs_i_global*KHD + r1];
        cand_i = knn_HD_read[neighbour*KHD + r2];
    }
    float_cand_idxs[cand_number] = (float) cand_i;
    // ------- compute the distance to the candidate -------
    float* X_cand = &Xld_read[cand_i * Mld];
    float dist = squared_euclidean_distance(X_i, X_cand, Mld);
    cand_dists[cand_number] = dist;
    // -------  approximate sorting, ascending this time (opposite to neighbour sort)  --------
    reduce1d_argmin_float(cand_dists, float_cand_idxs, N_CAND_LD, cand_number);
    bool cand_divisible_by_2 = (cand_number % 2) == 0;
    bool cand_divisible_by_3 = (cand_number % 3) == 0;
    magicSwaps_global_ascending(cand_dists, float_cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, float_cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, cand_divisible_by_3);
    seed_shared = (seed_shared / 2u) + 209383u;
    magicSwaps_global_ascending(cand_dists, float_cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, seed_shared);
    magicSwaps_local_ascending(cand_dists, float_cand_idxs, cand_number, N_CAND_LD, cand_divisible_by_2, cand_divisible_by_3);
    // -------  write the results to global memory  --------
    __syncthreads();
    candidate_dists_LD[obs_i_global*N_CAND_LD + cand_number] = cand_dists[cand_number];
    candidate_idx_LD[obs_i_global*N_CAND_LD + cand_number]   = (uint32_t) float_cand_idxs[cand_number];
    return;
}

__global__ void candidates_LD_insert(uint32_t N, uint32_t Mld, float* Xld_read, uint32_t* knn_LD_write, float* candidate_dists_LD, uint32_t* candidate_idx_LD, uint32_t seed){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N){
        return;}
    return;
}

// --------------------------------------------------------------------------------------------------
// -------------------------------------  neighbour dists  ------------------------------------------
// --------------------------------------------------------------------------------------------------
__global__ void compute_all_LD_sqdists(uint32_t N, uint32_t Mld, float* Xld_read, uint32_t* knn_LD_read,  uint32_t* knn_LD_write, float* sqdists_LD_write, float* farthest_dist_LD_write,\
     float* simiNominators_LD_write, double* lvl1_sumsSimiNominators_LD_write, float cauchy_alpha, uint32_t seed){
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
    float* X_i                   = &smem_LD_sqdists[obs_i_in_block * Mld];
    float* smem_dists            = &smem_LD_sqdists[n_obs_in_block * Mld + obs_i_in_block*KLD];
    float* smem_floatIdxs_neighs = &smem_LD_sqdists[n_obs_in_block * Mld + n_obs_in_block*KLD + obs_i_in_block*KLD]; // it's okay

    // --------  Xi: load  Xi to shared memory (todo: make euclidean faster by prefetching to smem during loop?)  --------
    if(KLD >= Mld){ //  coalesced access to global memory: nice
        if(k < Mld){
            float Xi_m = Xld_read[obs_i_global * Mld + k]; // <--------- SLOW (global memory read)
            X_i[k]     = Xi_m;
        }
    }
    else{  // A loop to access global memory, blocking all other threads. Absolutely disgusting.
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
    smem_floatIdxs_neighs[k] = (float) j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_floatIdxs_neighs, KLD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_floatIdxs_neighs, k, KLD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_floatIdxs_neighs, k, KLD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_floatIdxs_neighs[k]; // likely different j (during parallel reduction)
    knn_LD_write[obs_i_global*KLD + k] = j;
    sqdists_LD_write[obs_i_global*KLD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_LD_write[obs_i_global] = sq_eucl;
    }

    // --------  compute the similarity in LD and save it --------
    float* smems_snoms       = &smem_LD_sqdists[obs_i_in_block * KLD];
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
    if(is_0_thread){
        lvl1_sumsSimiNominators_LD_write[obs_i_global] = (double) smems_snoms[0];
    }
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
    float* smem_floatIdxs_neighs = &smem_HD_sqdists_euclidean[n_obs_in_block * Mhd + n_obs_in_block*KHD + obs_i_in_block*KHD]; // it's okay

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
    smem_floatIdxs_neighs[k] = (float) j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_floatIdxs_neighs, KHD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_floatIdxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_HD_write[obs_i_global] = sq_eucl;
    }
    return;
}
__global__ void compute_all_HD_sqdists_manhattan(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_manhattan[];
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;

    if(obs_i_global >= N){ //  no need to check for k: sizes are adjusted on CPU to be correct
        return;}
    bool is_0_thread = (k == 0u);
    uint32_t j = knn_HD_read[obs_i_global*KHD + k]; // <--------- SLOW (global memory read)

    // --------  shared memory partition  --------
    float* X_i                   = &smem_HD_sqdists_manhattan[obs_i_in_block * Mhd];
    float* smem_dists            = &smem_HD_sqdists_manhattan[n_obs_in_block * Mhd + obs_i_in_block*KHD];
    float* smem_floatIdxs_neighs = &smem_HD_sqdists_manhattan[n_obs_in_block * Mhd + n_obs_in_block*KHD + obs_i_in_block*KHD]; // it's okay

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
    smem_floatIdxs_neighs[k] = (float) j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_floatIdxs_neighs, KHD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_floatIdxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_HD_write[obs_i_global] = sq_eucl;
    }
    return;
}
__global__ void compute_all_HD_sqdists_cosine(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_cosine[];
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;

    if(obs_i_global >= N){ //  no need to check for k: sizes are adjusted on CPU to be correct
        return;}
    bool is_0_thread = (k == 0u);
    uint32_t j = knn_HD_read[obs_i_global*KHD + k]; // <--------- SLOW (global memory read)

    // --------  shared memory partition  --------
    float* X_i                   = &smem_HD_sqdists_cosine[obs_i_in_block * Mhd];
    float* smem_dists            = &smem_HD_sqdists_cosine[n_obs_in_block * Mhd + obs_i_in_block*KHD];
    float* smem_floatIdxs_neighs = &smem_HD_sqdists_cosine[n_obs_in_block * Mhd + n_obs_in_block*KHD + obs_i_in_block*KHD]; // it's okay

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
    smem_floatIdxs_neighs[k] = (float) j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_floatIdxs_neighs, KHD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_floatIdxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_HD_write[obs_i_global] = sq_eucl;
    }
    
    return;
}

__global__ void compute_all_HD_sqdists_custom(uint32_t N, uint32_t Mhd, float* Xhd, uint32_t* knn_HD_read, uint32_t* knn_HD_write, float* sqdists_HD_write, float* farthest_dist_HD_write, uint32_t seed){
    extern __shared__ float smem_HD_sqdists_custom[];
    uint32_t n_obs_in_block = blockDim.y;
    uint32_t obs_i_in_block = threadIdx.y;
    uint32_t k              = threadIdx.x;
    uint32_t obs_i_global   = obs_i_in_block + blockIdx.x * n_obs_in_block;

    if(obs_i_global >= N){ //  no need to check for k: sizes are adjusted on CPU to be correct
        return;}
    bool is_0_thread = (k == 0u);
    uint32_t j = knn_HD_read[obs_i_global*KHD + k]; // <--------- SLOW (global memory read)

    // --------  shared memory partition  --------
    float* X_i                   = &smem_HD_sqdists_custom[obs_i_in_block * Mhd];
    float* smem_dists            = &smem_HD_sqdists_custom[n_obs_in_block * Mhd + obs_i_in_block*KHD];
    float* smem_floatIdxs_neighs = &smem_HD_sqdists_custom[n_obs_in_block * Mhd + n_obs_in_block*KHD + obs_i_in_block*KHD]; // it's okay

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
    smem_floatIdxs_neighs[k] = (float) j;
    __syncthreads();

    // --------  find the farthest distance (& agrsort~ish the array descending, for free)  --------
    reduce1d_argmax_float(smem_dists, smem_floatIdxs_neighs, KHD, k);

    // --------  sorting helper with greedy swaps at non-overlapping indices (really fast). These completely change the dynamics of successive reduce1d_argmax_float calls by breaking the patterns --------
    bool k_divisible_by_2 = (k % 2) == 0;
    bool k_divisible_by_3 = (k % 3) == 0;
    magicSwaps_global(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, seed);
    magicSwaps_local(smem_dists, smem_floatIdxs_neighs, k, KHD, k_divisible_by_2, k_divisible_by_3);

    // --------  write dists and neigbours to global memory  --------
    __syncthreads();
    sq_eucl = smem_dists[k];
    j       = (uint32_t) smem_floatIdxs_neighs[k]; // likely different j (during parallel reduction)
    knn_HD_write[obs_i_global*KHD + k] = j;
    sqdists_HD_write[obs_i_global*KHD + k] = sq_eucl;
    if(is_0_thread){ // k=0 contains the furthest dist after  reduction
        farthest_dist_HD_write[obs_i_global] = sq_eucl;
    }
    
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

__global__ void kernel_X_to_transpose(float* Xread, float* Xwrite, uint32_t N, uint32_t M){
    uint32_t obs_i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t var_i    = blockIdx.y;
    if (obs_i >= N || var_i >= M) { return; }
    uint32_t flat_i   = obs_i * M + var_i;
    float value = Xread[flat_i];
    uint32_t out_i = var_i * N + obs_i;
    Xwrite[out_i] = value;
}

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







