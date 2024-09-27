import numpy as np
np.set_printoptions(linewidth=200)
__MAX_INT32_T__ = (2**31) - 1
import multiprocessing
from multiprocessing import shared_memory

# import & init pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from fastSNE.cuda_kernels import all_the_cuda_code

__DEVICE_NUMBER__ = 0 # the GPU device to use

__MIN_PERPLEXITY__ = 1.5
__MAX_KERNEL_ALPHA__ = 100.0
__MIN_KERNEL_ALPHA__ = 0.05
__MAX_ATTRACTION_MULTIPLIER__ = 10.0
__MIN_ATTRACTION_MULTIPLIER__ = 0.1



# rien ne garantie que le voisin du voisin de soit pas deja dans ses propres voisins (voisinn mutuel)
# -->  il faut pour chaque candidate faire le full check des voisins


# these are defined in the compiled side of the project (in cuda_kernels.py)
__MAX_PERPLEXITY__ = None
__Khd__       = None
__Kld__       = None
__N_CAND_LD__ = None
__N_CAND_HD__ = None

def generate_orthogonal_matrix(Mhd, Mld):
    """ random_matrix = np.random.randn(Mhd, Mld).astype(np.float32)
    Q, R = np.linalg.qr(random_matrix)
    std = np.mean(np.std(Q, axis=0)) + 1e-6
    target_std = 1.0
    out = Q * (target_std / std)
    return out """
    return np.random.randn(Mhd, Mld).astype(np.float32)

class MaxGpu:
    def __init__(self, dtype, N, compiled_cuda_code, cuda_device_attributes):
        self.dtype = dtype
        self.reduce_code = None
        if dtype == np.float32:
            self.reduce_code = compiled_cuda_code.get_function("kernel_floatMaxReduction_one_step")
        else:
            raise Exception("MaxGpu: dtype must be np.float32")
        self.N = N
        self.L_Kshapes   = []
        self.L_lvl_sizes = []
        size_at_level = N
        while size_at_level > 1:
            start_size      = size_at_level
            size_at_level   = size_at_level // cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
            if size_at_level == 0:
                size_at_level = 1
            n_threads   = start_size
            multiple_of = 32 if n_threads > 32 else 1
            smem_n_float32_per_thread = 2 
            Kshape = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, multiple_of)
            Kshape.grid_y_size = 1
            self.L_Kshapes.append(Kshape)
            self.L_lvl_sizes.append(np.uint32(start_size))
        self.lvl1_ = gpuarray.to_gpu(np.zeros(N, dtype=dtype))
        self.lvl2_ = None
        self.lvl3_ = None
        self.lvl4_ = None
        for level, shapes in enumerate(self.L_Kshapes):
            if level == 0:
                self.lvl2_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 1:
                self.lvl3_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 2:
                self.lvl4_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
        self.resultArr_async = np.zeros((1,), dtype=dtype)
    
    def get(self): # this supposes taht a stream sync was done between the last async copy and this get 
        return self.resultArr_async[0]

    def async_reduce_this(self, gpu_array_to_reduce, stream):
        # copy the array to reduce to the lvl1_ array
        cuda.memcpy_dtod_async(self.lvl1_.gpudata, gpu_array_to_reduce.gpudata, gpu_array_to_reduce.nbytes, stream)

        # compute the sum of the nominators of the LD neighbours
        self.async_reduce(stream)

    def async_reduce(self, stream):
        # compute the sum of the nominators of the LD neighbours
        n_levels = len(self.L_Kshapes)
        for level in range(n_levels):
            Kshape       = self.L_Kshapes[level]
            block_shape  = Kshape.threads_per_block, 1, 1
            grid_shape   = Kshape.grid_x_size, Kshape.grid_y_size, 1
            smem_n_bytes = Kshape.smem_n_bytes_per_block
            array_to_reduce = None
            array_result    = None
            if level == 0:
                array_to_reduce = self.lvl1_
                array_result    = self.lvl2_
                """ # cpy array_to_reduce to a new cpu array:
                cpu_array_N_float = np.zeros((self.N,), dtype=np.float32)
                stream.synchronize()
                cuda.memcpy_dtoh(cpu_array_N_float, array_to_reduce.gpudata)
                stream.synchronize()
                sum_arr = np.max(cpu_array_N_float)
                print("\nmax _arr: ", sum_arr) """
            elif level == 1:
                array_to_reduce = self.lvl2_
                array_result    = self.lvl3_
            elif level == 2:
                array_to_reduce = self.lvl3_
                array_result    = self.lvl4_
            input_size = self.L_lvl_sizes[level]
            self.reduce_code(array_to_reduce, array_result, input_size, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
            if level == n_levels - 1:
                # copy result to cpu once done
                cuda.memcpy_dtoh_async(self.resultArr_async, array_result.gpudata, stream)

class MinGpu:
    def __init__(self, dtype, N, compiled_cuda_code, cuda_device_attributes):
        self.dtype = dtype
        self.reduce_code = None
        if dtype == np.float32:
            self.reduce_code = compiled_cuda_code.get_function("kernel_floatMinReduction_one_step")
        else:
            raise Exception("MinGpu: dtype must be np.float32")
        self.N = N
        self.L_Kshapes   = []
        self.L_lvl_sizes = []
        size_at_level = N
        while size_at_level > 1:
            start_size      = size_at_level
            size_at_level   = size_at_level // cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
            if size_at_level == 0:
                size_at_level = 1
            n_threads   = start_size
            multiple_of = 32 if n_threads > 32 else 1
            smem_n_float32_per_thread = 2 
            Kshape = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, multiple_of)
            Kshape.grid_y_size = 1
            self.L_Kshapes.append(Kshape)
            self.L_lvl_sizes.append(np.uint32(start_size))
        self.lvl1_ = gpuarray.to_gpu(np.zeros(N, dtype=dtype))
        self.lvl2_ = None
        self.lvl3_ = None
        self.lvl4_ = None
        for level, shapes in enumerate(self.L_Kshapes):
            if level == 0:
                self.lvl2_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 1:
                self.lvl3_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 2:
                self.lvl4_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
        self.resultArr_async = np.zeros((1,), dtype=dtype)
    
    def get(self): # this supposes taht a stream sync was done between the last async copy and this get 
        return self.resultArr_async[0]

    def async_reduce_this(self, gpu_array_to_reduce, stream):
        # copy the array to reduce to the lvl1_ array
        cuda.memcpy_dtod_async(self.lvl1_.gpudata, gpu_array_to_reduce.gpudata, gpu_array_to_reduce.nbytes, stream)
        # compute the sum of the nominators of the LD neighbours
        self.async_reduce(stream)

    def async_reduce(self, stream):
        # compute the sum of the nominators of the LD neighbours
        n_levels = len(self.L_Kshapes)
        for level in range(n_levels):
            Kshape       = self.L_Kshapes[level]
            block_shape  = Kshape.threads_per_block, 1, 1
            grid_shape   = Kshape.grid_x_size, Kshape.grid_y_size, 1
            smem_n_bytes = Kshape.smem_n_bytes_per_block
            array_to_reduce = None
            array_result    = None
            if level == 0:
                array_to_reduce = self.lvl1_
                array_result    = self.lvl2_
                """ # cpy array_to_reduce to a new cpu array:
                cpu_array_N_float = np.zeros((self.N,), dtype=np.float32)
                stream.synchronize()
                cuda.memcpy_dtoh(cpu_array_N_float, array_to_reduce.gpudata)
                stream.synchronize()
                # sum the array
                sum_arr = np.min(cpu_array_N_float)
                print("\nmin _arr: ", sum_arr) """
                
            elif level == 1:
                array_to_reduce = self.lvl2_
                array_result    = self.lvl3_
            elif level == 2:
                array_to_reduce = self.lvl3_
                array_result    = self.lvl4_
            input_size = self.L_lvl_sizes[level]
            self.reduce_code(array_to_reduce, array_result, input_size, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
            if level == n_levels - 1:
                # copy result to cpu once done
                cuda.memcpy_dtoh_async(self.resultArr_async, array_result.gpudata, stream)
            
                

class SumGpu:
    def __init__(self, dtype, N, compiled_cuda_code, cuda_device_attributes):
        self.dtype = dtype
        self.reduce_code = None
        if dtype == np.float32:
            self.reduce_code = compiled_cuda_code.get_function("kernel_floatSumReduction_one_step")
        elif dtype == np.double:
            self.reduce_code = compiled_cuda_code.get_function("kernel_doubleSumReduction_one_step")
        elif dtype == np.uint32:
            self.reduce_code = compiled_cuda_code.get_function("kernel_uint32SumReduction_one_step")
        else:
            raise Exception("SumGpu: dtype must be np.float32 or np.double or np.uint32")
        self.N = N
        self.L_Kshapes   = []
        self.L_lvl_sizes = []
        size_at_level = N
        while size_at_level > 1:
            start_size      = size_at_level
            size_at_level   = size_at_level // cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
            if size_at_level == 0:
                size_at_level = 1
            n_threads   = start_size
            multiple_of = 32 if n_threads > 32 else 1
            smem_n_float32_per_thread = 2 # reduce using double precision: 2x32bits per item
            Kshape = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, multiple_of)
            Kshape.grid_y_size = 1
            self.L_Kshapes.append(Kshape)
            self.L_lvl_sizes.append(np.uint32(start_size))
        self.lvl1_ = gpuarray.to_gpu(np.zeros(N, dtype=dtype))
        self.lvl2_ = None
        self.lvl3_ = None
        self.lvl4_ = None
        for level, shapes in enumerate(self.L_Kshapes):
            if level == 0:
                self.lvl2_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 1:
                self.lvl3_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
            elif level == 2:
                self.lvl4_ = gpuarray.to_gpu(np.zeros(shapes.grid_x_size, dtype=dtype))
        self.resultArr_async = np.zeros((1,), dtype=dtype)
    
    def get(self): # this supposes taht a stream sync was done between the last async copy and this get 
        return self.resultArr_async[0]

    def async_reduce(self, stream):
        # compute the sum of the nominators of the LD neighbours
        n_levels = len(self.L_Kshapes)
        for level in range(n_levels):
            Kshape       = self.L_Kshapes[level]
            block_shape  = Kshape.threads_per_block, 1, 1
            grid_shape   = Kshape.grid_x_size, Kshape.grid_y_size, 1
            smem_n_bytes = Kshape.smem_n_bytes_per_block
            array_to_reduce = None
            array_result    = None
            if level == 0:
                array_to_reduce = self.lvl1_
                array_result    = self.lvl2_
                """ # cpy array_to_reduce to a new cpu array:
                cpu_array_N_doubles = np.zeros((self.N,), dtype=np.double)
                cuda.memcpy_dtoh(cpu_array_N_doubles, array_to_reduce.gpudata)
                # sync device
                stream.synchronize()
                # sum the array
                sum_arr = np.sum(cpu_array_N_doubles)
                print("\nsum_arr: ", sum_arr) """
            elif level == 1:
                array_to_reduce = self.lvl2_
                array_result    = self.lvl3_
            elif level == 2:
                array_to_reduce = self.lvl3_
                array_result    = self.lvl4_
            input_size = self.L_lvl_sizes[level]
            self.reduce_code(array_to_reduce, array_result, input_size, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
            if level == n_levels - 1:
                # copy result to cpu once done
                cuda.memcpy_dtoh_async(self.resultArr_async, array_result.gpudata, stream)

class Kernel_shapes:
    def __init__(self, N_threads_total, threads_per_block_multiple_of, smem_n_float32_per_thread, cuda_device_attributes, constant_additional_smem_n_float32):
        max_threads_per_block = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        max_shared_memory_per_block = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
        if max_shared_memory_per_block < (constant_additional_smem_n_float32 + smem_n_float32_per_thread) * np.dtype(np.float32).itemsize:
            raise Exception("Shared memory requirements too large for the GPU. Solution: reduce the dimensionality of your input (for instance, use the 50 first principal components)")
        # find the number of threads per block: start with threads_per_block_multiple_of, and add threads_per_block_multiple_of until one of the constraints is violated
        threads_per_block = threads_per_block_multiple_of
        smem_n_bytes_per_block = threads_per_block * smem_n_float32_per_thread * np.dtype(np.float32).itemsize
        n_blocks  = (N_threads_total + threads_per_block - 1) // threads_per_block
        while True:
            if threads_per_block >= N_threads_total:
                break
            next_threads_per_block      = threads_per_block + threads_per_block_multiple_of
            next_smem_n_bytes_per_block = (next_threads_per_block * smem_n_float32_per_thread + constant_additional_smem_n_float32) * np.dtype(np.float32).itemsize
            next_n_blocks               = (N_threads_total + next_threads_per_block - 1) // next_threads_per_block
            next_tpb_ok                 = next_threads_per_block <= max_threads_per_block
            next_smem_ok                = next_smem_n_bytes_per_block <= max_shared_memory_per_block
            if next_tpb_ok and next_smem_ok:
                threads_per_block = next_threads_per_block
                smem_n_bytes_per_block = next_smem_n_bytes_per_block
                n_blocks = next_n_blocks
            else:
                break 
        # save the results
        self.threads_per_block      = threads_per_block
        self.smem_n_bytes_per_block = smem_n_bytes_per_block
        self.grid_x_size            = n_blocks
        self.grid_y_size            = 1

class Kernel_shapes_2dBlocks:
    # fixed size block_x, find the optimal block_x
    def __init__(self, N_threads_total, N_threads_block_x, smem_n_float32_per_thread, cuda_device_attributes, constant_additional_smem_n_float32, smem_n_float32_per_block_y):
        max_threads_per_block = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        max_shared_memory_per_block = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
        max_block_x = cuda_device_attributes[cuda.device_attribute.MAX_BLOCK_DIM_X]
        max_block_y = cuda_device_attributes[cuda.device_attribute.MAX_BLOCK_DIM_Y]
        if max_block_x < N_threads_block_x:
            raise Exception("Kernel_shapes_2dBlocks: N_threads_block_y is too large")
        if max_shared_memory_per_block < (constant_additional_smem_n_float32 + smem_n_float32_per_thread + smem_n_float32_per_block_y) * np.dtype(np.float32).itemsize:
            raise Exception("Shared memory requirements too large for the GPU. Solution: reduce the dimensionality of your input (for instance, use the 50 first principal components)")
        # find the number of threads per block: grox block_x until one of the constraints is violated
        block_x = N_threads_block_x
        block_y = 1
        threads_per_block = block_x * block_y
        smem_n_bytes_per_block = (threads_per_block * smem_n_float32_per_thread + block_y*smem_n_float32_per_block_y) * np.dtype(np.float32).itemsize
        n_blocks  = (N_threads_total + threads_per_block - 1) // threads_per_block
        while True:
            if threads_per_block >= N_threads_total:
                break
            next_block_y = block_y + 1
            next_threads_per_block = next_block_y * block_x
            next_smem_n_bytes_per_block = (next_threads_per_block * smem_n_float32_per_thread + constant_additional_smem_n_float32 + block_y*smem_n_float32_per_block_y) * np.dtype(np.float32).itemsize
            next_n_blocks = (N_threads_total + next_threads_per_block - 1) // next_threads_per_block
            next_tpb_ok     = next_threads_per_block <= max_threads_per_block
            next_smem_ok    = next_smem_n_bytes_per_block <= max_shared_memory_per_block
            next_block_y_ok = next_block_y <= max_block_y
            if next_tpb_ok and next_smem_ok and next_block_y_ok:
                block_y = next_block_y
                threads_per_block = next_threads_per_block
                smem_n_bytes_per_block = next_smem_n_bytes_per_block
                n_blocks = next_n_blocks
            else:
                break
        # save the results
        self.block_x = block_x
        self.block_y = block_y
        self.threads_per_block = threads_per_block
        self.smem_n_bytes_per_block = smem_n_bytes_per_block
        self.grid_x_size = n_blocks
        self.grid_y_size = 1
    
    def print(self):
        print("block_x: ", self.block_x, " block_y: ", self.block_y)
        print("threads_per_block: ", self.threads_per_block)
        print("smem_n_bytes_per_block: ", self.smem_n_bytes_per_block)
        print("grid_x_size: ", self.grid_x_size, " grid_y_size: ", self.grid_y_size)



class fastSNE:    
    def __init__(self, with_GUI, n_components=2, random_state=None):
        # compiling the cuda code
        compiler_options = ["-O3", "--use_fast_math", "-prec-div=false", "-ftz=true", "-prec-sqrt=false", "-fmad=true"] # safe arithmetics are for the weak
        self.compiled_cuda_code = SourceModule(all_the_cuda_code, options=compiler_options)
        # fetch the cuda-defined constants! (defined in cuda for better compilation optimisations)
        self.fetch_constants_from_cuda()
        if (__Kld__ % 32) != 0:
            raise Exception("\033[38;2;255;165;0mWARNING\033[0m:  __Kld__ is not a multiple of 32. This will result in inefficient memory access patterns. Consider changing the value of __Kld__ in fastSNE.py")
        if (__Khd__ % 32) != 0:
            raise Exception("\033[38;2;255;165;0mWARNING\033[0m:  __Khd__ is not a multiple of 32. This will result in inefficient memory access patterns. Consider changing the value of __Khd__ in fastSNE.py")
        if (__Kld__ % 2) != 0:
            raise Exception("__Kld__ has to be a multiple of 2 (and preferably a multiple of 32 as well). Change the value of __Kld__ in fastSNE.py")
        if (__Khd__ % 2) != 0:
            raise Exception("__Khd__ has to be a multiple of 2 (and preferably a multiple of 32 as well). Change the value of __Khd__ in fastSNE.py")
        # state variables
        self.with_GUI     = with_GUI
        self.is_fitted    = False
        self.random_state = random_state
        if self.random_state is not None and not self.random_state > 0:
            raise Exception("fastSNE: random_state must be a strictly positive integer")
        # dataset and hyperparameters
        self.N            = None
        self.Mhd          = None
        self.Mld          = n_components
        if self.Mld > __Kld__:
            print("\033[38;2;255;165;0mWARNING\033[0m:  n_components is larger than the number of neighbours in LD (the constant __Kld__ in fastSNE.py). This will result in very inefficient memory access patterns, increasing __Kld__ might be worthwile. (but make sure __Kld__ is a multiple of 32, for efficiency reasons (see CUDA warps if you're curious why))")
        assert self.Mld >= 2
        self.kern_alpha   = np.float32(1.0)
        self.perplexity   = np.float32(5.0)
        self.attrac_mult  = np.float32(1.0)
        self.dist_metric  = 0
        assert self.dist_metric in [0, 1, 2, 3]
        assert self.kern_alpha < __MAX_KERNEL_ALPHA__ and self.kern_alpha > __MIN_KERNEL_ALPHA__
        assert self.perplexity < __MAX_PERPLEXITY__ and self.perplexity > __MIN_PERPLEXITY__
        assert self.attrac_mult < __MAX_ATTRACTION_MULTIPLIER__ and self.attrac_mult > __MIN_ATTRACTION_MULTIPLIER__
        # result
        self.cpu_Xld  = None
    
    def fit(self, N, M, Xhd, Y=None):
        # check yourself 
        if N < 5:
            raise Exception("fastSNE: the number of samples N must be at least 2")
        if M < 2:
            raise Exception("fastSNE: the number of dimensions M must be at least 2")
        if np.isnan(Xhd).any():
            raise Exception("fastSNE: the high-dimensional data contains NaNs")
        if __Khd__ >= (N/2-1):
            raise Exception("fastSNE: the number of neighbours K is too large for the number of samples N (reducting __MAX_PERPLEXITY__ should do the trick)")
        if __N_CAND_LD__ < 32:
            raise Exception("fastSNE: __N_CAND_LD__ must be at least 32 (and preferably a multiple of 32). Change the values of __N_CAND_LD__ and __N_CAND_HD__ in cuda_kernels.py")
        if __N_CAND_HD__ < 16:
            raise Exception("fastSNE: __N_CAND_HD__ must be at least 16 (and preferably a multiple of 32). Change the values of __N_CAND_LD__ and __N_CAND_HD__ in cuda_kernels.py")
        if __N_CAND_LD__ > __Kld__:
            raise Exception("fastSNE: __N_CAND_LD__ must be smaller than __Kld__ (the number of neighbours in LD). Change the value of __N_CAND_LD__ in cuda_kernels.py")
        if __N_CAND_HD__ > __Khd__:
            raise Exception("fastSNE: __N_CAND_HD__ must be smaller than __Khd__ (the number of neighbours in HD). Change the value of __N_CAND_HD__ in cuda_kernels.py")

        # on CPU
        self.N        = N
        self.Mhd      = M
        self.Xhd      = Xhd
        # project Xhd linearly to init Xld
        self.linear_projection_now    = generate_orthogonal_matrix(self.Mhd, self.Mld)
        self.linear_projection_target = generate_orthogonal_matrix(self.Mhd, self.Mld)
        self.cpu_Xld = np.dot(Xhd, self.linear_projection_now).astype(np.float32)
        # determine grid shapes, block shapes, smem size for each CUDA kernels & compile kernels
        self.configue_and_initialise_CUDA_kernels_please(__Khd__, __Kld__, self.Mhd, self.Mld, device_number=__DEVICE_NUMBER__)
        # cuda streams
        stream_minMax   = cuda.Stream() # used in GUI mode
        stream_neigh_HD = cuda.Stream()
        stream_neigh_LD = cuda.Stream()
        stream_grads    = cuda.Stream()
        all_streams = [stream_minMax, stream_neigh_HD , stream_neigh_LD, stream_grads]
        # init KNN indices quickly
        init_LD_neighs = np.zeros((N, __Kld__), dtype=np.uint32)
        init_HD_neighs = np.zeros((N, __Khd__), dtype=np.uint32)
        for i in range(N):
            # LD neighbours: windows on the right
            left_bound1  = i + 1
            right_bound1 = left_bound1 + __Kld__
            if right_bound1 >= N:
                left_bound1  = 0
                right_bound1 = __Kld__
            init_LD_neighs[i] = np.arange(left_bound1, right_bound1)
            # HD neighbours : windows on the left
            right_bound2  = i-1 
            left_bound2   = right_bound2 - __Khd__
            if left_bound2 < 0:
                right_bound2 = N-1
                left_bound2  = right_bound2 - __Khd__
            init_HD_neighs[i] = np.arange(left_bound2, right_bound2)

        # mallocs on the device
        cuda_Xhd                = gpuarray.to_gpu_async(Xhd)
        cuda_knn_HD_A           = gpuarray.to_gpu(init_HD_neighs)
        cuda_knn_HD_B           = gpuarray.to_gpu(init_HD_neighs)
        cuda_sqdists_HD_A       = gpuarray.to_gpu(np.zeros((N, __Khd__), dtype=np.float32))  
        cuda_sqdists_HD_B       = gpuarray.to_gpu(np.zeros((N, __Khd__), dtype=np.float32)) 
        cuda_farthest_dist_HD_A = gpuarray.to_gpu(np.ones(N, dtype=np.float32))             
        cuda_farthest_dist_HD_B = gpuarray.to_gpu(np.ones(N, dtype=np.float32))            

        # things for HD similarities: ind. radii, Pasm, Psym, sumsPasym
        cuda_has_new_HD_neighs_A = gpuarray.to_gpu(np.ones(N, dtype=np.uint32)) # todo: make this a bool* or uint8_t*
        cuda_has_new_HD_neighs_B = gpuarray.to_gpu(np.ones(N, dtype=np.uint32)) # todo: make this a bool* or uint8_t*
        cuda_invRadii_HD       = gpuarray.to_gpu(np.ones(N, dtype=np.float32)) # only used on update: no need of double buffering
        cuda_Pasm              = gpuarray.to_gpu(np.zeros((N, __Khd__), dtype=np.float32)) 
        cuda_Pasm_sums         = gpuarray.to_gpu(np.ones((N,), dtype=np.float32)) 
        cuda_Psym              = gpuarray.to_gpu(np.ones((N, __Khd__), dtype=np.float32)) 
        cuda_Psym_knn          = gpuarray.to_gpu(np.ones((N, __Khd__), dtype=np.uint32)) 
        # each times a point gets new HD neighbours: recompute HD radius and HD similarities for the point

        cuda_Xld_true_A           = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_true_B           = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_nest             = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_mmtm             = gpuarray.to_gpu(np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype))
        cuda_knn_LD_A             = gpuarray.to_gpu(init_LD_neighs)
        cuda_knn_LD_B             = gpuarray.to_gpu(init_LD_neighs)
        cuda_sqdists_LD_A         = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_sqdists_LD_B         = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_simiNominators_LD_A  = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_simiNominators_LD_B  = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_farthest_dist_LD_A   = gpuarray.to_gpu(np.ones(N, dtype=np.float32))             # TODO: init this
        cuda_farthest_dist_LD_B   = gpuarray.to_gpu(np.ones(N, dtype=np.float32))             # TODO: init this
        neighbours_sumSnorms_LD = SumGpu(np.double, self.N, self.compiled_cuda_code, cuda.Device(__DEVICE_NUMBER__).get_attributes())


        big_dic = {
            "cuda_Xhd"                : cuda_Xhd,
            "cuda_knn_HD_A"           : cuda_knn_HD_A,
            "cuda_knn_HD_B"           : cuda_knn_HD_B,
            "cuda_sqdists_HD_A"       : cuda_sqdists_HD_A,
            "cuda_sqdists_HD_B"       : cuda_sqdists_HD_B,
            "cuda_farthest_dist_HD_A" : cuda_farthest_dist_HD_A,
            "cuda_farthest_dist_HD_B" : cuda_farthest_dist_HD_B,
            "cuda_Xld_true_A"         : cuda_Xld_true_A,
            "cuda_Xld_true_B"         : cuda_Xld_true_B,
            "cuda_Xld_nest"           : cuda_Xld_nest,
            "cuda_Xld_mmtm"           : cuda_Xld_mmtm,
            "cuda_knn_LD_A"           : cuda_knn_LD_A,
            "cuda_knn_LD_B"           : cuda_knn_LD_B,
            "cuda_sqdists_LD_A"       : cuda_sqdists_LD_A,
            "cuda_sqdists_LD_B"       : cuda_sqdists_LD_B,
            "cuda_simiNominators_LD_A": cuda_simiNominators_LD_A,
            "cuda_simiNominators_LD_B": cuda_simiNominators_LD_B,
            "neighbours_sumSnorms_LD" : neighbours_sumSnorms_LD,
            "cuda_farthest_dist_LD_A" : cuda_farthest_dist_LD_A,
            "cuda_farthest_dist_LD_B" : cuda_farthest_dist_LD_B,
            "cuda_has_new_HD_neighs_A"  : cuda_has_new_HD_neighs_A,
            "cuda_has_new_HD_neighs_B"  : cuda_has_new_HD_neighs_B,
            "cuda_invRadii_HD"        : cuda_invRadii_HD,
            "cuda_Pasm"               : cuda_Pasm,
            "cuda_Pasm_sums"          : cuda_Pasm_sums,
            "cuda_Psym"               : cuda_Psym,
            "cuda_Psym_knn"           : cuda_Psym_knn,
            "all_streams"             : all_streams
        }
        self.periodic_1000 = 0
        # launch the tSNE optimisation
        if self.with_GUI:
            self.fit_with_gui(Y, big_dic)
        else:
            self.fit_without_gui(big_dic)
        self.Xhd = None
        self.is_fitted = True

    def transform(self):
        if not self.is_fitted:
            raise Exception("fastSNE: transform() called before fit(), or fit failed crashingly")
        # return self.cpu_Xld
        return None

    def fit_with_gui(self, Y, big_dic):
        # fetch from the big dictionary
        cuda_Xhd, cuda_Xld_mmtm, cuda_Xld_nest = [big_dic[key] for key in ["cuda_Xhd", "cuda_Xld_mmtm", "cuda_Xld_nest"]] 
        cuda_knn_HD_A, cuda_sqdists_HD_A, cuda_farthest_dist_HD_A, cuda_Xld_true_A = [big_dic[key] for key in ["cuda_knn_HD_A", "cuda_sqdists_HD_A", "cuda_farthest_dist_HD_A", "cuda_Xld_true_A"]]
        cuda_knn_HD_B, cuda_sqdists_HD_B, cuda_farthest_dist_HD_B, cuda_Xld_true_B = [big_dic[key] for key in ["cuda_knn_HD_B", "cuda_sqdists_HD_B", "cuda_farthest_dist_HD_B", "cuda_Xld_true_B"]]
        cuda_knn_LD_A, cuda_sqdists_LD_A, cuda_farthest_dist_LD_A = [big_dic[key] for key in ["cuda_knn_LD_A", "cuda_sqdists_LD_A", "cuda_farthest_dist_LD_A"]]
        cuda_knn_LD_B, cuda_sqdists_LD_B, cuda_farthest_dist_LD_B = [big_dic[key] for key in ["cuda_knn_LD_B", "cuda_sqdists_LD_B", "cuda_farthest_dist_LD_B"]]
        cuda_has_new_HD_neighs_A, cuda_has_new_HD_neighs_B, cuda_invRadii_HD, cuda_Pasm, cuda_Pasm_sums, cuda_Psym, cuda_Psym_knn = [big_dic[key] for key in ["cuda_has_new_HD_neighs_A", "cuda_has_new_HD_neighs_B", "cuda_invRadii_HD", "cuda_Pasm", "cuda_Pasm_sums", "cuda_Psym", "cuda_Psym_knn"]]
        simiNominators_LD_A = big_dic["cuda_simiNominators_LD_A"]
        simiNominators_LD_B = big_dic["cuda_simiNominators_LD_B"]
        all_streams = big_dic["all_streams"]
        stream_minMax, stream_neigh_HD , stream_neigh_LD, stream_grads = all_streams
        read_Xld      = cuda_Xld_true_A
        write_Xld     = cuda_Xld_true_B
        knn_HD_read   = cuda_knn_HD_A
        knn_HD_write  = cuda_knn_HD_B
        sqdists_HD_read = cuda_sqdists_HD_A
        sqdists_HD_write = cuda_sqdists_HD_B
        farthest_dist_HD_read = cuda_farthest_dist_HD_A
        farthest_dist_HD_write = cuda_farthest_dist_HD_B
        knn_LD_read   = cuda_knn_LD_A   
        knn_LD_write  = cuda_knn_LD_B
        sqdists_LD_read = cuda_sqdists_LD_A
        sqdists_LD_write = cuda_sqdists_LD_B
        farthest_dist_LD_read = cuda_farthest_dist_LD_A
        farthest_dist_LD_write = cuda_farthest_dist_LD_B
        neighbours_sumSnorms_LD = big_dic["neighbours_sumSnorms_LD"]
        simiNominators_LD_read = simiNominators_LD_A
        simiNominators_LD_write = simiNominators_LD_B
        has_new_HD_neighs_read = cuda_has_new_HD_neighs_A
        has_new_HD_neighs_write = cuda_has_new_HD_neighs_B

        self.gui_Xld_minFinder = MinGpu(np.float32, self.N*self.Mld, self.compiled_cuda_code, cuda.Device(__DEVICE_NUMBER__).get_attributes())
        self.gui_Xld_maxFinder = MaxGpu(np.float32, self.N*self.Mld, self.compiled_cuda_code, cuda.Device(__DEVICE_NUMBER__).get_attributes())
        self.cpu_double_scalar = np.zeros((1,), dtype=np.double)

        # init neighbours dists , fartherst dists, and simiNominators
        self.low_dim_updateSim_and_refineKNN(cuda_Xld_true_A, cuda_knn_LD_A, cuda_knn_HD_A, cuda_knn_LD_B, cuda_sqdists_LD_B, cuda_farthest_dist_LD_B, simiNominators_LD_B, neighbours_sumSnorms_LD, self.kern_alpha, stream_neigh_LD)
        self.low_dim_updateSim_and_refineKNN(cuda_Xld_true_B, cuda_knn_LD_B, cuda_knn_HD_B, cuda_knn_LD_A, cuda_sqdists_LD_A, cuda_farthest_dist_LD_A, simiNominators_LD_A, neighbours_sumSnorms_LD, self.kern_alpha, stream_neigh_LD)
        self.fill_all_sqdists_HD(cuda_Xhd, cuda_knn_HD_A, cuda_knn_HD_B, cuda_sqdists_HD_B, cuda_farthest_dist_HD_B, stream_neigh_HD)
        self.fill_all_sqdists_HD(cuda_Xhd, cuda_knn_HD_B, cuda_knn_HD_A, cuda_sqdists_HD_A, cuda_farthest_dist_HD_A, stream_neigh_HD)
        stream_neigh_HD.synchronize()
        stream_neigh_LD.synchronize()


        # 1. configure the process launch mode 
        multiprocessing.set_start_method('spawn') # this is crucial for the GUI to work correctly. Python is wierd and often annoying

        # 2. shared memory with GUI (on CPU)
        cpu_shared_mem      = shared_memory.SharedMemory(create=True, size=int(self.N * self.Mld * np.dtype(np.float32).itemsize))
        cpu_Xld_arr_on_smem = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem.buf)
        # copy (GPU->CPU) cuda_Xld_true_A to shared memory
        cuda_Xld_true_A.get(cpu_Xld_arr_on_smem)
        # temp structures related to preprocessing the data for the GUI
        cuda_Xld_temp_Xld         = gpuarray.to_gpu(np.zeros((self.N, self.Mld), dtype=np.float32))

        # 3.   Launching the process responsible for the GUI
        from fastSNE.fastSNE_gui import gui_worker
        #  Shared hyperparameters
        kernel_alpha   = multiprocessing.Value('f', self.kern_alpha)
        perplexity     = multiprocessing.Value('f', self.perplexity)
        attrac_mult    = multiprocessing.Value('f', self.attrac_mult)
        dist_metric    = multiprocessing.Value('i', self.dist_metric)
        # MDS_strength   = multiprocessing.Value('f', self.MDS_strength) # TODO next: incorporate MDS gradietns
        #  Shared state variables
        gui_closed                 = multiprocessing.Value('b', False)
        points_ready_for_rendering = multiprocessing.Value('b', False)
        points_rendering_finished  = multiprocessing.Value('b', True)
        iteration                  = multiprocessing.Value('i', 0)
        explosion_please           = multiprocessing.Value('b', False) 
        # 3.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem, Y, self.N, self.Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please, __MIN_PERPLEXITY__, __MAX_PERPLEXITY__, __MIN_KERNEL_ALPHA__, __MAX_KERNEL_ALPHA__, __MIN_ATTRACTION_MULTIPLIER__, __MAX_ATTRACTION_MULTIPLIER__))
        process_gui.start()

        # 4.   Optimise until the GUI is closed
        iteration_int         = 0
        isPhaseA              = True
        gui_data_prep_phase   = 0
        busy_copying__for_GUI = False
        gui_was_closed        = False
        warmup                = True
        while not gui_was_closed:
            '''
                -  when inserting: parallel reduction on abs(cand_idx - neigh_idx) --> if min value is 0 then neighbour is already there
                -  neighbours: much slower than gradients (about 45 iter per second for K=256)  --->  DO 4 GRADIENT UPDATES PER NEIGHBOUR KERNEL
                -  this means that phase change is only one in 4!!
            '''
            # One iteration of the tSNE optimisation
            if isPhaseA:
                read_Xld      = cuda_Xld_true_A
                write_Xld     = cuda_Xld_true_B
                knn_HD_read   = cuda_knn_HD_A
                knn_HD_write  = cuda_knn_HD_B
                sqdists_HD_read = cuda_sqdists_HD_A
                sqdists_HD_write = cuda_sqdists_HD_B
                farthest_dist_HD_read = cuda_farthest_dist_HD_A
                farthest_dist_HD_write = cuda_farthest_dist_HD_B
                knn_LD_read   = cuda_knn_LD_A   
                knn_LD_write  = cuda_knn_LD_B
                sqdists_LD_read = cuda_sqdists_LD_A
                sqdists_LD_write = cuda_sqdists_LD_B
                farthest_dist_LD_read = cuda_farthest_dist_LD_A
                farthest_dist_LD_write = cuda_farthest_dist_LD_B
                simiNominators_LD_read = simiNominators_LD_A
                simiNominators_LD_write = simiNominators_LD_B
                has_new_HD_neighs_read = cuda_has_new_HD_neighs_A
                has_new_HD_neighs_write = cuda_has_new_HD_neighs_B
            else:
                read_Xld     = cuda_Xld_true_B
                write_Xld    = cuda_Xld_true_A
                knn_HD_read  = cuda_knn_HD_B
                knn_HD_write = cuda_knn_HD_A
                sqdists_HD_read = cuda_sqdists_HD_B
                sqdists_HD_write = cuda_sqdists_HD_A
                farthest_dist_HD_read = cuda_farthest_dist_HD_B
                farthest_dist_HD_write = cuda_farthest_dist_HD_A
                knn_LD_read  = cuda_knn_LD_B
                knn_LD_write = cuda_knn_LD_A
                sqdists_LD_read = cuda_sqdists_LD_B
                sqdists_LD_write = cuda_sqdists_LD_A
                farthest_dist_LD_read = cuda_farthest_dist_LD_B
                farthest_dist_LD_write = cuda_farthest_dist_LD_A
                simiNominators_LD_read = simiNominators_LD_B
                simiNominators_LD_write = simiNominators_LD_A
                has_new_HD_neighs_read = cuda_has_new_HD_neighs_B
                has_new_HD_neighs_write = cuda_has_new_HD_neighs_A

            # sync all streams (else read/writes will conflict with versions A and B)
            stream_neigh_LD.synchronize()
            stream_minMax.synchronize()
            stream_grads.synchronize()
            stream_neigh_HD.synchronize()
            denom_LD_neighbours = float(neighbours_sumSnorms_LD.get())
            # print("neighbours denoms part: ",  denom_LD_neighbours)

            # update hyperparameters (if HD config changed: recompute distances to neighbours & farthest distances)
            self.kern_alpha   = np.float32(kernel_alpha.value)
            self.attrac_mult  = attrac_mult.value
            new_perplexity    = perplexity.value
            new_dist_metric   = dist_metric.value
            HD_config_changed = (new_perplexity != self.perplexity or new_dist_metric != self.dist_metric)
            self.perplexity   = new_perplexity
            self.dist_metric  = new_dist_metric

            if HD_config_changed:
                self.fill_all_sqdists_HD(cuda_Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, stream_neigh_HD)
                self.fill_all_sqdists_HD(cuda_Xhd, knn_HD_write, knn_HD_read, sqdists_HD_read, farthest_dist_HD_read, stream_neigh_HD)
                stream_neigh_HD.synchronize()

            # if not isPhaseA:
            self.one_iteration(warmup, cuda_Xhd, read_Xld, write_Xld, cuda_Xld_nest, cuda_Xld_mmtm, knn_HD_read, knn_HD_write, sqdists_HD_read, sqdists_HD_write, farthest_dist_HD_read, farthest_dist_HD_write, knn_LD_read, knn_LD_write, sqdists_LD_read, sqdists_LD_write, farthest_dist_LD_read, farthest_dist_LD_write, stream_neigh_HD, stream_neigh_LD, simiNominators_LD_write, neighbours_sumSnorms_LD, stream_grads, has_new_HD_neighs_write, cuda_invRadii_HD, cuda_Pasm, cuda_Pasm_sums, cuda_Psym, cuda_Psym_knn)

            # GUI communication & preparation of the data for the GUI
            if gui_data_prep_phase == 0: # copy cuda_Xld_true_A/B to cuda_Xld_temp in an async manner using stream_minMax*
                # if we were copying the data for the GUI, notify the GUI that the data is ready
                if busy_copying__for_GUI:
                    busy_copying__for_GUI = False
                    with points_rendering_finished.get_lock():
                        points_rendering_finished.value = False
                    with points_ready_for_rendering.get_lock():
                        points_ready_for_rendering.value = True
                self.gui_Xld_minFinder.async_reduce_this(gpu_array_to_reduce = read_Xld, stream=stream_minMax)
                self.gui_Xld_maxFinder.async_reduce_this(gpu_array_to_reduce = read_Xld, stream=stream_minMax)
            elif gui_data_prep_phase == 1: # perform the min-max reduction on cuda_Xld_temp, & scale the data to [0, 1] with the results
                self.scaling_of_points(read_Xld, cuda_Xld_temp_Xld, stream_minMax)
                gui_done = False
                busy_copying__for_GUI = False
                with points_rendering_finished.get_lock():
                    gui_done = points_rendering_finished.value
                if gui_done:
                    cuda_Xld_temp_Xld.get_async(stream=stream_minMax, ary=cpu_Xld_arr_on_smem)
                    busy_copying__for_GUI = True
                    iteration.value = iteration_int
            gui_data_prep_phase = (gui_data_prep_phase + 1) % 2

            isPhaseA = not isPhaseA  # ONLY CHANGE EVERY 4 ITERATION (because 4 grads per neigh update)
            iteration_int += 1

            if warmup and iteration_int >= 40000:
                warmup = False
            with gui_closed.get_lock():
                gui_was_closed = gui_closed.value
        process_gui.join()
        cpu_shared_mem.unlink()
        self.free_all_GPU_memory(cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        return
    
    def fit_without_gui(self, big_dic):
        1/0

    # all CUDA 'kernels' run in parallel, sync at the start of the iterations loop outside of this function
    def one_iteration(self, warmup, Xhd, read_Xld, write_Xld, Xld_nest, Xld_mmtm, knn_HD_read, knn_HD_write, sqdists_HD_read, sqdists_HD_write,\
                     farthest_dist_HD_read, farthest_dist_HD_write,  knn_LD_read, knn_LD_write, sqdists_LD_read, sqdists_LD_write, farthest_dist_LD_read, farthest_dist_LD_write, stream_neigh_HD, stream_neigh_LD, simiNominators_LD_write, neighbours_sumSnorms_LD, stream_grads,\
                      has_new_HD_neighs_write, invRadii_HD, Pasm, Pasym_sums, Psym, Psym_knn):
        self.periodic_1000 = (self.periodic_1000 + 1) % 1000
        # 0/ - gradient computations & update positions
        alpha = 0.01
        if warmup:
            if (self.periodic_1000 % 85) == 0:
                self.linear_projection_target = generate_orthogonal_matrix(self.Mhd, self.Mld)
            elif (self.periodic_1000 % 1) == 0:
                self.linear_projection_now = (1.0 - alpha) * self.linear_projection_now + alpha * self.linear_projection_target
                self.cpu_Xld = np.dot(self.Xhd, self.linear_projection_now).astype(np.float32)
                write_Xld.set_async(self.cpu_Xld, stream=stream_grads)
                
        # else:
        #     print("launch gradients kernel")
        
        # 1/ - first update LD neighbour dists & simiNominators
        #    - then refine KNN in LD space
        #if (self.periodic_1000 % 3) == 0:
        self.low_dim_updateSim_and_refineKNN(read_Xld, knn_LD_read, knn_HD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, neighbours_sumSnorms_LD, self.kern_alpha, stream_neigh_LD)

        # 2/ or 3/ :
        #     - find candidate neighbours for HD space, compute their distances on insertion recompute farthest distances
        # OR  - points with new HD neighbours: recompute HD similarities & local HD state (invRadii, Pasym, Pasym_sums)
        self.high_dim_refineKNN(Xhd, knn_HD_read, knn_HD_write, knn_LD_read, sqdists_HD_write, farthest_dist_HD_write, stream_neigh_HD, has_new_HD_neighs_write)
        # if (self.periodic_1000  % 11) == 0:
        #     self.high_dim_filtered_updateHDstates_and_Psym(Xhd, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, has_new_HD_neighs_write, invRadii_HD, Pasm, Pasym_sums, Psym, Psym_knn, stream_neigh_HD)
            
            # stream_neigh_HD.synchronize()
            # cpu_knn_HD_write = knn_HD_write.get()
            # print("knn_HD_write: ", cpu_knn_HD_write[0][:4])
            # stream_neigh_HD.synchronize()
            # print()
        

        if (self.periodic_1000 % 61) == 0 or self.periodic_1000 < 3:
            # stream_neigh_LD.synchronize()
            # stream_grads.synchronize()
            # verify_neighdists(read_Xld, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, self.N, self.Mld, __Kld__, stream_neigh_LD)
            stream_neigh_HD.synchronize()
            stream_grads.synchronize()
            print()
            verify_neighdists(Xhd, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, self.N, self.Mhd, __Khd__, stream_neigh_HD)


        # 5/  - accumulated percent of changed is high engough: update HD similarities
        #     - for each i that has a 'HD neigh changed' flag activated (which is persistant across iterations):
        #     - recompute radius and Pasym 
        #     - reset 'HD neigh changed' flags
        return

    # def scaling_of_points(self, cuda_Xld_temp_Xld, cuda_Xld_temp_lvl1_mins, cuda_Xld_temp_lvl1_maxs, stream_minMax):
    def scaling_of_points(self, Xld_read, Xld_scaled, stream_minMax):
        stream_minMax.synchronize()
        global_min = (self.gui_Xld_minFinder.get())
        global_max = (self.gui_Xld_maxFinder.get())
        block_shape  = self.Kshapes_transpose.block_x, self.Kshapes_transpose.block_y, 1
        grid_shape   = self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size, 1
        smem_n_bytes = self.Kshapes_transpose.smem_n_bytes_per_block
        self.scaling_X_cu(Xld_read, Xld_scaled, np.float32(global_min), np.float32(global_max), np.uint32(self.N), np.uint32(self.Mld),\
                            block=block_shape, grid=grid_shape, stream=stream_minMax, shared=smem_n_bytes)
    def fill_all_sqdists_HD(self, Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, stream):
        kernel = None 
        if self.dist_metric == 0:
            kernel = self.all_HD_sqdists_euclidean_cu
        elif self.dist_metric == 1:
            kernel = self.all_HD_sqdists_manhattan_cu
        elif self.dist_metric == 2:
            kernel = self.all_HD_sqdists_cosine_cu
        else:
            kernel = self.all_HD_sqdists_custom_cu
        block_shape  = self.Kshapes2d_NxKhd_threads.block_x, self.Kshapes2d_NxKhd_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKhd_threads.grid_x_size, self.Kshapes2d_NxKhd_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKhd_threads.smem_n_bytes_per_block
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
        kernel(np.uint32(self.N), np.uint32(self.Mhd), Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
        print("this should only be called once at init")
        # verify_neighdists(Xhd, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, self.N, self.Mhd, __Khd__, stream)

    def low_dim_updateSim_and_refineKNN(self, Xld_read, knn_LD_read, knn_HD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, neighbours_sumSnorms_LD, cauchy_alpha, stream):
        # 1.  squared dists to LD neighbours, sort neighbours, find farthest dists
        #     compute similarity nominators and first reduction on them for each i
        block_shape  = self.Kshapes2d_NxKld_threads.block_x, self.Kshapes2d_NxKld_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKld_threads.grid_x_size, self.Kshapes2d_NxKld_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKld_threads.smem_n_bytes_per_block
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3)
        self.all_LD_sqdists_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, neighbours_sumSnorms_LD.lvl1_, cauchy_alpha, seed, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # 2. candidate neighbours: generate, compute dists, and partial sort
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3)
        block_shape = self.Kshapes2d_NxNcandLD_threads.block_x, self.Kshapes2d_NxNcandLD_threads.block_y, 1
        grid_shape  = self.Kshapes2d_NxNcandLD_threads.grid_x_size, self.Kshapes2d_NxNcandLD_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxNcandLD_threads.smem_n_bytes_per_block
        self.candidates_LD_generate_and_sort_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, knn_HD_read, seed, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # 4. finishing the similarity nominators reduction, with multi-level reductions
        neighbours_sumSnorms_LD.async_reduce(stream)
    
    def high_dim_refineKNN(self, Xhd, knn_HD_read, knn_HD_write, knn_LD_read, sqdist_HD_write, farthest_dist_HD_write, stream, has_new_HD_neighs):
        # 1.  candidate neighbours: generate, compute dists, and partial sort
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) 
        block_shape  = self.Kshapes2d_NxNcandHD_threads.block_x, self.Kshapes2d_NxNcandHD_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxNcandHD_threads.grid_x_size, self.Kshapes2d_NxNcandHD_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxNcandHD_threads.smem_n_bytes_per_block
        self.candidates_HD_generate_and_sort_euclidean_cu(np.uint32(self.N), np.uint32(self.Mhd), has_new_HD_neighs, Xhd, knn_HD_read, knn_HD_write, sqdist_HD_write, farthest_dist_HD_write, knn_LD_read, seed, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # print("only doing euclidean distance for now here")

    def high_dim_filtered_updateHDstates_and_Psym(self, Xhd, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, has_new_HD_neighs, invRadii_HD, Pasm, Pasym_sums, Psym, Psym_knn, stream_neigh_HD):
        # 1.  points with new HD neighbours: recompute HD similarities & local HD state (invRadii, Pasym, Pasym_sums)
        # 2.  save knn_HD_write as Psym_knn
        # print("only doing euclidean distance for now here")
        # 2.  save knn_HD_write as Psym_knn
        return


    def configue_and_initialise_CUDA_kernels_please(self, Khd, Kld, Mhd, Mld, device_number=0):
        N = self.N
        cuda_device = cuda.Device(device_number)
        cuda_device_attributes = cuda_device.get_attributes()

        # ------------ 0. kernels used for getting the transpose of Xld  -------
        n_threads   = N * Mld
        multiple_of = 32 if n_threads > 32 else 1
        block_x     = Mld
        smem_n_float32_per_thread = 0 
        smem_n_float_per_y = 0
        smem_n_float_const = 0
        self.Kshapes_transpose = Kernel_shapes_2dBlocks(n_threads, block_x, smem_n_float32_per_thread, cuda_device_attributes, smem_n_float_const, smem_n_float_per_y)
       
        # ------------ 2. kernels used for neighboru related things -------
        #   N threads
        n_threads = N
        multiple_of = 32 if n_threads > 32 else 1
        smem_n_float32_per_thread = 1
        self.Kshapes_N_threads = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, Mhd)
        self.Kshapes_N_threads.grid_y_size = 1 
        #   N x Kld threads , 1d grid, 2d block
        n_threads   = N * Kld
        block_x     = Kld
        smem_n_float32_per_thread = 2 
        smem_n_float_per_y = Mld
        smem_n_float_const = 1
        self.Kshapes2d_NxKld_threads = Kernel_shapes_2dBlocks(n_threads, block_x, smem_n_float32_per_thread, cuda_device_attributes, smem_n_float_const, smem_n_float_per_y)
        #   N x Khd threads , 1d grid, 2d block
        n_threads   = N * Khd
        block_x     = Khd
        smem_n_float32_per_thread = 2 
        smem_n_float_per_y = Mhd
        smem_n_float_const = 0
        self.Kshapes2d_NxKhd_threads = Kernel_shapes_2dBlocks(n_threads, block_x, smem_n_float32_per_thread, cuda_device_attributes, smem_n_float_const, smem_n_float_per_y)

        #  N x __N_CAND_LD__ threads, 1d grid, 2d block
        n_threads   = N * __N_CAND_LD__
        block_x     = __N_CAND_LD__
        smem_n_float32_per_thread = 4
        smem_n_float_per_y = Mld + 1
        smem_n_float_const = 0
        self.Kshapes2d_NxNcandLD_threads = Kernel_shapes_2dBlocks(n_threads, block_x, smem_n_float32_per_thread, cuda_device_attributes, smem_n_float_const, smem_n_float_per_y)

        #  N x __N_CAND_HD__ threads, 1d grid, 2d block
        n_threads   = N * __N_CAND_HD__
        block_x     = __N_CAND_HD__
        smem_n_float32_per_thread = 4 
        smem_n_float_per_y = Mhd + 1
        smem_n_float_const = 0
        self.Kshapes2d_NxNcandHD_threads = Kernel_shapes_2dBlocks(n_threads, block_x, smem_n_float32_per_thread, cuda_device_attributes, smem_n_float_const, smem_n_float_per_y)
        # ------------ 3. kernels used for the tSNE gradients -------   
        
        # ------------ 4. fetching the CUDA kernels -------
        self.min_max_reduction_cu  = self.compiled_cuda_code.get_function("perform_minMax_reduction")
        self.X_to_transpose_cu     = self.compiled_cuda_code.get_function("kernel_X_to_transpose")
        self.scaling_X_cu          = self.compiled_cuda_code.get_function("kernel_scale_X")
        self.update_EMA_LD = self.compiled_cuda_code.get_function("kernel_update_EMA_LD")
        self.all_HD_sqdists_euclidean_cu = self.compiled_cuda_code.get_function("compute_all_HD_sqdists_euclidean")
        self.all_HD_sqdists_manhattan_cu = self.compiled_cuda_code.get_function("compute_all_HD_sqdists_manhattan")
        self.all_HD_sqdists_cosine_cu    = self.compiled_cuda_code.get_function("compute_all_HD_sqdists_cosine")
        self.all_HD_sqdists_custom_cu    = self.compiled_cuda_code.get_function("compute_all_HD_sqdists_custom")
        self.all_LD_sqdists_cu           = self.compiled_cuda_code.get_function("compute_all_LD_sqdists")
        self.candidates_LD_generate_and_sort_cu = self.compiled_cuda_code.get_function("candidates_LD_generate_and_sort")
        self.candidates_HD_generate_and_sort_euclidean_cu = self.compiled_cuda_code.get_function("candidates_HD_generate_and_sort_euclidean")
        self.kernel_doubleSumReduction_one_step = self.compiled_cuda_code.get_function("kernel_doubleSumReduction_one_step")
        self.kernel_floatSumReduction_one_step = self.compiled_cuda_code.get_function("kernel_floatSumReduction_one_step")
        self.update_HD_sim_and_local_state_euclidean_cu = self.compiled_cuda_code.get_function("update_HD_sim_and_local_state_euclidean")
        self.compiled_cuda_code.get_function("kernel_floatMaxReduction_one_step")
        self.compiled_cuda_code.get_function("kernel_floatMinReduction_one_step")

    def free_all_GPU_memory(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # cuda_context.pop() # not needed if pycuda.autoinit is used
        cuda_Xhd.gpudata.free()
        cuda_Xld_true_A.gpudata.free()
        cuda_Xld_true_B.gpudata.free()
        cuda_Xld_nest.gpudata.free()
        cuda_Xld_mmtm.gpudata.free()
        raise Exception("here need to free all CUDA ressources!!")

    def fetch_constants_from_cuda(self):
        global __MAX_PERPLEXITY__, __Khd__, __Kld__, __N_CAND_LD__, __N_CAND_HD__
        # cuda: get_constants(float* max_perplexity, uint32_t* khd, uint32_t* kld, uint32_t* n_cand_ld, uint32_t* n_cand_hd)
        # Allocate memory on the GPU for the constants
        max_perplexity_gpu = cuda.mem_alloc(np.float32().nbytes)
        khd_gpu = cuda.mem_alloc(np.uint32().nbytes)
        kld_gpu = cuda.mem_alloc(np.uint32().nbytes)
        n_cand_ld_gpu = cuda.mem_alloc(np.uint32().nbytes)
        n_cand_hd_gpu = cuda.mem_alloc(np.uint32().nbytes)
        # fetch data on GPU
        self.compiled_cuda_code.get_function("get_constants")(max_perplexity_gpu, khd_gpu, kld_gpu, n_cand_ld_gpu, n_cand_hd_gpu, block=(1, 1, 1), grid=(1, 1))
        cuda.Context.synchronize()
        max_perplexity = np.empty(1, dtype=np.float32)
        khd = np.empty(1, dtype=np.uint32)
        kld = np.empty(1, dtype=np.uint32)
        n_cand_ld = np.empty(1, dtype=np.uint32)
        n_cand_hd = np.empty(1, dtype=np.uint32)
        # Copy the values from the GPU to the CPU
        cuda.memcpy_dtoh(max_perplexity, max_perplexity_gpu)
        cuda.memcpy_dtoh(khd, khd_gpu)
        cuda.memcpy_dtoh(kld, kld_gpu)
        cuda.memcpy_dtoh(n_cand_ld, n_cand_ld_gpu)
        cuda.memcpy_dtoh(n_cand_hd, n_cand_hd_gpu)
        # Free the memory on the GPU
        max_perplexity_gpu.free()
        khd_gpu.free()
        kld_gpu.free()
        n_cand_ld_gpu.free()
        n_cand_hd_gpu.free()
        # save to "constants" on cpu
        __MAX_PERPLEXITY__ = float(max_perplexity[0])
        __Khd__ = int(khd[0])
        __Kld__ = int(kld[0])
        __N_CAND_LD__ = int(n_cand_ld[0])
        __N_CAND_HD__ = int(n_cand_hd[0])

    
    def testing_neighdists_LD(self, cuda_Xld_true_A, cuda_knn_LD_A, cuda_knn_LD_B, cuda_sqdists_LD_B, cuda_farthest_dist_LD_B, cuda_Xld_true_B, cuda_sqdists_LD_A, cuda_farthest_dist_LD_A, cuda_simiNominators_LD_A, cuda_simiNominators_LD_B, cuda_lvl1_sumSimiNominators_LD, stream_neigh_LD):
        # cuda_Xld_true_A, cuda_knn_LD_A, cuda_knn_LD_B, cuda_sqdists_LD_B, cuda_farthest_dist_LD_B, stream_neigh_LD

        block_shape  = self.Kshapes2d_NxKld_threads.block_x, self.Kshapes2d_NxKld_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKld_threads.grid_x_size, self.Kshapes2d_NxKld_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKld_threads.smem_n_bytes_per_block
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
        Xld_read = cuda_Xld_true_A
        knn_LD_read = cuda_knn_LD_A
        knn_LD_write = cuda_knn_LD_B
        sqdists_LD_write = cuda_sqdists_LD_B
        farthest_dist_LD_write = cuda_farthest_dist_LD_B
        simiNominators_LD_write = cuda_simiNominators_LD_B
        self.all_LD_sqdists_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, cuda_lvl1_sumSimiNominators_LD, self.kern_alpha, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_LD, shared=smem_n_bytes)
        verify_neighdists(Xld_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, self.N, self.Mld, __Kld__, stream_neigh_LD)
        
        import time
        niter = 2
        start = time.time()
        for i in range(niter):
            seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
            Xld_read = cuda_Xld_true_B
            knn_LD_read = cuda_knn_LD_B
            knn_LD_write = cuda_knn_LD_A
            sqdists_LD_write = cuda_sqdists_LD_A
            farthest_dist_LD_write = cuda_farthest_dist_LD_A
            simiNominators_LD_write = cuda_simiNominators_LD_A
            self.all_LD_sqdists_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, cuda_lvl1_sumSimiNominators_LD, self.kern_alpha, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_LD, shared=smem_n_bytes)
            seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
            Xld_read = cuda_Xld_true_A
            knn_LD_read = cuda_knn_LD_A
            knn_LD_write = cuda_knn_LD_B
            sqdists_LD_write = cuda_sqdists_LD_B
            farthest_dist_LD_write = cuda_farthest_dist_LD_B
            simiNominators_LD_write = cuda_simiNominators_LD_B
            self.all_LD_sqdists_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, cuda_lvl1_sumSimiNominators_LD, self.kern_alpha, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_LD, shared=smem_n_bytes)
        stream_neigh_LD.synchronize()
        end = time.time()
        time_niter    = ((end-start) / 2)
        print("time taken for all_LD_sqdists_cu: ", time_niter, 4)
        time_per_iter = time_niter / niter
        niter_per_second = 1 / time_per_iter
        print("niter per second for K =", __Kld__, " : ", niter_per_second)
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
        Xld_read = cuda_Xld_true_B
        knn_LD_read = cuda_knn_LD_B
        knn_LD_write = cuda_knn_LD_A
        sqdists_LD_write = cuda_sqdists_LD_A
        farthest_dist_LD_write = cuda_farthest_dist_LD_A
        simiNominators_LD_write = cuda_simiNominators_LD_A
        self.all_LD_sqdists_cu(np.uint32(self.N), np.uint32(self.Mld), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, simiNominators_LD_write, cuda_lvl1_sumSimiNominators_LD, self.kern_alpha, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_LD, shared=smem_n_bytes)
        verify_neighdists(Xld_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, self.N, self.Mld, __Kld__, stream_neigh_LD)

    def testing_neighdists_HD(self, cuda_Xhd, cuda_knn_HD_A, cuda_knn_HD_B, cuda_sqdists_HD_B, cuda_farthest_dist_HD_B, cuda_sqdists_HD_A, cuda_farthest_dist_HD_A, stream_neigh_HD):
        block_shape  = self.Kshapes2d_NxKhd_threads.block_x, self.Kshapes2d_NxKhd_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKhd_threads.grid_x_size, self.Kshapes2d_NxKhd_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKhd_threads.smem_n_bytes_per_block
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
        Xhd_read = cuda_Xhd
        knn_HD_read = cuda_knn_HD_A
        knn_HD_write = cuda_knn_HD_B
        sqdists_HD_write = cuda_sqdists_HD_B
        farthest_dist_HD_write = cuda_farthest_dist_HD_B
        self.all_HD_sqdists_euclidean_cu(np.uint32(self.N), np.uint32(self.Mhd), Xhd_read, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_HD, shared=smem_n_bytes)
        import time
        niter = 100
        start = time.time()
        for i in range(niter):
            seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
            Xhd_read = cuda_Xhd
            knn_HD_read = cuda_knn_HD_B
            knn_HD_write = cuda_knn_HD_A
            sqdists_HD_write = cuda_sqdists_HD_A
            farthest_dist_HD_write = cuda_farthest_dist_HD_A
            self.all_HD_sqdists_euclidean_cu(np.uint32(self.N), np.uint32(self.Mhd), Xhd_read, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_HD, shared=smem_n_bytes)
            seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
            Xhd_read = cuda_Xhd
            knn_HD_read = cuda_knn_HD_A
            knn_HD_write = cuda_knn_HD_B
            sqdists_HD_write = cuda_sqdists_HD_B
            farthest_dist_HD_write = cuda_farthest_dist_HD_B
            self.all_HD_sqdists_euclidean_cu(np.uint32(self.N), np.uint32(self.Mhd), Xhd_read, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_HD, shared=smem_n_bytes)
        stream_neigh_HD.synchronize()
        end = time.time()
        time_niter    = ((end-start) / 2)
        print("time taken for all_HD_sqdists_cu: ", time_niter, 4)
        time_per_iter = time_niter / niter
        niter_per_second = 1 / time_per_iter
        print("niter per second for K =", __Khd__, " : ", niter_per_second)
        seed = (np.uint32(np.random.randint(low = 1, high = __MAX_INT32_T__)) // 3) + 287378
        Xhd_read = cuda_Xhd
        knn_HD_read = cuda_knn_HD_B
        knn_HD_write = cuda_knn_HD_A
        sqdists_HD_write = cuda_sqdists_HD_A
        farthest_dist_HD_write = cuda_farthest_dist_HD_A
        self.all_HD_sqdists_euclidean_cu(np.uint32(self.N), np.uint32(self.Mhd), Xhd_read, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed, block=block_shape, grid=grid_shape, stream=stream_neigh_HD, shared=smem_n_bytes)
        verify_neighdists(Xhd_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, self.N, self.Mhd, __Khd__, stream_neigh_HD)

def verify_neighdists(cu_X, cu_neighbours, cu_neighdists, cu_farthests, N, M, K, stream):
    # sync stream
    stream.synchronize()
    cpu_X = np.zeros(shape=(N, M), dtype=np.float32)
    cpu_neighbours        = np.zeros(shape=(N, K), dtype=np.uint32)
    cpu_neighdists        = np.zeros(shape=(N, K), dtype=np.float32)
    cpu_farthestdists = np.zeros(shape=(N,),   dtype=np.float32)
    cpu_neighdists_recomputed = np.zeros(shape=(N, K), dtype=np.float32) # computed on CPU, should be equal to cpu_neighdists
    # Copy data from GPU to CPU
    cuda.memcpy_dtoh(cpu_X, cu_X.gpudata)
    cuda.memcpy_dtoh(cpu_neighbours, cu_neighbours.gpudata)
    cuda.memcpy_dtoh(cpu_neighdists, cu_neighdists.gpudata)
    cuda.memcpy_dtoh(cpu_farthestdists, cu_farthests.gpudata)
    # sync the device
    cuda.Context.synchronize()
    distances_match = True
    farthests_match = True
    sortedness_short = 0.0
    sortedness_far   = 0.0
    n_votes_short    = 0
    n_votes_far      = 0

    sum_dists  = np.zeros((K,), dtype=np.float32)
    last_dists = np.zeros((K,), dtype=np.float32)
    n_evals = 0.0

    N_commparisons = 3
    for comp in range(N_commparisons):
        i = np.random.randint(0, N)
        scaling = 1.0
        # GPU values
        dists_according_to_gpu    = cpu_neighdists[i] *scaling
        sum_dists  += dists_according_to_gpu
        n_evals += 1.0
        last_dists  = dists_according_to_gpu
        farthest_according_to_gpu = cpu_farthestdists[i]  *scaling
        # recompute on CPU
        X_i = cpu_X[i]
        for k in range(K):
            j    = cpu_neighbours[i, k]
            X_j  = cpu_X[j]
            diff = (X_i - X_j)
            cpu_neighdists_recomputed[i, k] = np.sum(diff*diff)  *scaling
        farthest_according_to_cpu = np.max(cpu_neighdists_recomputed[i])
        
        abs_dist_differences = np.abs(cpu_neighdists_recomputed[i] - dists_according_to_gpu)
        rel_diff = abs_dist_differences / np.mean(dists_according_to_gpu)
        distances_all_close = np.mean(rel_diff) < 1e-5
        farthest_ok         = (np.abs(farthest_according_to_cpu - farthest_according_to_gpu) / farthest_according_to_cpu < 1e-5)
        if not farthest_ok:
            if i < 3:
                print("farthest dists are wrong. This is normal after LD neighbours update. BUT HD NEIGHBBORUS UPDATES MUST ALSO UPDATE FARTHEST DISTS!!!")
            farthest_ok = True

        if (not distances_all_close)  or (not farthest_ok):
            # print the index where the distances anr not all close
            idx_different = np.where(abs_dist_differences > 1e-5)[0]
            print("idx_different : ", idx_different)

            print("i :", i)
            print("neigh[:17] ", cpu_neighbours[i][:17])
            print("diff: ", abs_dist_differences[:17])
            print("GPU:  ", np.round(dists_according_to_gpu, 2)[:17])
            print("cpu:  ", np.round(cpu_neighdists_recomputed[i], 2)[:17])
            print("----  farthest dists     GPU : ", farthest_according_to_gpu, "CPU : ", farthest_according_to_cpu)
            print("distances_all_close: ", distances_all_close, "farthest_ok: ", farthest_ok)

            largest_diff = np.max(abs_dist_differences)
            print("largest disance difference : ", largest_diff)
            raise Exception("error with neighs and dists")
        # check that each neighbour is unique
        neighbours = cpu_neighbours[i]
        for k1 in range(K):
            j1 = neighbours[k1]
            if j1 == i:
                print("i : ",i,   "   k1", k1,  "   j1", j1)
                print("neighbours[:32]:",neighbours[:32])
                raise Exception("neighbour is the point itself")
            for k2 in range(k1+1, K):
                j2 = neighbours[k2]
                if j1 == j2:
                    print("neighbours[:32]:",neighbours[:32])
                    print("same neighbours : ", j1, j2, " at indices : ", k1, k2)
                    raise Exception("neighbours are not unique")
                
        # short sortedness
        for k in range(K-1):
            if cpu_neighdists[i, k] > cpu_neighdists[i, k+1]:
                sortedness_short += 1.0
            else:
                sortedness_short += -1.0
            n_votes_short += 1
        # far sortedness
        for k in range(K):
            idx  = k 
            idx2 = np.random.randint(0, K)
            if idx == idx2:
                continue
            if idx2 < idx:
                tmpi = idx2
                idx2 = idx 
                idx  = tmpi
            if cpu_neighdists[i, idx] > cpu_neighdists[i, idx2]:
                sortedness_far += 1.0
            else:
                sortedness_far += -1.0
            n_votes_far += 1 
    sortedness_far    /= n_votes_far
    sortedness_short  /= n_votes_short
    print("OK    sortedness_far: ", sortedness_far, "sortedness_short: ", sortedness_short)
    # import matplotlib.pyplot as plt
    # plt.plot(sum_dists / n_evals)
    # plt.plot(last_dists)
    # plt.show()