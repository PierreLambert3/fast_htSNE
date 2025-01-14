import pycuda.driver   as cuda
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np






"""
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
----------------------- old structures that are still used in this code  -----------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
"""


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
        self.block_x = int(block_x)
        self.block_y = int(block_y)
        self.threads_per_block = int(threads_per_block)
        self.smem_n_bytes_per_block = int(smem_n_bytes_per_block)
        self.grid_x_size = int(n_blocks)
        self.grid_y_size = 1

    def smem(self):
        return self.smem_n_bytes_per_block

    def block(self):
        return (self.block_x, self.block_y, 1)

    def grid(self):
        return (self.grid_x_size, self.grid_y_size, 1)
    
    def print(self):
        print("block_x: ", self.block_x, " block_y: ", self.block_y)
        print("threads_per_block: ", self.threads_per_block)
        print("smem_n_bytes_per_block: ", self.smem_n_bytes_per_block)
        print("grid_x_size: ", self.grid_x_size, " grid_y_size: ", self.grid_y_size)

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
        self.threads_per_block      = int(threads_per_block)
        self.smem_n_bytes_per_block = int(smem_n_bytes_per_block)
        self.grid_x_size            = int(n_blocks)

    def smem(self):
        return self.smem_n_bytes_per_block

    def block(self):
        return (self.threads_per_block, 1, 1)

    def grid(self):
        return (self.grid_x_size, 1, 1)

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
    
    def free(self):
        try:
            self.lvl1_.free()
            self.lvl2_.free()
            self.lvl3_.free()
            self.lvl4_.free()
        except:
            pass
    
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
    
    def free(self):
        try:
            self.lvl1_.free()
            self.lvl2_.free()
            self.lvl3_.free()
            self.lvl4_.free()
        except:
            pass

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
            self.reduce_code = compiled_cuda_code.get_function("kernel_uint32_tSumReduction_one_step")
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
    
    def free(self):
        try:
            self.lvl1_.free()
            self.lvl2_.free()
            self.lvl3_.free()
            self.lvl4_.free()
        except:
            pass

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




"""
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
---------------------  new structures, migrate towards these from now on  ----------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------
"""

class _cuShape_grid2d:
    def __init__(self, cuda_device_attributes: 'deviceAttributes', n_thds: int, n_thds_per_grid_row: int, blockSize_multiple_of:int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> None:
        if not (n_thds % n_thds_per_grid_row == 0):
            raise Exception("cuShape_grid2d: n_thds is not divisible by n_thds_per_grid_row")
        if not (n_thds_per_grid_row % 32 == 0):
            raise Exception("cuShape_grid2d: n_thds_per_grid_row is not divisible by 32")
        
        # determine block size : start at max size and remove 32 until it fits or is 1
        now_thds_per_block   = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        while now_thds_per_block > n_thds_per_grid_row+32+1:
            now_thds_per_block -= 32

        used_shmem           = (now_thds_per_block * n_32b_perThd) + additional_n_32b
        block_max_shmem_n32b = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 4
        while used_shmem > block_max_shmem_n32b:
            if now_thds_per_block == 1:
                raise Exception("cuShape: not enough shared memory")
            now_thds_per_block -= 32
            if now_thds_per_block <= 0:
                now_thds_per_block = 1
            used_shmem = (now_thds_per_block * n_32b_perThd) + additional_n_32b
        self.block_x = int(now_thds_per_block)
        if not (self.block_x % blockSize_multiple_of == 0):
            raise Exception("cuShape_grid2d: block_x is not divisible by blockSize_multiple_of")
        self.shmem_n_bytes = int(used_shmem * 4)
        # how many blocks for each row to get to n_thds_per_grid_row
        self.grid_x  = int((n_thds_per_grid_row + now_thds_per_block - 1) // now_thds_per_block)
        self.grid_y  = int((n_thds + n_thds_per_grid_row - 1) // n_thds_per_grid_row)
        
        max_grid_dim = (cuda_device_attributes[cuda.device_attribute.MAX_GRID_DIM_X], cuda_device_attributes[cuda.device_attribute.MAX_GRID_DIM_Y])
        total_n_threads_per_row = self.block_x * self.grid_x
        n_buffers_per_row = (total_n_threads_per_row - n_thds_per_grid_row)
        self.sanity_check(max_grid_dim, n_thds, n_buffers_per_row)

    def smem(self):
        return self.shmem_n_bytes

    def block(self):
        return (self.block_x, 1, 1)

    def grid(self):
        return (self.grid_x, self.grid_y, 1)

    def sanity_check(self, max_grid_dim, requested_n_thds_total, n_buffer_thds_lastBlock_each_row):
        assert max_grid_dim[0] >= self.grid_x
        assert max_grid_dim[1] >= self.grid_y
        threads_per_block = self.block_x
        used_threads    = threads_per_block * self.grid_x * self.grid_y - n_buffer_thds_lastBlock_each_row*self.grid_y
        assert requested_n_thds_total == used_threads
        assert (threads_per_block == 1) or (threads_per_block % 32 == 0)
        assert self.block_x > 0 and self.grid_x > 0 and self.grid_y > 0

class _cuShape:
    def __init__(self, cuda_device_attributes: 'deviceAttributes', n_thds: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> None:
        self.block_x:       int = 0
        self.block_y:       int = 1
        self.n_blocks:      int = 0
        self.shmem_n_bytes: int = 0
        block_max_thds       = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        block_max_shmem_n32b = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 4
        # find the optimal block size, that respects the shared memory constraint and the number of threads constraint
        # the optimal block size is the one that maximizes the number of threads per block, but it should be a multiple of 32
        now_block_x         = block_max_thds
        now_thds_per_block  = now_block_x
        used_shmem          = (now_thds_per_block * n_32b_perThd) + additional_n_32b
        while used_shmem > block_max_shmem_n32b:
            if now_block_x == 1:
                raise Exception("cuShape: not enough shared memory")
            now_block_x -= 32
            if now_block_x <= 0:
                now_block_x = 1
            now_thds_per_block  = now_block_x
            used_shmem          = (now_thds_per_block * n_32b_perThd) + additional_n_32b
        self.block_x       = now_block_x
        self.block_y       = 1
        self.n_blocks      = (n_thds + now_thds_per_block - 1) // now_thds_per_block
        self.shmem_n_bytes = used_shmem * 4
        self.sanity_check(n_thds)

    def smem(self) -> int:
        return int(self.shmem_n_bytes)

    def block(self):
        return (int(self.block_x), int(self.block_y), int(1))

    def grid(self):
        return (int(self.n_blocks), int(1), int(1))
    
    def sanity_check(self, n_thds_total: int) -> None:
        threads_per_block = self.block_x * self.block_y
        assert n_thds_total <= threads_per_block * self.n_blocks
        assert (threads_per_block == 1) or (threads_per_block % 32 == 0)
        assert self.block_x > 0 and self.block_y > 0 and self.n_blocks > 0 and threads_per_block > 0

class _cuShape_2d(_cuShape):
    def __init__(self, cuda_device_attributes: 'deviceAttributes', n_thds: int, block_x_requested: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> None:
        self.block_x:       int = block_x_requested # force block_x to be the one given
        self.block_y:       int = 1
        self.n_blocks:      int = 0
        self.shmem_n_bytes: int = 0
        block_max_thds       = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        block_max_shmem_n32b = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] // 4
        buffer                  = 0
        if (self.block_x % 32) != 0:
            print("\033[93m    [WARNING] cuShape_2d: block_x is not divisible by 32: adding a buffer\033[0m")
            buffer = 32 - (self.block_x % 32)
            self.block_x += buffer
            assert (self.block_x % 32) == 0
        if self.block_x > block_max_thds:
            raise Exception("cuShape_2d: block_x exceeds the maximum threads per block")
        # Find the optimal block_y that respects the shared memory constraint
        now_block_y        = block_max_thds // self.block_x
        now_thds_per_block_all   = self.block_x * now_block_y
        used_shmem         = (now_thds_per_block_all * n_32b_perThd) + additional_n_32b
        while (used_shmem > block_max_shmem_n32b) or (now_thds_per_block_all > block_max_thds):
            if now_block_y == 1:
                raise Exception("cuShape_2d: not enough shared memory")
            now_block_y -= 1
            now_thds_per_block_all   = self.block_x * now_block_y
            used_shmem         = (now_thds_per_block_all * n_32b_perThd) + additional_n_32b
        self.block_y = now_block_y
        now_usefulthds_per_block = (self.block_x-buffer) * now_block_y
        self.n_blocks = (n_thds + now_usefulthds_per_block - 1) // now_usefulthds_per_block
        self.shmem_n_bytes = used_shmem * 4
        self.sanity_check(n_thds, block_x_requested, buffer, block_max_thds)

    def sanity_check(self, n_thds_requested, requested_block_x, buffer, block_max_thds) -> None:
        threads_per_block_all  = self.block_x * self.block_y
        threads_per_block_used = (self.block_x-buffer) * self.block_y
        assert threads_per_block_all <= block_max_thds
        assert self.block_x >= requested_block_x
        assert (self.block_x % 32) == 0
        assert n_thds_requested <= threads_per_block_used * self.n_blocks
        assert self.block_x > 0 and self.block_y > 0 and self.n_blocks > 0 and threads_per_block_used > 0

class CUDA_kernel:
    def __init__(self, kernel_name: str, compiled_code: SourceModule, cuShape: _cuShape) -> None:
        self.kernel  = compiled_code.get_function(kernel_name)
        self.cuShape = cuShape
        self._blocker_stream = cuda.Stream() # not exposed to the user

    # example launch: my_kernel.async_launch(stream, array1, number, array2)    
    def async_launch(self, stream, *args) -> None:
        self.kernel(*args, block=self.cuShape.block(), grid=self.cuShape.grid(), shared=self.cuShape.smem(), stream=stream)
    
    def blocking_launch(self, *args) -> None:
        self.kernel(*args, block=self.cuShape.block(), grid=self.cuShape.grid(), shared=self.cuShape.smem(), stream=self._blocker_stream)
        self._blocker_stream.synchronize()

class GPU_context:
    def __init__(self, ctx_global: 'GPU_hidden_wrapper', context_label: str) -> None:
        self.ctx_global = ctx_global
        self.label      = context_label

    def malloc(self, np_array: np.ndarray) -> gpuarray.GPUArray:
        return self.ctx_global.malloc(np_array, self.label)
    
    def copy_gpu2gpu_async(self, dest_gpuarray, src_gpuarray, stream) -> None:
        cuda.memcpy_dtod_async(dest_gpuarray.gpudata, src_gpuarray.gpudata, src_gpuarray.nbytes, stream=stream)

    def free_context(self) -> None:
        self.ctx_global.free_context(self.label)
    
    def free(self, gpu_array: gpuarray.GPUArray) -> None:
        self.ctx_global.free(gpu_array)
    
    def stream(self) -> cuda.Stream:
        return self.ctx_global.stream()

    def compile(self, cuda_code: str) -> SourceModule:
        return self.ctx_global.compile(cuda_code)
    
    def make_cuShape(self, n_thds: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> _cuShape:
        return self.ctx_global.make_cuShape(n_thds, n_32b_perThd, additional_n_32b)
    
    def make_cuShape_2d(self, n_thds: int, block_x: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> _cuShape_2d:
        return self.ctx_global.make_cuShape_2d(n_thds, block_x, n_32b_perThd, additional_n_32b)
    
    def make_cuShape_grid2d(self, n_thds: int, n_thds_per_grid_row: int, n_32b_perThd: int, additional_n_32b: int, blockSize_multiple_of:int):
        return self.ctx_global.make_cuShape_grid2d(n_thds=n_thds, n_thds_per_grid_row=n_thds_per_grid_row, n_32b_perThd=n_32b_perThd, additional_n_32b=additional_n_32b, blockSize_multiple_of=blockSize_multiple_of)
    
    def close(self) -> None:
        self.free_context(self.label)
    
    def sync_full_context(self) -> None:
        self.ctx_global.ctx.synchronize()

    def get_device_attributes(self) -> 'deviceAttributes':
        return self.ctx_global.get_device_attributes()

class GPU_hidden_wrapper:
    def __init__(self, device_id: int= 0) -> None:
        # weird init,but apparently necessary when using custom cuda kernels and pytorch on the same GPU
        cuda.init()
        self.device_id = device_id
        ctx = cuda.Device(device_id).make_context()
        ctx.attach()
        self.ctx      = ctx
        self.gpuarray = gpuarray
        self.compile_opts = ["-O3", "--use_fast_math", "-prec-div=false", "-ftz=true", "-prec-sqrt=false", "-fmad=true"]
        self.compiled_codes   = []
        self.allocated_arrays = {}
    
    def make_context(self, context_label: str) -> GPU_context:
        return GPU_context(self, context_label)
    
    def stream(self) -> cuda.Stream:
        return cuda.Stream()

    def compile(self, cuda_code: str) -> SourceModule:
        compiled_code = SourceModule(cuda_code, options=self.compile_opts)
        self.compiled_codes.append(compiled_code)
        return compiled_code
    
    def malloc(self, np_array: np.ndarray, context_label: str) -> gpuarray.GPUArray:
        gpu_array = gpuarray.to_gpu(np_array)
        if context_label not in self.allocated_arrays:
            self.allocated_arrays[context_label] = []
        self.allocated_arrays[context_label].append(gpu_array)
        return gpu_array

    def free_context(self, context_label: str) -> None:
        if context_label in self.allocated_arrays:
            for gpu_array in self.allocated_arrays[context_label]:
                self.free(gpu_array)
            del self.allocated_arrays[context_label]

    def free(self, gpu_array: gpuarray.GPUArray) -> None:
        if gpu_array.gpudata is not None:
            gpu_array.gpudata.free()
            gpu_array.gpudata = None

    def get_device_attributes(self) -> 'deviceAttributes':
        return cuda.Device(self.device_id).get_attributes()

    def make_cuShape(self, n_thds: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> _cuShape:
        device_attributes = cuda.Device(self.device_id).get_attributes()
        return _cuShape(device_attributes, n_thds, n_32b_perThd, additional_n_32b)

    def make_cuShape_2d(self, n_thds: int, block_x: int, n_32b_perThd: int = 0, additional_n_32b: int = 0) -> _cuShape_2d:
        device_attributes = cuda.Device(self.device_id).get_attributes()
        return _cuShape_2d(device_attributes, n_thds, block_x, n_32b_perThd, additional_n_32b)
    
    def make_cuShape_grid2d(self, n_thds: int, n_thds_per_grid_row: int, n_32b_perThd: int, additional_n_32b: int, blockSize_multiple_of:int):
        device_attributes = cuda.Device(self.device_id).get_attributes()
        return _cuShape_grid2d(device_attributes, n_thds=n_thds, n_thds_per_grid_row=n_thds_per_grid_row, n_32b_perThd=n_32b_perThd, additional_n_32b=additional_n_32b, blockSize_multiple_of=blockSize_multiple_of)

    def close(self) -> None:
        if self.ctx is None:
            return
        for context_label in list(self.allocated_arrays.keys()):
            self.free_context(context_label)
        self.ctx.pop()
        self.ctx.detach()
        self.ctx = None

    def __del__(self) -> None:
        self.close()







