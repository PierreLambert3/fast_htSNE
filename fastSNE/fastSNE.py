import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import & init pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from fastSNE.cuda_kernels import kernel_minMax_reduction, kernel_X_to_transpose, kernel_scale_X
from fastSNE.cuda_kernels import kernel_compute_all_HD_sqdists_euclidean, kernel_compute_all_HD_sqdists_manhattan, kernel_compute_all_HD_sqdists_cosine, kernel_compute_all_HD_sqdists_custom
from fastSNE.cuda_kernels import kernel_compute_all_LD_sqdists

__MAX_PERPLEXITY__ = 80.0
__MIN_PERPLEXITY__ = 1.5
__MAX_KERNEL_ALPHA__ = 100.0
__MIN_KERNEL_ALPHA__ = 0.05
__MAX_ATTRACTION_MULTIPLIER__ = 10.0
__MIN_ATTRACTION_MULTIPLIER__ = 0.1

__Khd__       = ((int(__MAX_PERPLEXITY__ * 3) // 32) + 1) * 32 # n neighbours in HD. Needs to be divisible by 32
__Kld__       = 32  # n neighbours in LD. Needs to be divisible by 32
__N_CAND_LD__ = 32  # number of candidate points during iterative neighbourhood estimation Needs to be divisible by 32
__N_CAND_HD__ = 32  # Needs to be divisible by 32



'''
ALL CONSTANTS IN CUDA INSTEAD OF PYTHON (and then read from python code)

cuda: tout mettre dans un seul code pour eviter les doublons: 1 seule compilation
ensuite pull les fonctions les unes pares les autres
'''


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
    # fixed size block_y, find the optimal block_x
    def __init__(self, N_threads_total, N_threads_block_y, smem_n_float32_per_thread, cuda_device_attributes, constant_additional_smem_n_float32, smem_n_float32_per_block_x):
        max_threads_per_block = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        max_shared_memory_per_block = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
        max_block_x = cuda_device_attributes[cuda.device_attribute.MAX_BLOCK_DIM_X]
        max_block_y = cuda_device_attributes[cuda.device_attribute.MAX_BLOCK_DIM_Y]
        if max_block_y < N_threads_block_y:
            raise Exception("Kernel_shapes_2dBlocks: N_threads_block_y is too large")
        if max_shared_memory_per_block < (constant_additional_smem_n_float32 + smem_n_float32_per_thread + smem_n_float32_per_block_x) * np.dtype(np.float32).itemsize:
            raise Exception("Shared memory requirements too large for the GPU. Solution: reduce the dimensionality of your input (for instance, use the 50 first principal components)")
        # find the number of threads per block: grox block_x until one of the constraints is violated
        block_x = 1
        block_y = N_threads_block_y
        threads_per_block = block_x * block_y
        smem_n_bytes_per_block = (threads_per_block * smem_n_float32_per_thread + block_x*smem_n_float32_per_block_x) * np.dtype(np.float32).itemsize
        n_blocks  = (N_threads_total + threads_per_block - 1) // threads_per_block
        while True:
            if threads_per_block >= N_threads_total:
                break
            next_block_x = block_x + 1
            next_threads_per_block = next_block_x * block_y
            next_smem_n_bytes_per_block = (next_threads_per_block * smem_n_float32_per_thread + constant_additional_smem_n_float32 + block_x*smem_n_float32_per_block_x) * np.dtype(np.float32).itemsize
            next_n_blocks = (N_threads_total + next_threads_per_block - 1) // next_threads_per_block
            next_tpb_ok = next_threads_per_block <= max_threads_per_block
            next_smem_ok = next_smem_n_bytes_per_block <= max_shared_memory_per_block
            next_block_x_ok = next_block_x <= max_block_x
            if next_tpb_ok and next_smem_ok and next_block_x_ok:
                block_x = next_block_x
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

class fastSNE:
    def __init__(self, with_GUI, n_components=2, random_state=None):
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
        # on CPU
        self.N        = N
        self.Mhd      = M
        self.cpu_Xld  = ((np.random.uniform(size=(N, self.Mld)).astype(np.float32) - 0.5) * 2.0) 
        # determine grid shapes, block shapes, smem size for each CUDA kernels & compile kernels
        self.configue_and_initialise_CUDA_kernels_please(__Khd__, __Kld__, self.Mhd, self.Mld)
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
            left_bound1  = i + 1
            right_bound1 = left_bound1 + __Kld__
            if right_bound1 >= N:
                left_bound1  = 0
                right_bound1 = __Kld__
            init_LD_neighs[i] = np.arange(left_bound1, right_bound1)
            right_bound2  = i-1 
            left_bound2   = right_bound2 - __Khd__
            if left_bound2 < 0:
                left_bound2 = 0
                right_bound2 = __Khd__
            init_HD_neighs[i] = np.arange(left_bound2, right_bound2)
        # mallocs on the device
        cuda_Xhd                = gpuarray.to_gpu_async(Xhd)
        cuda_knn_HD_A           = gpuarray.to_gpu(init_HD_neighs)
        cuda_knn_HD_B           = gpuarray.to_gpu(init_HD_neighs)
        cuda_sqdists_HD_A       = gpuarray.to_gpu(np.zeros((N, __Khd__), dtype=np.float32))  # TODO: init this
        cuda_sqdists_HD_B       = gpuarray.to_gpu(np.zeros((N, __Khd__), dtype=np.float32))  # TODO: init this
        cuda_farthest_dist_HD_A = gpuarray.to_gpu(np.ones(N, dtype=np.float32))              # TODO: init this
        cuda_farthest_dist_HD_B = gpuarray.to_gpu(np.ones(N, dtype=np.float32))              # TODO: init this
        cuda_candidate_dists_HD = gpuarray.to_gpu(np.zeros((N, __N_CAND_HD__), dtype=np.float32)) # TODO: init this

        cuda_Xld_true_A           = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_true_B           = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_nest             = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_mmtm             = gpuarray.to_gpu(np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype))
        cuda_knn_LD_A             = gpuarray.to_gpu(init_LD_neighs)
        cuda_knn_LD_B             = gpuarray.to_gpu(init_LD_neighs)
        cuda_sqdists_LD_A         = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_sqdists_LD_B         = gpuarray.to_gpu(np.zeros((N, __Kld__), dtype=np.float32)) # TODO: init this
        cuda_farthest_dist_LD_A   = gpuarray.to_gpu(np.ones(N, dtype=np.float32))             # TODO: init this
        cuda_farthest_dist_LD_B   = gpuarray.to_gpu(np.ones(N, dtype=np.float32))             # TODO: init this
        cuda_candidate_dists_LD   = gpuarray.to_gpu(np.zeros((N, __N_CAND_LD__), dtype=np.float32)) # TODO: init this
        self.fill_all_sqdists_HD(cuda_Xhd, cuda_knn_HD_A, cuda_knn_HD_B, cuda_sqdists_HD_B, cuda_farthest_dist_HD_B, stream_neigh_HD)
        self.fill_all_sqdists_HD(cuda_Xhd, cuda_knn_HD_B, cuda_knn_HD_A, cuda_sqdists_HD_A, cuda_farthest_dist_HD_A, stream_neigh_HD)
        self.fill_all_sqdists_LD(cuda_Xld_true_A, cuda_knn_LD_A, cuda_knn_LD_B, cuda_sqdists_LD_B, cuda_farthest_dist_LD_B, stream_neigh_LD)
        self.fill_all_sqdists_LD(cuda_Xld_true_B, cuda_knn_LD_B, cuda_knn_LD_A, cuda_sqdists_LD_A, cuda_farthest_dist_LD_A, stream_neigh_LD)
        stream_neigh_HD.synchronize()
        stream_neigh_LD.synchronize()

        # launch the tSNE optimisation
        if self.with_GUI:
            self.fit_with_gui(__Khd__, __Kld__, Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm, all_streams)
        else:
            self.fit_without_gui(__Khd__, __Kld__, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm, all_streams)
        self.is_fitted = True

    def transform(self):
        if not self.is_fitted:
            raise Exception("fastSNE: transform() called before fit(), or fit failed crashingly")
        # return self.cpu_Xld
        return None

    def one_iteration(self, cuda_Xld_true):
        return

    def fit_with_gui(self, Khd, Kld, Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm, all_streams):
        stream_minMax, stream_neigh_HD , stream_neigh_LD, stream_grads = all_streams
        # 1. configure the process launch mode 
        multiprocessing.set_start_method('spawn') # this is crucial for the GUI to work correctly. Python is wierd and often annoying

        # 2. shared memory with GUI (on CPU)
        cpu_shared_mem      = shared_memory.SharedMemory(create=True, size=int(self.N * self.Mld * np.dtype(np.float32).itemsize))
        cpu_Xld_arr_on_smem = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem.buf)
        # copy (GPU->CPU) cuda_Xld_true_A and cuda_Xld_true_B to shared memory
        cuda_Xld_true_A.get(cpu_Xld_arr_on_smem)
        # temp structures related to preprocessing the data for the GUI
        cuda_Xld_temp_Xld         = gpuarray.to_gpu(np.zeros((self.N, self.Mld), dtype=np.float32))
        cuda_Xld_T_temp_lvl1_mins = gpuarray.to_gpu(np.zeros((self.N, self.Mld), dtype=np.float32))
        cuda_Xld_T_temp_lvl1_maxs = gpuarray.to_gpu(np.zeros((self.N, self.Mld), dtype=np.float32))

        # 3.   Launching the process responsible for the GUI
        from fastSNE.fastSNE_gui import gui_worker
        #  Shared hyperparameters
        kernel_alpha   = multiprocessing.Value('f', self.kern_alpha)
        perplexity     = multiprocessing.Value('f', self.perplexity)
        attrac_mult    = multiprocessing.Value('f', self.attrac_mult)
        dist_metric    = multiprocessing.Value('i', self.dist_metric)
        #  Shared state variables
        gui_closed                 = multiprocessing.Value('b', False)
        points_ready_for_rendering = multiprocessing.Value('b', False)
        points_rendering_finished  = multiprocessing.Value('b', True)
        iteration                  = multiprocessing.Value('i', 0)
        # 3.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem, Y, self.N, self.Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration))
        process_gui.start()

        # 4.   Optimise until the GUI is closed
        iteration_int         = 0
        isPhaseA              = True
        gui_data_prep_phase   = 0
        busy_copying__for_GUI = False
        gui_was_closed        = False
        while not gui_was_closed:
            # sync all streams (else read/writes will conflict with versions A and B)
            stream_minMax.synchronize()
            stream_neigh_HD.synchronize()
            stream_neigh_LD.synchronize()
            stream_grads.synchronize()

            # update hyperparameters
            new_perplexity    = perplexity.value
            new_dist_metric   = dist_metric.value
            HD_config_changed = (new_perplexity != self.perplexity or new_dist_metric != self.dist_metric)
            self.kern_alpha   = kernel_alpha.value
            self.attrac_mult  = attrac_mult.value
            self.perplexity   = new_perplexity
            self.dist_metric  = new_dist_metric

            if HD_config_changed:
                1/0 # todo: recompute HD distances and P for all points

            # One iteration of the tSNE optimisation
            if isPhaseA:
                self.one_iteration(cuda_Xld_true_A)
            else:
                self.one_iteration(cuda_Xld_true_B)

            # GUI communication & preparation of the data for the GUI
            if gui_data_prep_phase == 0: # copy cuda_Xld_true_A/B to cuda_Xld_temp in an async manner using stream_minMax*
                # if we were copying the data for the GUI, notify the GUI that the data is ready
                if busy_copying__for_GUI:
                    busy_copying__for_GUI = False
                    with points_rendering_finished.get_lock():
                        points_rendering_finished.value = False
                    with points_ready_for_rendering.get_lock():
                        points_ready_for_rendering.value = True
                # copy the fresh data to cuda_Xld_temp as a transposed matrix
                Xld_to_transpose = cuda_Xld_true_B if isPhaseA else cuda_Xld_true_A
                cuda_Xld_temp_Xld.set_async(Xld_to_transpose, stream=stream_minMax)
                self.compiled_X_to_transpose.get_function("kernel_X_to_transpose")(Xld_to_transpose, cuda_Xld_T_temp_lvl1_maxs, np.uint32(self.N), np.uint32(self.Mld), block=(self.Kshapes_transpose.threads_per_block, 1, 1), grid=(self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size), stream=stream_minMax)
                cuda_Xld_T_temp_lvl1_mins.set_async(cuda_Xld_T_temp_lvl1_maxs, stream=stream_minMax)
            elif gui_data_prep_phase == 1: # perform the min-max reduction on cuda_Xld_temp, & scale the data to [0, 1] with the results
                self.scaling_of_points(cuda_Xld_temp_Xld, cuda_Xld_T_temp_lvl1_mins, cuda_Xld_T_temp_lvl1_maxs, stream_minMax)
            else: # copy cuda_Xld_temp to cpu_Xld_arr_on_smem, if the GUI is ready
                gui_done = False
                busy_copying__for_GUI = False
                with points_rendering_finished.get_lock():
                    gui_done = points_rendering_finished.value
                if gui_done:
                    cuda_Xld_temp_Xld.get_async(stream=stream_minMax, ary=cpu_Xld_arr_on_smem)
                    busy_copying__for_GUI = True
                    iteration.value = iteration_int
            gui_data_prep_phase = (gui_data_prep_phase + 1) % 3

            isPhaseA = not isPhaseA
            iteration_int += 1
            with gui_closed.get_lock():
                gui_was_closed = gui_closed.value
        process_gui.join()

        cpu_shared_mem.unlink()
        self.free_all_GPU_memory(cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        
        return
    
    
    def fit_without_gui(self, Khd, Kld, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm, all_streams):
        1/0

    def scaling_of_points(self, cuda_Xld_temp_Xld, cuda_Xld_temp_lvl1_mins, cuda_Xld_temp_lvl1_maxs, stream_minMax):
        cuda_kernel = self.compiled_minMax_reduction.get_function("perform_minMax_reduction")
        stream      = stream_minMax
        # level 1 reduction (special case because we don't want to write on the cuda_Xld_temp_Xld)
        block_size  = self.Kshapes_minMax_lvl_1.threads_per_block
        grid_shape  = (self.Kshapes_minMax_lvl_1.grid_x_size, self.Kshapes_minMax_lvl_1.grid_y_size)
        smem_n_bytes = self.Kshapes_minMax_lvl_1.smem_n_bytes_per_block
        N_after_reduction = self.perdim_remaining_after_reduction1
        cuda_kernel(cuda_Xld_temp_lvl1_mins, cuda_Xld_temp_lvl1_maxs,\
                    self.reduction1_result_mins, self.reduction1_result_maxs, np.uint32(self.N), np.uint32(self.Mld), np.uint32(N_after_reduction),\
                    block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # level 2 reduction
        if not self.do_reduction2:
            mins = self.reduction1_result_mins
            maxs = self.reduction1_result_maxs
            block_size = self.Kshapes_transpose.threads_per_block
            grid_shape = (self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size)
            smem_n_bytes = block_size * 4
            self.compiled_scaling_X.get_function("kernel_scale_X")(cuda_Xld_temp_Xld, mins, maxs, np.uint32(self.N), np.uint32(self.Mld), block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
            return
        block_size  = self.Kshapes_minMax_lvl_2.threads_per_block
        grid_shape  = (self.Kshapes_minMax_lvl_2.grid_x_size, self.Kshapes_minMax_lvl_2.grid_y_size)
        smem_n_bytes = self.Kshapes_minMax_lvl_2.smem_n_bytes_per_block
        N_after_reduction = self.perdim_remaining_after_reduction2
        cuda_kernel(self.reduction1_result_mins, self.reduction1_result_maxs,\
                    self.reduction2_result_mins, self.reduction2_result_maxs, np.uint32(self.perdim_remaining_after_reduction1), np.uint32(self.Mld), np.uint32(N_after_reduction),\
                    block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # level 3 reduction
        if not self.do_reduction3:
            mins = self.reduction2_result_mins
            maxs = self.reduction2_result_maxs
            block_size = self.Kshapes_transpose.threads_per_block
            grid_shape = (self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size)
            smem_n_bytes = block_size * 4
            self.compiled_scaling_X.get_function("kernel_scale_X")(cuda_Xld_temp_Xld, mins, maxs, np.uint32(self.N), np.uint32(self.Mld), block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
            return
        block_size  = self.Kshapes_minMax_lvl_3.threads_per_block
        grid_shape  = (self.Kshapes_minMax_lvl_3.grid_x_size, self.Kshapes_minMax_lvl_3.grid_y_size)
        smem_n_bytes = self.Kshapes_minMax_lvl_3.smem_n_bytes_per_block
        N_after_reduction = self.perdim_remaining_after_reduction3
        cuda_kernel(self.reduction2_result_mins, self.reduction2_result_maxs,\
                    self.reduction3_result_mins, self.reduction3_result_maxs, np.uint32(self.perdim_remaining_after_reduction2), np.uint32(self.Mld), np.uint32(N_after_reduction),\
                    block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # level 4 reduction
        if not self.do_reduction4:
            mins = self.reduction3_result_mins
            maxs = self.reduction3_result_maxs
            block_size = self.Kshapes_transpose.threads_per_block
            grid_shape = (self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size)
            smem_n_bytes = block_size * 4
            self.compiled_scaling_X.get_function("kernel_scale_X")(cuda_Xld_temp_Xld, mins, maxs, np.uint32(self.N), np.uint32(self.Mld), block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
            return
        block_size  = self.Kshapes_minMax_lvl_4.threads_per_block
        grid_shape  = (self.Kshapes_minMax_lvl_4.grid_x_size, self.Kshapes_minMax_lvl_4.grid_y_size)
        smem_n_bytes = self.Kshapes_minMax_lvl_4.smem_n_bytes_per_block
        N_after_reduction = self.perdim_remaining_after_reduction4
        cuda_kernel(self.reduction3_result_mins, self.reduction3_result_maxs,\
                    self.reduction4_result_mins, self.reduction4_result_maxs, np.uint32(self.perdim_remaining_after_reduction3), np.uint32(self.Mld), np.uint32(N_after_reduction),\
                    block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)
        # final scaling
        mins = self.reduction4_result_mins
        maxs = self.reduction4_result_maxs
        block_size = self.Kshapes_transpose.threads_per_block
        grid_shape = (self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size)
        smem_n_bytes = block_size * 4
        self.compiled_scaling_X.get_function("kernel_scale_X")(cuda_Xld_temp_Xld, mins, maxs, np.uint32(self.N), np.uint32(self.Mld), block=(block_size, 1, 1), grid=grid_shape, stream=stream, shared=smem_n_bytes)


    def fill_all_sqdists_HD(self, Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, stream):
        dist_type = self.dist_metric
        kernel = None 
        if dist_type == 0:
            kernel = self.kernel_all_HD_sqdists_euclidean
        elif dist_type == 1:
            kernel = self.kernel_all_HD_sqdists_manhattan
        elif dist_type == 2:
            kernel = self.kernel_all_HD_sqdists_cosine
        else:
            kernel = self.kernel_all_HD_sqdists_custom
        block_shape  = self.Kshapes2d_NxKhd_threads.block_x, self.Kshapes2d_NxKhd_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKhd_threads.grid_x_size, self.Kshapes2d_NxKhd_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKhd_threads.smem_n_bytes_per_block
        kernel(np.uint32(self.N), np.uint32(self.Mhd), np.uint32(__Khd__), Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)

    def fill_all_sqdists_LD(self, Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, stream):
        kernel = self.kernel_all_LD_sqdists
        block_shape  = self.Kshapes2d_NxKld_threads.block_x, self.Kshapes2d_NxKld_threads.block_y, 1
        grid_shape   = self.Kshapes2d_NxKld_threads.grid_x_size, self.Kshapes2d_NxKld_threads.grid_y_size, 1
        smem_n_bytes = self.Kshapes2d_NxKld_threads.smem_n_bytes_per_block
        kernel(np.uint32(self.N), np.uint32(self.Mld), np.uint32(__Kld__), Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, block=block_shape, grid=grid_shape, stream=stream, shared=smem_n_bytes)

        print("carefull when checking: I write to an array and read from another")


    def configue_and_initialise_CUDA_kernels_please(self, Khd, Kld, Mhd, Mld):
        N = self.N
        cuda_device = cuda.Device(0)
        cuda_device_attributes = cuda_device.get_attributes()

        # ------------ 0. kernels used for getting the transpose of Xld  -------
        n_threads = N
        multiple_of = 32 if n_threads > 32 else 1
        smem_n_float32_per_thread = 1
        self.Kshapes_transpose = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, 1)
        self.Kshapes_transpose.grid_y_size = Mld

        # ------------ 1. kernels used for rendering (ie: scaling points between 0.0f and 1.0f)  -------
        #    finding the min and max values for each Mld dimension
        #    2d grid: dim1~N, dim2~Mld^
        #    find for only 1 dimension (as if Mld=1), and then add grid_y dimensions for the others
        #     1.a  reduction level 2 
        self.Kshapes_minMax_lvl_1 = None; self.Kshapes_minMax_lvl_2 = None; self.Kshapes_minMax_lvl_3 = None; self.Kshapes_minMax_lvl_4 = None;
        self.do_reduction2 = False; self.do_reduction3 = False; self.do_reduction4 = False;
        self.perdim_remaining_after_reduction1 = 1; self.perdim_remaining_after_reduction2 = 1; self.perdim_remaining_after_reduction3 = 1; self.perdim_remaining_after_reduction4 = 1;
        n_threads   = N
        multiple_of = 32 if n_threads > 32 else 1
        smem_n_float32_per_thread = 2
        self.Kshapes_minMax_lvl_1 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, 1)
        self.Kshapes_minMax_lvl_1.grid_y_size = Mld
        # the array containing the results of the reduction
        self.perdim_remaining_after_reduction1 = self.Kshapes_minMax_lvl_1.grid_x_size
        self.reduction1_result_mins = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction1, Mld), dtype=np.float32))
        self.reduction1_result_maxs = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction1, Mld), dtype=np.float32))
        if self.perdim_remaining_after_reduction1 <= 1:
            self.do_reduction2 = False
            self.do_reduction3 = False
            self.do_reduction4 = False
        else:
            #     1.b  reduction level 2 
            self.do_reduction2 = True
            n_threads = self.perdim_remaining_after_reduction1
            multiple_of = 32 if n_threads > 32 else 1
            self.Kshapes_minMax_lvl_2 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, 1)
            self.Kshapes_minMax_lvl_2.grid_y_size = Mld
            # the array containing the results of the reduction
            self.perdim_remaining_after_reduction2 = self.Kshapes_minMax_lvl_2.grid_x_size
            self.reduction2_result_mins = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction2, Mld), dtype=np.float32))
            self.reduction2_result_maxs = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction2, Mld), dtype=np.float32))
            if self.perdim_remaining_after_reduction2 <= 1:
                self.do_reduction3 = False
                self.do_reduction4 = False
            else:
                #     1.c  reduction level 3 
                self.do_reduction3 = True
                n_threads = self.perdim_remaining_after_reduction2
                multiple_of = 32 if n_threads > 32 else 1
                self.Kshapes_minMax_lvl_3 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, 1)
                self.Kshapes_minMax_lvl_3.grid_y_size = Mld
                # the array containing the results of the reduction
                self.perdim_remaining_after_reduction3 = self.Kshapes_minMax_lvl_3.grid_x_size
                self.reduction3_result_mins = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction3, Mld), dtype=np.float32))
                self.reduction3_result_maxs = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction3, Mld), dtype=np.float32))
                if self.perdim_remaining_after_reduction3 <= 1:
                    self.do_reduction4 = False
                else:
                    #     1.d  reduction level 4 
                    self.do_reduction4 = True
                    n_threads = self.perdim_remaining_after_reduction3
                    multiple_of = 32 if n_threads > 32 else 1
                    self.Kshapes_minMax_lvl_4 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, 1)
                    self.Kshapes_minMax_lvl_4.grid_y_size = Mld
                    # the array containing the results of the reduction
                    self.perdim_remaining_after_reduction4 = self.Kshapes_minMax_lvl_4.grid_x_size
                    self.reduction4_result_mins = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction4, Mld), dtype=np.float32))
                    self.reduction4_result_maxs = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction4, Mld), dtype=np.float32))
                    if self.perdim_remaining_after_reduction4 > 1:
                        raise Exception("Splendid, you have more that 1e12 points in your dataset. This hardcoded limit was written in the past where such large datasets were not common. Contact me by e-mail or by cyberpigeon, whichever is the norm in your time.")

        # ------------ 2. kernels used for neighboru related things -------
        #   N threads
        n_threads = N
        multiple_of = 32 if n_threads > 32 else 1
        smem_n_float32_per_thread = 1
        self.Kshapes_N_threads = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes, Mhd)
        self.Kshapes_N_threads.grid_y_size = 1
        #   N x Kld threads , 1d grid, 2d block
        n_threads   = N * Kld
        block_y     = Kld
        smem_n_float32_per_thread = 3 
        smem_const_additional = Mld
        self.Kshapes2d_NxKld_threads = Kernel_shapes_2dBlocks(n_threads, block_y, smem_n_float32_per_thread, cuda_device_attributes, 1, smem_const_additional)
        #   N x Khd threads , 1d grid, 2d block
        n_threads   = N * Khd
        block_y     = Khd
        smem_n_float32_per_thread = 3 
        smem_const_additional = Mhd
        self.Kshapes2d_NxKhd_threads = Kernel_shapes_2dBlocks(n_threads, block_y, smem_n_float32_per_thread, cuda_device_attributes, 1, smem_const_additional)
        #  N x __N_CAND_LD__ threads, 1d grid, 2d block
        n_threads   = N * __N_CAND_LD__
        block_y     = __N_CAND_LD__
        smem_n_float32_per_thread = 3 
        smem_const_additional = Mld
        self.Kshapes2d_NxNcandLD_threads = Kernel_shapes_2dBlocks(n_threads, block_y, smem_n_float32_per_thread, cuda_device_attributes, 1, smem_const_additional)
        #  N x __N_CAND_HD__ threads, 1d grid, 2d block
        n_threads   = N * __N_CAND_HD__
        block_y     = __N_CAND_HD__
        smem_n_float32_per_thread = 3 
        smem_const_additional = Mhd
        self.Kshapes2d_NxNcandHD_threads = Kernel_shapes_2dBlocks(n_threads, block_y, smem_n_float32_per_thread, cuda_device_attributes, 1, smem_const_additional)
        
        # print the Kernel shapes for N x Kld:
        print("Kernel_shapes_N_threads: threads_per_block = {}, block_x = {}, block_y = {}, grid_x_size = {}, grid_y_size = {}".format(self.Kshapes2d_NxKld_threads.block_x*self.Kshapes2d_NxKld_threads.block_y, self.Kshapes2d_NxKld_threads.block_x, self.Kshapes2d_NxKld_threads.block_y, self.Kshapes2d_NxKld_threads.grid_x_size, self.Kshapes2d_NxKld_threads.grid_y_size))
        print("Kernel_shapes_N_threads: threads_per_block = {}, block_x = {}, block_y = {}, grid_x_size = {}, grid_y_size = {}".format(self.Kshapes2d_NxKhd_threads.block_x*self.Kshapes2d_NxKhd_threads.block_y, self.Kshapes2d_NxKhd_threads.block_x, self.Kshapes2d_NxKhd_threads.block_y, self.Kshapes2d_NxKhd_threads.grid_x_size, self.Kshapes2d_NxKhd_threads.grid_y_size))
        print("Kernel_shapes_N_threads: threads_per_block = {}, block_x = {}, block_y = {}, grid_x_size = {}, grid_y_size = {}".format(self.Kshapes2d_NxNcandHD_threads.block_x*self.Kshapes2d_NxNcandHD_threads.block_y, self.Kshapes2d_NxNcandHD_threads.block_x, self.Kshapes2d_NxNcandHD_threads.block_y, self.Kshapes2d_NxNcandHD_threads.grid_x_size, self.Kshapes2d_NxNcandHD_threads.grid_y_size))
        print("Kernel_shapes_N_threads: threads_per_block = {}, block_x = {}, block_y = {}, grid_x_size = {}, grid_y_size = {}".format(self.Kshapes2d_NxNcandLD_threads.block_x*self.Kshapes2d_NxNcandLD_threads.block_y, self.Kshapes2d_NxNcandLD_threads.block_x, self.Kshapes2d_NxNcandLD_threads.block_y, self.Kshapes2d_NxNcandLD_threads.grid_x_size, self.Kshapes2d_NxNcandLD_threads.grid_y_size))

        # ------------ 3. kernels used for the tSNE gradients -------   


        # ------------ 4. Compiling the CUDA kernels -------
        compiler_options = ["-O3", "--use_fast_math", "-prec-div=false", "-ftz=true", "-prec-sqrt=false", "-fmad=true"] # safe arithmetics are for the weak
        self.compiled_minMax_reduction = SourceModule(kernel_minMax_reduction, options=compiler_options)
        self.compiled_X_to_transpose   = SourceModule(kernel_X_to_transpose, options=compiler_options)
        self.compiled_scaling_X        = SourceModule(kernel_scale_X, options=compiler_options)
        self.kernel_all_HD_sqdists_euclidean = SourceModule(kernel_compute_all_HD_sqdists_euclidean, options=compiler_options).get_function("compute_all_HD_sqdists_euclidean")
        self.kernel_all_HD_sqdists_manhattan = SourceModule(kernel_compute_all_HD_sqdists_manhattan, options=compiler_options).get_function("compute_all_HD_sqdists_manhattan")
        self.kernel_all_HD_sqdists_cosine    = SourceModule(kernel_compute_all_HD_sqdists_cosine, options=compiler_options).get_function("compute_all_HD_sqdists_cosine")
        self.kernel_all_HD_sqdists_custom    = SourceModule(kernel_compute_all_HD_sqdists_custom, options=compiler_options).get_function("compute_all_HD_sqdists_custom")
        self.kernel_all_LD_sqdists = SourceModule(kernel_compute_all_LD_sqdists, options=compiler_options).get_function("compute_all_LD_sqdists")

    def free_all_GPU_memory(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # cuda_context.pop() # not needed if pycuda.autoinit is used
        cuda_Xhd.gpudata.free()
        cuda_Xld_true_A.gpudata.free()
        cuda_Xld_true_B.gpudata.free()
        cuda_Xld_nest.gpudata.free()
        cuda_Xld_mmtm.gpudata.free()
        raise Exception("here need to free all CUDA ressources!!")
        1/0
