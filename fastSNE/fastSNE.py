import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import & init pycuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

from fastSNE.cuda_kernels import kernel_minMax_reduction, kernel_X_to_transpose, kernel_scale_X

__MAX_PERPLEXITY__ = 80.0
__MIN_PERPLEXITY__ = 1.5
__MAX_K__  = __MAX_PERPLEXITY__ * 3
__MAX_KERNEL_ALPHA__ = 100.0
__MIN_KERNEL_ALPHA__ = 0.05
__MAX_ATTRACTION_MULTIPLIER__ = 10.0
__MIN_ATTRACTION_MULTIPLIER__ = 0.1

class Kernel_shapes:
    def __init__(self, N_threads_total, threads_per_block_multiple_of, smem_n_float32_per_thread, cuda_device_attributes):
        max_threads_per_block = cuda_device_attributes[cuda.device_attribute.MAX_THREADS_PER_BLOCK]
        max_shared_memory_per_block = cuda_device_attributes[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
        max_grid_x = cuda_device_attributes[cuda.device_attribute.MAX_GRID_DIM_X]
        max_grid_y = cuda_device_attributes[cuda.device_attribute.MAX_GRID_DIM_Y]
        # find the number of threads per block: start with threads_per_block_multiple_of, and add threads_per_block_multiple_of until one of the constraints is violated
        threads_per_block = threads_per_block_multiple_of
        smem_n_bytes_per_block = threads_per_block * smem_n_float32_per_thread * np.dtype(np.float32).itemsize
        n_blocks  = (N_threads_total + threads_per_block - 1) // threads_per_block
        tpb_ok    = threads_per_block <= max_threads_per_block
        smem_ok   = smem_n_bytes_per_block <= max_shared_memory_per_block
        while True:
            if threads_per_block >= N_threads_total:
                break
            next_threads_per_block      = threads_per_block + threads_per_block_multiple_of
            next_smem_n_bytes_per_block = next_threads_per_block * smem_n_float32_per_thread * np.dtype(np.float32).itemsize
            next_n_blocks               = (N_threads_total + next_threads_per_block - 1) // next_threads_per_block
            next_tpb_ok                 = next_threads_per_block <= max_threads_per_block
            next_smem_ok                = next_smem_n_bytes_per_block <= max_shared_memory_per_block
            if next_tpb_ok and next_smem_ok:
                threads_per_block = next_threads_per_block
                smem_n_bytes_per_block = next_smem_n_bytes_per_block
                n_blocks = next_n_blocks
                tpb_ok = next_tpb_ok
                smem_ok = next_smem_ok
            else:
                break 
        # save the results
        self.threads_per_block      = threads_per_block
        self.smem_n_bytes_per_block = smem_n_bytes_per_block
        self.grid_x_size            = n_blocks
        self.grid_y_size            = 1

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
        # variables allocated on the device
        self.cuda_Xhd        = None
        self.cuda_Xld_true   = None # read by the GUI
        self.cuda_Xld_nest   = None
        self.cuda_Xld_mmtm   = None
        # cuda streams
        self.stream_minMax   = cuda.Stream() # used in GUI mode
        self.stream_neigh_HD = cuda.Stream()
        self.stream_neigh_LD = cuda.Stream()
        self.stream_grads    = cuda.Stream()
    
    def fit(self, N, M, Xhd, Y=None):
        # check input data
        if N < 5:
            raise Exception("fastSNE: the number of samples N must be at least 2")
        if M < 2:
            raise Exception("fastSNE: the number of dimensions M must be at least 2")
        if np.isnan(Xhd).any():
            raise Exception("fastSNE: the high-dimensional data contains NaNs")
        # on CPU
        self.N        = N
        self.M        = M
        self.cpu_Xld  = ((np.random.uniform(size=(N, self.Mld)).astype(np.float32) - 0.5) * 2.0) * 4.2
        # malloc GPU memory
        cuda_Xhd        = gpuarray.to_gpu_async(Xhd)
        cuda_Xld_true_A = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_true_B = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_nest   = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_mmtm   = gpuarray.to_gpu(np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype))

        self.configue_and_initialise_CUDA_kernels_please()

        # launch the tSNE optimisation
        if self.with_GUI:
            self.fit_with_gui(Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        else:
            self.fit_without_gui(cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        self.is_fitted = True

    def transform(self):
        if not self.is_fitted:
            raise Exception("fastSNE: transform() called before fit(), or fit failed crashingly")
        # return self.cpu_Xld
        return None

    def one_iteration(self, cuda_Xld_true):
        return

    def fit_with_gui(self, Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
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
        gui_closed     = multiprocessing.Value('b', False)
        points_ready_for_rendering = multiprocessing.Value('b', False)
        points_rendering_finished  = multiprocessing.Value('b', True)
        # 3.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem, Y, self.N, self.Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished))
        process_gui.start()

        # 4.   Optimise until the GUI is closed
        isPhaseA       = True
        gui_data_prep_phase = 0
        busy_copying__for_GUI = False
        gui_was_closed = False
        while not gui_was_closed:
            # sync all streams (else read/writes my conflict with versions A and B)
            self.stream_minMax.synchronize()
            self.stream_neigh_HD.synchronize()
            self.stream_neigh_LD.synchronize()
            self.stream_grads.synchronize()

            # One iteration of the tSNE optimisation
            if isPhaseA:
                self.one_iteration(cuda_Xld_true_A)
            else:
                self.one_iteration(cuda_Xld_true_B)
            
            # on phase A : write to A, read from B

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
                cuda_Xld_temp_Xld.set_async(Xld_to_transpose, stream=self.stream_minMax)
                self.compiled_X_to_transpose.get_function("kernel_X_to_transpose")(Xld_to_transpose, cuda_Xld_T_temp_lvl1_maxs, np.uint32(self.N), np.uint32(self.Mld), block=(self.Kshapes_transpose.threads_per_block, 1, 1), grid=(self.Kshapes_transpose.grid_x_size, self.Kshapes_transpose.grid_y_size), stream=self.stream_minMax)
                cuda_Xld_T_temp_lvl1_mins.set_async(cuda_Xld_T_temp_lvl1_maxs, stream=self.stream_minMax)

            elif gui_data_prep_phase == 1: # perform the min-max reduction on cuda_Xld_temp, & scale the data to [0, 1] with the results
                self.scaling_of_points(cuda_Xld_temp_Xld, cuda_Xld_T_temp_lvl1_mins, cuda_Xld_T_temp_lvl1_maxs)

            else: # copy cuda_Xld_temp to cpu_Xld_arr_on_smem, if the GUI is ready
                gui_done = False
                busy_copying__for_GUI = False
                with points_rendering_finished.get_lock():
                    gui_done = points_rendering_finished.value
                if gui_done:
                    cuda_Xld_temp_Xld.get_async(stream=self.stream_minMax, ary=cpu_Xld_arr_on_smem)
                    busy_copying__for_GUI = True
            gui_data_prep_phase = (gui_data_prep_phase + 1) % 3

            # switch phase
            isPhaseA = not isPhaseA

            with gui_closed.get_lock():
                gui_was_closed = gui_closed.value
        process_gui.join()

        cpu_shared_mem.unlink()
        self.free_all_GPU_memory(cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        
        return
    
    
    def fit_without_gui(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        1/0


    def scaling_of_points(self, cuda_Xld_temp_Xld, cuda_Xld_temp_lvl1_mins, cuda_Xld_temp_lvl1_maxs):
        cuda_kernel = self.compiled_minMax_reduction.get_function("perform_minMax_reduction")
        stream      = self.stream_minMax
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

    def configue_and_initialise_CUDA_kernels_please(self):
        N, M, Mld = self.N, self.M, self.Mld
        cuda_device = cuda.Device(0)
        cuda_device_attributes = cuda_device.get_attributes()

        # ------------ 0. kernels used for getting the transpose of Xld  -------
        n_threads = N
        multiple_of = 32 if n_threads > 32 else 1
        smem_n_float32_per_thread = 1
        self.Kshapes_transpose = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes)
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
        self.Kshapes_minMax_lvl_1 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes)
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
            self.Kshapes_minMax_lvl_2 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes)
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
                self.Kshapes_minMax_lvl_3 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes)
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
                    self.Kshapes_minMax_lvl_4 = Kernel_shapes(n_threads, multiple_of, smem_n_float32_per_thread, cuda_device_attributes)
                    self.Kshapes_minMax_lvl_4.grid_y_size = Mld
                    # the array containing the results of the reduction
                    self.perdim_remaining_after_reduction4 = self.Kshapes_minMax_lvl_4.grid_x_size
                    self.reduction4_result_mins = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction4, Mld), dtype=np.float32))
                    self.reduction4_result_maxs = gpuarray.to_gpu(np.zeros((self.perdim_remaining_after_reduction4, Mld), dtype=np.float32))
                    if self.perdim_remaining_after_reduction4 > 1:
                        raise Exception("Splendid, you have more that 1e12 points in your dataset. This hardcoded limit was written in the past where such large datasets were not common. Contact me by e-mail or by cyberpigeon, whichever is the norm in your time.")

        # ------------ 2. kernels used for the tSNE optimisation -------
        # TODO

        # ------------ 3. Compiling the CUDA kernels -------
        compiler_options = ["-O3", "--use_fast_math", "-prec-div=false", "-ftz=true", "-prec-sqrt=false", "-fmad=true"] # safe arithmetics are for the weak
        self.compiled_minMax_reduction = SourceModule(kernel_minMax_reduction, options=compiler_options)
        self.compiled_X_to_transpose   = SourceModule(kernel_X_to_transpose, options=compiler_options)
        self.compiled_scaling_X        = SourceModule(kernel_scale_X, options=compiler_options)

    def free_all_GPU_memory(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # cuda_context.pop() # not needed if pycuda.autoinit is used
        cuda_Xhd.gpudata.free()
        cuda_Xld_true_A.gpudata.free()
        cuda_Xld_true_B.gpudata.free()
        cuda_Xld_nest.gpudata.free()
        cuda_Xld_mmtm.gpudata.free()
        
        return
