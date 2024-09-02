import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import & init pycuda
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

__MAX_PP__ = 80
__MAX_K__  = __MAX_PP__ * 3



kernel_code = """
#include <curand_kernel.h>

extern "C" {
__global__ void move_points(float *Xld, int N, int M, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;

    if (idx >= N*M) {
        return;
    }
    
    // Initialize CURAND
    curand_init(seed, idx, 0, &state);

    for (int i = idx; i < N*M; i += stride) {
        // Generate a random number between -0.1 and 0.1
        float rand_val = curand_uniform(&state) * 0.2f - 0.1f;
        Xld[i] += rand_val;
    }
}
}
"""


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
        self.Mld          = np.uint32(n_components)
        self.kern_alpha   = np.float32(1.0)
        self.perplexity   = np.float32(5.0)
        self.dist_metric  = 0
        # result
        self.cpu_Xld  = None
        # variables allocated on the device
        self.cuda_Xhd        = None
        self.cuda_Xld_true   = None # read by the GUI
        self.cuda_Xld_nest   = None
        self.cuda_Xld_mmtm   = None
        # cuda streams
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
        self.cpu_Xld  = (np.random.uniform(size=(N, self.Mld)).astype(np.float32) - 0.5) * 2.0
        # malloc GPU memory
        cuda_Xhd        = gpuarray.to_gpu_async(Xhd)
        cuda_Xld_true_A = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_true_B = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_nest   = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_mmtm   = gpuarray.to_gpu(np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype))
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

    def one_iteration(self):
        module = SourceModule(kernel_code)
        # move_points = module.get_function("move_points")
        """ N, M = self.cuda_Xld_true_A.shape
        stream = cuda.Stream()
        block_size = 256
        num_blocks = (N * M + block_size - 1) // block_size
        seed = np.random.randint(0, 2**32) """
        """ move_points(self.cuda_Xld_true_A, np.int32(N), np.int32(M), np.uint64(seed), block=(block_size, 1, 1), grid=(num_blocks, 1), stream=stream)
        stream.synchronize() """


    def fit_with_gui(self, Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # 1. configure the process launch mode 
        multiprocessing.set_start_method('spawn') # this is crucial for the GUI to work correctly. Python is wierd and often annoying

        # 2. shared memory with GUI (on CPU)
        cpu_shared_mem      = shared_memory.SharedMemory(create=True, size=(self.N * self.Mld * np.dtype(np.float32).itemsize))
        cpu_Xld_arr_on_smem = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem.buf)
        # copy (GPU->CPU) cuda_Xld_true_A and cuda_Xld_true_B to shared memory
        cuda_Xld_true_A.get(cpu_Xld_arr_on_smem)

        # 3.   Launching the process responsible for the GUI
        from fastSNE.fastSNE_gui import gui_worker
        #  Shared hyperparameters
        kernel_alpha   = multiprocessing.Value('f', self.kern_alpha)
        perplexity     = multiprocessing.Value('f', self.perplexity)
        dist_metric    = multiprocessing.Value('i', self.dist_metric)
        #  Shared state variables
        gui_closed     = multiprocessing.Value('b', False)
        points_ready_for_rendering = multiprocessing.Value('b', False)
        points_rendering_finished  = multiprocessing.Value('b', True)
        # 3.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem, Y, self.N, self.Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished))
        process_gui.start()

        # streams related to the GUI
        stream_write_to_CPU = cuda.Stream()

        # 4.   Optimise until the GUI is closed
        isPhaseA       = True
        busy_copying__for_GUI = False
        gui_was_closed = False
        while not gui_was_closed:
            # One iteration of the tSNE optimisation
            self.one_iteration()

            # if the GUI is done rendering the previous frame, feed it data
            gui_done = False
            with points_rendering_finished.get_lock():
                gui_done = points_rendering_finished.value
            if gui_done:
                if busy_copying__for_GUI:
                    busy_copying__for_GUI = False
                    # wait for the previous copy to finish
                    stream_write_to_CPU.synchronize()
                    # notify the GUI that the data is ready
                    with points_rendering_finished.get_lock():
                        points_rendering_finished.value = False
                    with points_ready_for_rendering.get_lock():
                        points_ready_for_rendering.value = True
                else:
                    busy_copying__for_GUI = True
                    # copy contents of cuda_Xld_true_A/B to cpu_Xld_arr_on_smem
                    if isPhaseA:
                        cuda_Xld_true_B.get_async(stream=stream_write_to_CPU, ary=cpu_Xld_arr_on_smem)
                    else:
                        cuda_Xld_true_A.get_async(stream=stream_write_to_CPU, ary=cpu_Xld_arr_on_smem)

            with gui_closed.get_lock():
                gui_was_closed = gui_closed.value
        process_gui.join()

        cpu_shared_mem.unlink()
        self.free_all_GPU_memory(cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm)
        
        return
    
    
    def fit_without_gui(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        1/0

    def free_all_GPU_memory(self, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # cuda_context.pop() # not needed if pycuda.autoinit is used
        cuda_Xhd.gpudata.free()
        cuda_Xld_true_A.gpudata.free()
        cuda_Xld_true_B.gpudata.free()
        cuda_Xld_nest.gpudata.free()
        cuda_Xld_mmtm.gpudata.free()
        
        return
