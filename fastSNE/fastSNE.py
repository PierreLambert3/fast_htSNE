import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import & init pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

__MAX_PP__ = 80
__MAX_K__  = __MAX_PP__ * 3

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
        self.cuda_Xld_true_A = None # read by the GUI
        self.cuda_Xld_true_B = None # read by the GUI
        self.cuda_Xld_nest   = None
        self.cuda_Xld_mmtm   = None
        # cuda streams
        self.stream_neigh_HD = cuda.Stream()
        self.stream_neigh_LD = cuda.Stream()
        self.stream_grads    = cuda.Stream()
    
    def fit(self, N, M, Xhd, Y=None):
        self.N        = N
        self.M        = M
        self.cpu_Xld  = (np.random.uniform(size=(N, self.Mld)).astype(np.float32) - 0.5) * 2.0
        # malloc GPU memory
        cuda_Xhd        = gpuarray.to_gpu_async(Xhd)
        cuda_Xld_true_A = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_true_B = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_nest   = gpuarray.to_gpu(self.cpu_Xld)
        cuda_Xld_mmtm   = gpuarray.to_gpu(np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype))
        
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
        return

    def fit_with_gui(self, Y, cuda_Xhd, cuda_Xld_true_A, cuda_Xld_true_B, cuda_Xld_nest, cuda_Xld_mmtm):
        # 1. configure the process launch mode 
        multiprocessing.set_start_method('spawn') # this is crucial for the GUI to work correctly. Python is wierd.

        # 2. shared memory to pass data between the main process and the GUI
        buffer_size = self.N * self.Mld * np.dtype(np.float32).itemsize
        cpu_shared_mem_A = shared_memory.SharedMemory(create=True, size=buffer_size)
        cpu_shared_mem_B = shared_memory.SharedMemory(create=True, size=buffer_size)
        cpu_Xld_A = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem_A.buf)
        cpu_Xld_B = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem_B.buf)
        # copy (GPU->CPU) cuda_Xld_true_A and cuda_Xld_true_B to shared memory
        cuda_Xld_true_A.get(cpu_Xld_A)
        cuda_Xld_true_B.get(cpu_Xld_B)

        print("self.cpu_Xld_A:   before ", cpu_Xld_A)


        # 3.   Launching the process responsible for the GUI
        from fastSNE.fastSNE_gui import gui_worker
        #  Shared variable on CPU:
        #      hyperparameters
        kernel_alpha      = multiprocessing.Value('f', self.kern_alpha)
        perplexity        = multiprocessing.Value('f', self.perplexity)
        dist_metric       = multiprocessing.Value('i', self.dist_metric)
        #      state variables
        gui_closed        = multiprocessing.Value('b', False)
        isPhaseA          = multiprocessing.Value('b', True)
        points_updated    = multiprocessing.Value('b', True)
        # 3.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem_A, cpu_shared_mem_B, Y, self.N, self.Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated))
        process_gui.start()

        # 4.   Optimise until the GUI is closed
        gui_was_closed = False
        while not gui_was_closed:
            self.one_iteration()

            with points_updated.get_lock():
                points_updated.value = True

            with gui_closed.get_lock():
                gui_was_closed = gui_closed.value
        process_gui.join()
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

""" 
class fastSNE:

    def fit(self, N, M, X, Y=None):
        self.N   = N
        self.M   = M
        self.cpu_Xhd  = X
        self.stream_neigh_HD = pycuda.driver.Stream()
        self.stream_neigh_LD = pycuda.driver.Stream()
        self.stream_grads    = pycuda.driver.Stream()
        self.cuda_Xhd        = gpuarray.to_gpu_async(self.cpu_Xhd)
        self.cpu_Xld         = (np.random.uniform(size=(N, self.n_components)).astype(np.float32) - 0.5) * 2.0
        self.cuda_Xld_true_A = gpuarray.to_gpu(self.cpu_Xld)
        self.cuda_Xld_true_B = gpuarray.to_gpu(self.cpu_Xld)
        self.cuda_Xld_nest   = gpuarray.to_gpu(self.cpu_Xld)
        zeros_array = np.zeros(self.cpu_Xld.shape, self.cpu_Xld.dtype)
        self.cuda_Xld_mmtm   = gpuarray.to_gpu(zeros_array)
        
        if self.with_GUI:
            self.fit_with_gui(Y)
        else:
            self.fit_without_gui()
        self.is_fitted = True

    def transform(self):
        if not self.is_fitted:
            raise Exception("fastSNE: transform() called before fit(), or fit failed crashingly")
        return self.cpu_Xld

    def one_iteration(self):
        return;

    def fit_with_gui(self, Y):
        multiprocessing.set_start_method('spawn')
        from fastSNE.fastSNE_gui import Shared_variables
        shared_variables = Shared_variables(self.cuda_Xld_true_A, self.cuda_Xld_true_B, self.kern_alpha, self.perplexity, self.dist_metric)
        from fastSNE.fastSNE_gui import gui_worker
        cuda_Y = gpuarray.to_gpu(Y)
        process_gui = multiprocessing.Process(target=gui_worker, args=(cuda_Y, shared_variables, self.N, self.M, self.n_components))
        process_gui.start()
        gui_was_closed = False
        while not gui_was_closed:
            self.one_iteration()

            with shared_variables.shared_points_have_moved.get_lock():
                shared_variables.shared_points_have_moved.value = True

            with shared_variables.shared_gui_closed.get_lock():
                gui_was_closed = shared_variables.shared_gui_closed.value
        process_gui.join()
        self.cpu_Xld = self.cuda_Xld_true_A.get()
        self.free_all_GPU_memory()
    
    def fit_without_gui(self):
        for i in range(100):
            self.one_iteration()
        self.cpu_Xld = self.cuda_Xld_true_A.get()
        self.free_all_GPU_memory()

    def free_all_GPU_memory(self):
        self.cuda_Xld_true_A.gpudata.free()
        self.cuda_Xld_true_B.gpudata.free()
        self.cuda_Xld_nest.gpudata.free()
        self.cuda_Xld_mmtm.gpudata.free()
        self.cuda_Xhd.gpudata.free()


        
 """