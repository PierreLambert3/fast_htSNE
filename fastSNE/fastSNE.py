import numpy as np
import multiprocessing


# import & init pycuda
import pycuda.driver as cuda
import pycuda.autoinit

__MAX_PP__ = 80
__MAX_K__  = __MAX_PP__ * 3

class fastSNE:
    def __init__(self, with_GUI, n_components=2, random_state=None):
        self.with_GUI     = with_GUI
        self.is_fitted    = False
        self.random_state = random_state
        if self.random_state is not None and not self.random_state > 0:
            raise Exception("fastSNE: random_state must be a strictly positive integer")
        self.N        = None
        self.Mhd      = None
        self.cpu_Xhd  = None
        self.Mld      = np.uint32(n_components)
        self.kern_alpha   = np.float32(1.0)
        self.perplexity   = np.float32(5.0)
        self.dist_metric  = 0

        self.shared_buffer = None # for GUI

    
    def fit(self, N, M, X, Y=None):
        self.N   = N
        self.M   = M
        self.cpu_Xhd  = X
        
        if self.with_GUI:
            self.fit_with_gui(Y)
        else:
            self.fit_without_gui()
        self.is_fitted = True

    def transform(self):
        if not self.is_fitted:
            raise Exception("fastSNE: transform() called before fit(), or fit failed crashingly")
        # return self.cpu_Xld
        return None

    def one_iteration(self):
        return

    def fit_with_gui(self, Y):
        # 0. configure the process launch mode 
        multiprocessing.set_start_method('spawn')

        # 1.   Launching the process responsible for the GUI
        from fastSNE.fastSNE_gui import gui_worker, Shared_CPU_variables
        # 1.1  Inter-process communication for CUDA structures
        buffer_n_bytes = self.N * __MAX_K__ * 4 # x4 For float32
        self.shared_buffer_A = cuda.mem_alloc(buffer_n_bytes)
        self.shared_buffer_B = cuda.mem_alloc(buffer_n_bytes)
        ipc_handle_Xld_A = cuda.mem_get_ipc_handle(self.shared_buffer_A)
        ipc_handle_Xld_B = cuda.mem_get_ipc_handle(self.shared_buffer_B)
        # 1.2  Shared variable on CPU
        shared_CPU_variables = Shared_CPU_variables(self.kern_alpha, self.perplexity, self.dist_metric)
        print("Serialize Shared Variable: If the Shared_variable class instance is not directly serializable (e.g., for use with multiprocessing.Queue or Pipe), you might need to manually serialize it or ensure it only contains serializable elements.")
        # 1.3  Launching the GUI process proper
        process_gui = multiprocessing.Process(target=gui_worker, args=([ipc_handle_Xld_A, ipc_handle_Xld_B], Y, shared_CPU_variables, self.N, self.Mld))
        process_gui.start()

        # 2.   Optimise until the GUI is closed
        gui_was_closed = False
        while not gui_was_closed:
            self.one_iteration()

            with shared_CPU_variables.points_have_moved.get_lock():
                shared_CPU_variables.points_have_moved.value = True

            with shared_CPU_variables.gui_closed.get_lock():
                gui_was_closed = shared_CPU_variables.gui_closed.value
        process_gui.join()


        self.free_all_GPU_memory()
        print("fastSNE: GUI process has terminated")
        return
    
    def fit_without_gui(self):
        for i in range(100):
            self.one_iteration()
        self.cpu_Xld = self.cuda_Xld_true_A.get()
        self.free_all_GPU_memory()

    def free_all_GPU_memory(self):
        # cuda_context.pop()
        """ self.cuda_Xld_true_A.gpudata.free()
        self.cuda_Xld_true_B.gpudata.free()
        self.cuda_Xld_nest.gpudata.free()
        self.cuda_Xld_mmtm.gpudata.free()
        self.cuda_Xhd.gpudata.free() """
        return

""" 
class fastSNE:
    def __init__(self, with_GUI, n_components=2, random_state=None):
        self.with_GUI     = with_GUI
        self.is_fitted    = False
        self.random_state = random_state
        if self.random_state is not None and not self.random_state > 0:
            raise Exception("fastSNE: random_state must be a strictly positive integer")
        self.N   = None
        self.M   = None
        self.cpu_Xhd  = None
        self.cuda_Xhd = None
        self.cpu_Xld  = None
        self.cuda_Xld_true_A = None
        self.cuda_Xld_true_B = None
        self.cuda_Xld_nest   = None
        self.cuda_Xld_mmtm   = None
        print("For rendering: use directly the data that is on GPU (else GPU->CPU then CPU->GPU: wasteful)")
        self.n_components = np.uint32(n_components)
        self.kern_alpha   = np.float32(1.0)
        self.perplexity   = np.float32(5.0)
        self.dist_metric  = 0

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