# a class that will contain the data and the results of the fastSNE algorithm
import numpy as np
import multiprocessing

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

class fastSNE:
    def __init__(self, with_GUI, n_components=2, random_state=None):
        # model state
        self.with_GUI     = with_GUI
        self.is_fitted    = False
        self.random_state = random_state
        if self.random_state is not None and not self.random_state > 0:
            raise Exception("fastSNE: random_state must be a strictly positive integer")
        # dataset info 	8.9
        self.N   = None
        self.M   = None
        self.cpu_Xhd  = None
        self.cuda_Xhd = None
        # model results
        self.cpu_Xld  = None
        # optimiser
        self.cuda_Xld_true_A = None
        self.cuda_Xld_true_B = None
        self.cuda_Xld_nest   = None
        self.cuda_Xld_mmtm   = None
        print("For rendering: use directly the data that is on GPU (else GPU->CPU then CPU->GPU: wasteful)")
        # algorithm parameters
        self.n_components = np.uint32(n_components)
        self.kern_alpha   = np.float32(1.0)
        self.perplexity   = np.float32(5.0)
        self.dist_metric  = 0 # 0: euclidean, 1: cosine, 2: manhattan

    def fit(self, N, M, X, Y=None):
        self.N   = N
        self.M   = M
        self.cpu_Xhd  = X
        self.cuda_Xhd = gpuarray.to_gpu(self.cpu_Xhd)

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
        """ W = np.random.randn(self.M, self.n_components)
        self.Xld_true_A = np.dot(self.Xhd, W) """
        pass

    def fit_with_gui(self, Y): # Y is for visualisation only, it doesn't intervene in the algorithm
        multiprocessing.set_start_method('spawn')
        # initialise the shared variables between the GUI and the worker
        from fastSNE.fastSNE_gui import Shared_variables
        shared_variables = Shared_variables(self.Xld_true_A, self.Xld_true_B, self.kern_alpha, self.perplexity, self.dist_metric)
        # start the GUI
        from fastSNE.fastSNE_gui import gui_worker
        process_gui = multiprocessing.Process(target=gui_worker, args=(Y, shared_variables, self.N, self.M, self.n_components))
        process_gui.start()
        # run till gui is closed
        gui_was_closed = False
        while not gui_was_closed:
            # one iteration of the algorithm
            self.one_iteration()

            """ # tell the GUI that points have moved
            with shared_variables.shared_points_have_moved.get_lock():
                shared_variables.shared_points_have_moved.value = True """

            # check if the GUI was closed
            with shared_variables.shared_gui_closed.get_lock():
                gui_was_closed = shared_variables.shared_gui_closed.value
        process_gui.join()
        # synchronuous copy (GPU->CPU)
        self.cpu_Xld = self.cuda_Xld_true_A.get()
        # free all the GPU memory
        self.free_all_GPU_memory()
    
    def fit_without_gui(self):
        for i in range(100):
            self.one_iteration()
        # synchronuous copy (GPU->CPU)
        self.cpu_Xld = self.cuda_Xld_true_A.get()
        # free all the GPU memory
        self.free_all_GPU_memory()

    def free_GPU_memory(self):
        self.cuda_Xld_true_A.gpudata.free()
        self.cuda_Xld_true_B.gpudata.free()
        self.cuda_Xld_nest.gpudata.free()
        self.cuda_Xld_mmtm.gpudata.free()
        self.cuda_Xhd.gpudata.free()


        
