import numpy as np
def r():
    return np.random.uniform(low=0.0, high=1.0)

import time
__MAX_UINT32_T__ = np.uint32(np.iinfo(np.uint32).max)

import multiprocessing
from multiprocessing import shared_memory
from .pycuda_utils import GPU_hidden_wrapper, CUDA_kernel, Kernel_shapes_2dBlocks, Kernel_shapes

__CUDA_CODE__ = "no code"
with open("fast_htSNE/kernels.cu", "r") as f:
    __CUDA_CODE__ = f.read()

__DEVICE_NUMBER__ = 0 # the GPU device to use
__MIN_PERPLEXITY__ = 1.5
__MAX_KERNEL_ALPHA__ = 100.0
__MIN_KERNEL_ALPHA__ = 0.05
__MAX_ATTRACTION_MULTIPLIER__ = 1.0
__MIN_ATTRACTION_MULTIPLIER__ = 0.05
__PCT_HISTORY_SIZE__  = 500
MIN_NITER = 3000

# these are defined in the compiled side of the project (in cuda_kernels.py)
__MAX_PERPLEXITY__ = None
__Khd__       = None
__Kld__       = None
__N_CAND_LD__ = None
__N_CAND_HD__ = None
__N_INTERACTIONS_FAR__ = None

class _Kernels:
    def __init__(self, gpu_ctx, compiled, N, Mhd, Mld):
        self.compiled = compiled



        # compute_all_LD_sqdists Kernel 
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __Kld__, N_threads_block_x=__Kld__, smem_n_float32_per_thread=2,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=1, smem_n_float32_per_block_y=Mld)
        self.compute_all_LD_sqdists = CUDA_kernel("compute_all_LD_sqdists", self.compiled, cuShape)

        # candidates_LD_generate_and_sort 
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __N_CAND_LD__, N_threads_block_x=__N_CAND_LD__, smem_n_float32_per_thread=4,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mld + 1)
        self.candidates_LD_generate_and_sort = CUDA_kernel("candidates_LD_generate_and_sort", self.compiled, cuShape)


        # all HS sqdist _ metric
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __Khd__, N_threads_block_x=__Khd__, smem_n_float32_per_thread=3,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mhd + 1)
        self.all_HD_sqdists_euclidean = CUDA_kernel("compute_all_HD_sqdists_euclidean", self.compiled, cuShape)
        self.all_HD_sqdists_manhattan = CUDA_kernel("compute_all_HD_sqdists_manhattan", self.compiled, cuShape)
        self.all_HD_sqdists_cosine    = CUDA_kernel("compute_all_HD_sqdists_cosine", self.compiled, cuShape)
        self.all_HD_sqdists_custom    = CUDA_kernel("compute_all_HD_sqdists_custom", self.compiled, cuShape)

        # flag_all_newNeighs
        multiple_of = 32 if N > 32 else 1
        cuShape = Kernel_shapes(N_threads_total=N, threads_per_block_multiple_of=multiple_of, smem_n_float32_per_thread=1,\
                                cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0)
        self.flag_new_neighbours_for_all = CUDA_kernel("kernel_flag_all_newNeighs", self.compiled, cuShape)

        # filtered re computation of P
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __Khd__, N_threads_block_x=__Khd__, smem_n_float32_per_thread=3,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mhd + 1)
        self.radii_P_part1 = CUDA_kernel("kernel_radii_P_part1", self.compiled, cuShape)
        self.radii_P_part2 = CUDA_kernel("kernel_radii_P_part2", self.compiled, cuShape)

        # scaling_Xld
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * Mld, N_threads_block_x=Mld, smem_n_float32_per_thread=0,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=0)
        self.scale_Xld = CUDA_kernel("kernel_scale_X", self.compiled, cuShape)


        # generate and sort candidates HD
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __N_CAND_HD__, N_threads_block_x=__N_CAND_HD__, smem_n_float32_per_thread=4,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mhd + 1)
        self.candidates_HD_generate = CUDA_kernel("candidates_HD_generate", self.compiled, cuShape)
        
        # recompute farthest distances
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __Khd__, N_threads_block_x=__Khd__, smem_n_float32_per_thread=3,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mhd + 1)
        self.recompute_HD_farthest_distances = CUDA_kernel("kernel_HD_redetermine_farthest_dists_and_sort", self.compiled, cuShape)



        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * Mld, N_threads_block_x=Mld, smem_n_float32_per_thread=0,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=0)
        self.make_Xnesterov = CUDA_kernel("kernel_make_Xnesterov", self.compiled, cuShape)
        self.receive_gradients     = CUDA_kernel("receive_gradients", self.compiled, cuShape)

        # computing gradients
        cuShape = Kernel_shapes_2dBlocks(N_threads_total=N * __Khd__, N_threads_block_x=__Khd__, smem_n_float32_per_thread=3,\
                                            cuda_device_attributes=gpu_ctx.get_device_attributes(), constant_additional_smem_n_float32=0, smem_n_float32_per_block_y=Mhd + 1)
        self.compute_gradients = CUDA_kernel("kernel_gradients", self.compiled, cuShape)
       
class _Streams:
    def __init__(self, gpu_ctx):
        self.stream_neigh_HD = gpu_ctx.stream()
        self.stream_neigh_LD = gpu_ctx.stream()
        self.stream_minMax1   = gpu_ctx.stream()
        self.stream_minMax2   = gpu_ctx.stream()
        self.stream_grads    = gpu_ctx.stream()

        self.generic_stream1 = gpu_ctx.stream()
        self.generic_stream2 = gpu_ctx.stream()
        self.generic_stream3 = gpu_ctx.stream()

    def sync_all(self):
        self.stream_neigh_HD.synchronize()
        self.stream_neigh_LD.synchronize()
        self.stream_minMax1.synchronize()
        self.stream_minMax2.synchronize()
        self.stream_grads.synchronize()
        self.generic_stream1.synchronize()
        self.generic_stream2.synchronize()
        self.generic_stream3.synchronize()

"""
An ugly dump of allocated memory, mostly on the GPU
"""
class _Optimisation_structures:
    def __init__(self, gpu_ctx, compiled_code, N, Mhd, X, Y, Mld):
        self.Wlinproj_now    = np.random.randn(Mhd, Mld).astype(np.float32)
        self.Wlinproj_target = np.random.randn(Mhd, Mld).astype(np.float32)
        self.cpu_Xld         = np.random.randn(N, Mld).astype(np.float32) * 1e-4
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
        # mallocs on the device. A/B things are for double buffering (write on one while reading the other, alternating at each iteration)
        self.cu_Xhd = gpu_ctx.malloc(X)
        self.cu_knn_HD_A= gpu_ctx.malloc(init_HD_neighs)
        self.cu_knn_HD_B= gpu_ctx.malloc(init_HD_neighs)
        self.cu_sqdists_HD_A = gpu_ctx.malloc(np.zeros((N, __Khd__), dtype=np.float32))
        self.cu_sqdists_HD_B = gpu_ctx.malloc(np.zeros((N, __Khd__), dtype=np.float32))
        self.cu_far_dist_HD_A = gpu_ctx.malloc(np.ones(N, dtype=np.float32))
        self.cu_far_dist_HD_B = gpu_ctx.malloc(np.ones(N, dtype=np.float32))
        self.cu_grad_acc_global = gpu_ctx.malloc(np.zeros((N, Mld), dtype=np.float32))
        self.cu_has_new_HD_neighs = gpu_ctx.malloc(np.ones(N, dtype=np.uint32)) # todo: bool* or uint8_t*
        self.cu_has_new_HD_neighs_acc = gpu_ctx.malloc(np.zeros(N, dtype=np.uint32))
        self.cu_invRadii_HD = gpu_ctx.malloc(np.ones(N, dtype=np.float32))
        self.cu_Pasm = gpu_ctx.malloc(np.zeros((N, __Khd__), dtype=np.float32))
        self.cu_Pasm_sums = gpu_ctx.malloc(np.ones(N, dtype=np.float32))
        self.cu_Psym = gpu_ctx.malloc(np.ones((N, __Khd__), dtype=np.float32))
        self.cu_Psym_knn = gpu_ctx.malloc(np.ones((N, __Khd__), dtype=np.uint32))
        self.cu_Xld_true_A = gpu_ctx.malloc(self.cpu_Xld)
        self.cu_Xld_true_B = gpu_ctx.malloc(self.cpu_Xld)
        self.cu_Xld_nest = gpu_ctx.malloc(np.zeros_like(self.cpu_Xld))
        self.cu_Xld_mmtm = gpu_ctx.malloc(np.zeros_like(self.cpu_Xld))
        self.cu_knn_LD_A = gpu_ctx.malloc(init_LD_neighs)
        self.cu_knn_LD_B = gpu_ctx.malloc(init_LD_neighs)
        self.cu_sqdists_LD_A = gpu_ctx.malloc(np.zeros((N, __Kld__), dtype=np.float32))
        self.cu_sqdists_LD_B = gpu_ctx.malloc(np.zeros((N, __Kld__), dtype=np.float32))
        self.cu_far_dist_LD_A = gpu_ctx.malloc(np.ones(N, dtype=np.float32))
        self.cu_far_dist_LD_B = gpu_ctx.malloc(np.ones(N, dtype=np.float32))

        from .pycuda_utils import SumGpu, MinGpu, MaxGpu
        self.neighbours_sumSnorms_LD   = SumGpu(np.double, N, compiled_code, gpu_ctx.get_device_attributes())
        self.randoms_sumSnorms_LD      = SumGpu(np.double, N, compiled_code, gpu_ctx.get_device_attributes())
        self.HD_n_new_neighs_sum       = SumGpu(np.uint32, N, compiled_code, gpu_ctx.get_device_attributes())
        self.gui_Xld_minFinder         = MinGpu(np.float32, N*Mld, compiled_code, gpu_ctx.get_device_attributes())
        self.gui_Xld_maxFinder         = MaxGpu(np.float32, N*Mld, compiled_code, gpu_ctx.get_device_attributes())
    
    # cu_knn_HD, cu_sqdists_HD, cu_far_dist_HD, cu_Xld_true, cu_knn_LD, cu_sqdists_LD, cu_far_dist_LD
    def get_readWrite_variables(self, is_phaseA):
        A_set = self.cu_knn_HD_A, self.cu_sqdists_HD_A, self.cu_far_dist_HD_A, self.cu_Xld_true_A, self.cu_knn_LD_A, self.cu_sqdists_LD_A, self.cu_far_dist_LD_A
        B_set = self.cu_knn_HD_B, self.cu_sqdists_HD_B, self.cu_far_dist_HD_B, self.cu_Xld_true_B, self.cu_knn_LD_B, self.cu_sqdists_LD_B, self.cu_far_dist_LD_B
        if is_phaseA: # read A, write B
            return A_set, B_set
        else:         # read B, write A
            return B_set, A_set
    
    def free_old_structures(self):
        self.neighbours_sumSnorms_LD.free()
        self.randoms_sumSnorms_LD.free()
        self.HD_n_new_neighs_sum.free()
        self.gui_Xld_minFinder.free()
        self.gui_Xld_maxFinder.free()

class htSNE:    
    def __init__(self, with_gui=True, n_components=2, verbose=True):
        self.window = None
        # 1. context and helper for the GPU
        if verbose:
            print("\033[38;2;255;165;0m \nfast h-t-SNE: initialising the GPU context & compiling CUDA code...\033[0m", end='')
        self._gpu_hidden_wrapper = GPU_hidden_wrapper() # don't touch this
        self.gpu_context        = self._gpu_hidden_wrapper.make_context("fastSNE") # This guy handles the GPU context. All mallocs are done through here, for easy memory management
        self.compiled           = self.gpu_context.compile(__CUDA_CODE__)
        if verbose:
            print("\033[32m   DONE\033[0m")
        # 2. some constants are initialised in the CUDA code, for compiler visibility. Fetch these constants.
        self.fetch_constants_from_cuda(self.gpu_context, self.compiled)
        # 3. state variables
        self.is_fitted = False
        self.with_gui  = with_gui
        self.verbose   = verbose
        # 4. things supposed to be read from the outside
        self.knn_HD = None  # this is the KNN of Xhd, on the CPU (only loaded at the end of the fit)
        self.Xld = None  # this is the embedding, on the CPU (only loaded at the end of the fit)
        # 5. initialise hyperparameters
        self.Mld           = np.uint32(n_components); assert self.Mld >= 2
        self.kern_alpha    = np.float32(1.0)
        self.perplexity    = np.float32(35.0)
        self.attrac_mult   = np.float32(0.5)
        self.lr            = None
        self.lr_multiplier = np.float32(10.0)
        self.dist_metric   = 0
        self.check_yourself_init()
        self.last_frame_time = time.time()
        self.frame_counter = 0
        self.render_time_ema = 0.1

    """
    
    in tSNE, for very large datasets, the local repulsions can overwhelm the global attractions and we end up with a near-uniform disk of points.
    To counter that, try setting base_attraction_repulsion_ratio to a alarge value, and also setting "attraction" in the gui to, for instance, 0.95.
    """
    def fit(self, Xhd_raw, Y=None, purpose_is_KNN=False, max_n_sec=None, max_n_iter=None, lr_strength=None, with_warmup=True,\
                end_PP=None, end_kernel_alpha=None, end_attrac_mult=None, base_attraction_repulsion_ratio=1.0, warmup_n_iter=None):
        if end_PP is not None:
            self.end_PP = np.float32(end_PP)
        else:
            self.end_PP = np.float32(35.0)
        if end_kernel_alpha is not None:
            self.end_kernel_alpha = np.float32(end_kernel_alpha)
        else:
            self.end_kernel_alpha = np.float32(1.0)
        if end_attrac_mult is not None:
            self.end_attrac_mult = np.float32(end_attrac_mult)
        else:
            self.end_attrac_mult = np.float32(0.5)
        self.repulsion_base = np.float32(1.0 / base_attraction_repulsion_ratio)
        
        self.perplexity  = np.float32(self.perplexity)
        self.kern_alpha  = np.float32(self.kern_alpha)
        self.attrac_mult = np.float32(self.attrac_mult)
        self.user_set_warmup_n_iter = warmup_n_iter

        # 1. check if the algorithm is limited in time or in iterations or not at all
        limit_by_time, limit_by_niter, max_n_sec, max_n_iter = self.detemine_optimisation_termination_criterion(max_n_sec, max_n_iter)
        # 2. dataset properties
        if self.verbose:
            print("\033[38;2;255;165;0m \nVerifying dataset compatibility...                                 \033[0m", end='')
        N, M, Xhd_preprocessed, Y = self.sanitise_and_get_input_properties(Xhd_raw, Y)
        self.N   = np.uint32(N)
        self.Mhd = np.uint32(M)
        self.purpose_is_KNN = purpose_is_KNN # if True, then one of the purpose of the algorithm is to compute the KNN of Xhd. We therefore search for new KNN at every iteration
        
        """ lr_left  = np.float32(0.2  * float(N) / 12.0)
        lr_right = np.float32(0.1 * float(N) / 12.0)
        c       = 100 * 1000
        softmax = 1.0 / (1.0 + np.exp(-10.0*(float(N)-c)/c))
        self.lr = np.float32(lr_left * (1.0 - softmax) + lr_right * (softmax)) """
        """ x = float(N) / (100.0*1000.0)
        gx = np.float32(np.sqrt(x) / (0.3 + x))
        self.lr = np.float32(0.5*(gx*gx) * float(N)/12.0) """

        self.lr = np.float32(0.3 * float(N)/12.0)
        n = self.N
        while n > 100 * 1000:
            self.lr = np.float32(self.lr * 0.4)
            n = n // 2

        if lr_strength is not None:
            self.lr = np.float32(lr_strength * self.lr)
        self.init_lr = self.lr
        
        self.check_yourself_fit(Xhd_preprocessed, Y)
        if self.verbose:
            print("\033[32m   DONE\033[0m")
        # 3. initialise the optimisation structures
        self.kernels  = _Kernels(self.gpu_context, self.compiled, N, M, int(self.Mld))
        self.streams  = _Streams(self.gpu_context)
        if self.verbose:
            print("\033[38;2;255;165;0m \nClaiming zeros and ones on the GPU...                               \033[0m", end='')
        optimisation_structures = _Optimisation_structures(self.gpu_context, self.kernels.compiled, N, M, Xhd_preprocessed, Y, self.Mld)
        if self.verbose:
            print("\033[32m   DONE\033[0m")
        # 4. launch the optimisation
        self.optimisation_preliminaries(optimisation_structures)
        self.Xld_gpu = optimisation_structures.cu_Xld_true_A
        if self.with_gui:
            self.optimise_with_gui(Xhd_preprocessed, optimisation_structures, limit_by_time, limit_by_niter, max_n_sec, max_n_iter, Y, with_warmup)
        else:
            if max_n_sec is None and max_n_iter is None:
                print("\033[38;2;255;165;0mWARNING\033[0m:  the algorithm is not limited in time nor in iterations, and withou GUI: setting niter to 3000")
                max_n_iter = 3000
                limit_by_niter = True
            print("\033[38;2;255;165;0m Running the fast t-SNE algorithm without the GUI (not recomended)  \033[0m")
            self.optimise_soberly(Xhd_preprocessed, optimisation_structures, limit_by_time, limit_by_niter, max_n_sec, max_n_iter, Y, with_warmup)
        optimisation_structures.free_old_structures() # the legacy async sum/min/max structures are not managed by gpu_context: need to free them explicitely
        # 5. fetch the results
        self.Xld    = self.Xld_gpu.get()
        self.knn_HD = optimisation_structures.cu_knn_HD_A.get()
        self.is_fitted = True

        # 6. free the memory
        self.gpu_context.free_context()

        return self.Xld, self.knn_HD

    def detemine_optimisation_termination_criterion(self, max_n_sec, max_n_iter):
        limit_by_time  = max_n_sec  is not None
        limit_by_niter = max_n_iter is not None
        if limit_by_time and limit_by_niter:
            print("\033[38;2;255;165;0mWARNING\033[0m:  max_n_sec and max_n_iter are both set. max_n_sec will be ignored")
            limit_by_time  = False
        if limit_by_niter and max_n_iter < MIN_NITER:
            print("\033[38;2;255;165;0mWARNING\033[0m:  max_n_iter is less than the minimum number of iterations (3000). It will be set to 3000")
            max_n_iter = MIN_NITER
        return limit_by_time, limit_by_niter, max_n_sec, max_n_iter
    
    def sanitise_and_get_input_properties(self, Xhd, Y):
        # make sure that Xhd is float32_t
        if Xhd.dtype != np.float32:
            print("\033[38;2;255;165;0mWARNING\033[0m:  Xhd is not float32. Casting to float32")
            Xhd = Xhd.astype(np.float32)
        # max input dimensionality is 256
        N, M = Xhd.shape
        if M > 256:
            print("\033[38;2;255;105;180mWARNING:  the number of dimensions M is greater than 256. Currently 256 is the max, PCA is performed first to get M to 256. Consider reducing the number of dimensions (for instance, use feature selection techniques) to avoid this PCA step  \033[0m")
            M = 256
            from sklearn.decomposition import PCA
            Xhd = PCA(n_components=M).fit_transform(Xhd)
            N, M = Xhd.shape
        """ # make sure that Y is uint32_t
        if Y is not None and Y.dtype != np.uint32:
            print("\033[38;2;255;165;0mWARNING\033[0m:  Y is not uint32. Casting to uint32")
            Y = Y.astype(np.uint32) """
        # make sure that Y has a dummy 2nd dimension
        if Y is not None:
            if len(Y.shape) == 1:
                Y = Y.reshape((-1, 1))
            Y_is_RBG = (not N == Y.ravel().shape[0] )
            if not Y_is_RBG:
                if Y.min() != 0:
                    Y -= Y.min()
                Y = Y.reshape((-1, 1))
        return N, M, Xhd, Y

    def check_yourself_init(self):
        def wreck_yourself(message):
            raise Exception("\033[38;2;255;0;0mERROR: " + message + "\033[0m")
        if __Khd__ < 32 or __Kld__ < 32:
            wreck_yourself("fastSNE: __Khd__ and __Kld__ must be at least 32. Change the values of __Khd__ and __Kld__ in the .cu file")
        if (__Kld__ % 32) != 0:
            wreck_yourself(" __Kld__ is not a multiple of 32. This will result in inefficient memory access patterns. Consider changing the value of __Kld__ in the .cu file")
        if (__Khd__ % 32) != 0:
            wreck_yourself("__Khd__ is not a multiple of 32. This will result in inefficient memory access patterns. Consider changing the value of __Khd__ in the .cu file")
        if (__Kld__ % 2) != 0:
            wreck_yourself("__Kld__ has to be a multiple of 2 (and preferably a multiple of 32 as well). Change the value of __Kld__ in fastSNE.py")
        if (__Khd__ % 2) != 0:
            wreck_yourself("__Khd__ has to be a multiple of 2 (and preferably a multiple of 32 as well). Change the value of __Khd__ in fastSNE.py")
        if(__Khd__ < (__N_INTERACTIONS_FAR__ + __Kld__)):
            wreck_yourself("__Khd__ must be at least (__N_INTERACTIONS_FAR__ + __Kld__). Change the value of __Khd__ in fastSNE.py")
        assert self.dist_metric in [0, 1, 2, 3]
        assert self.kern_alpha < __MAX_KERNEL_ALPHA__ and self.kern_alpha > __MIN_KERNEL_ALPHA__
        assert self.perplexity < __MAX_PERPLEXITY__ and self.perplexity > __MIN_PERPLEXITY__
        assert self.attrac_mult < __MAX_ATTRACTION_MULTIPLIER__ and self.attrac_mult > __MIN_ATTRACTION_MULTIPLIER__

    def check_yourself_fit(self, Xhd, Y):
        def wreck_yourself(message):
            raise Exception("\033[38;2;255;0;0mERROR: " + message + "\033[0m")
        N, M = Xhd.shape
        if Xhd[0, 0].dtype != np.float32:
            wreck_yourself("X must be float32")
        if (Y is not None) and not (Y.dtype == np.uint32 or Y.dtype == np.float32):
            wreck_yourself("Y must be float32 or uint32")
        if N < 15:
            wreck_yourself("the number of samples N must be at least 15")
        if M < 2:
            wreck_yourself("the number of dimensions M must be at least 2")
        if np.isnan(Xhd).any():
            wreck_yourself("the high-dimensional data contains NaNs")
        if __Khd__ >= (N-10):
            wreck_yourself("the number of neighbours K is too large for the number of samples N (reducting __MAX_PERPLEXITY__ should do the trick)")
        if __N_CAND_LD__ < 32:
            wreck_yourself("__N_CAND_LD__ must be at least 32 (and preferably a multiple of 32). Change the values of __N_CAND_LD__ and __N_CAND_HD__ in kernels.cu")
        if __N_CAND_LD__ > __Kld__:
            wreck_yourself("__N_CAND_LD__ must be smaller than __Kld__ (the number of neighbours in LD). Change the value of __N_CAND_LD__ in kernels.cu")
        if __N_CAND_HD__ > __Khd__:
            wreck_yourself("__N_CAND_HD__ must be smaller than __Khd__ (the number of neighbours in HD). Change the value of __N_CAND_HD__ in kernels.cu")
        if __N_CAND_HD__ % 32 != 0:
            wreck_yourself("__N_CAND_HD__ must be a multiple of 32. Change the value of __N_CAND_HD__ in kernels.cu")
        if __N_CAND_HD__ < 16:
            wreck_yourself("__N_CAND_HD__ must be at least 16 (and preferably a multiple of 32). Change the values of __N_CAND_LD__ and __N_CAND_HD__ in kernels.cu")
        if __N_CAND_LD__ % 32 != 0:
            wreck_yourself("__N_CAND_LD__ must be a multiple of 32. Change the value of __N_CAND_LD__ in kernels.cu")
        
    def fetch_constants_from_cuda(self, gpu_ctx, compiled):
        global __MAX_PERPLEXITY__, __Khd__, __Kld__, __N_CAND_LD__, __N_CAND_HD__, __N_INTERACTIONS_FAR__
        kernel_get_constants = CUDA_kernel("get_constants", compiled, gpu_ctx.make_cuShape(n_thds=1, n_32b_perThd=0, additional_n_32b=0))
        # Allocate memory on the GPU for the constants
        max_perplexity_gpu     = self.gpu_context.malloc(np.zeros(1, dtype=np.float32))
        khd_gpu                = self.gpu_context.malloc(np.zeros(1, dtype=np.uint32))
        kld_gpu                = self.gpu_context.malloc(np.zeros(1, dtype=np.uint32))
        n_cand_ld_gpu          = self.gpu_context.malloc(np.zeros(1, dtype=np.uint32))
        n_cand_hd_gpu          = self.gpu_context.malloc(np.zeros(1, dtype=np.uint32))
        n_interactions_far_gpu = self.gpu_context.malloc(np.zeros(1, dtype=np.uint32))
        # fetch data on GPU
        kernel_get_constants.blocking_launch(max_perplexity_gpu, khd_gpu, kld_gpu, n_cand_ld_gpu, n_cand_hd_gpu, n_interactions_far_gpu)
        # copy to cpu 
        max_perplexity = max_perplexity_gpu.get()[0]
        khd = khd_gpu.get()[0]
        kld = kld_gpu.get()[0]
        n_cand_ld = n_cand_ld_gpu.get()[0]
        n_cand_hd = n_cand_hd_gpu.get()[0]
        n_interactions_far = n_interactions_far_gpu.get()[0]
        # save to "constants" on cpu
        __MAX_PERPLEXITY__ = float(max_perplexity)
        __Khd__ = int(khd)
        __Kld__ = int(kld)
        __N_CAND_LD__ = int(n_cand_ld)
        __N_CAND_HD__ = int(n_cand_hd)
        __N_INTERACTIONS_FAR__ = int(n_interactions_far)
        # free the memory
        self.gpu_context.free(max_perplexity_gpu)
        self.gpu_context.free(khd_gpu)
        self.gpu_context.free(kld_gpu)
        self.gpu_context.free(n_cand_ld_gpu)
        self.gpu_context.free(n_cand_hd_gpu)
        self.gpu_context.free(n_interactions_far_gpu)

    def optimisation_preliminaries(self, optimisation_structures):
        read_set, write_set = optimisation_structures.get_readWrite_variables(is_phaseA=True)
        read_knn_HD,  read_sqdists_HD,  read_far_dist_HD,  read_Xld_true,  read_knn_LD,  read_sqdists_LD,  read_far_dist_LD  = read_set
        write_knn_HD, write_sqdists_HD, write_far_dist_HD, write_Xld_true, write_knn_LD, write_sqdists_LD, write_far_dist_LD = write_set
        # 1. called to init similarities in LD
        self.low_dim_updateSim_and_refineKNN(read_Xld_true,  read_knn_LD,  read_knn_HD,  write_knn_LD, write_sqdists_LD, write_far_dist_LD, optimisation_structures.neighbours_sumSnorms_LD, self.kern_alpha, self.streams.stream_neigh_HD)
        self.low_dim_updateSim_and_refineKNN(write_Xld_true, write_knn_LD, write_knn_HD, read_knn_LD,  read_sqdists_LD,  read_far_dist_LD,  optimisation_structures.neighbours_sumSnorms_LD, self.kern_alpha, self.streams.stream_neigh_HD)
        # 2. called to compute all distances in HD
        self.fill_all_sqdists_HD(optimisation_structures.cu_Xhd, read_knn_HD,  write_knn_HD, write_sqdists_HD, write_far_dist_HD, self.streams.stream_neigh_HD)
        self.fill_all_sqdists_HD(optimisation_structures.cu_Xhd, write_knn_HD, read_knn_HD,  read_sqdists_HD,  read_far_dist_HD, self.streams.stream_neigh_HD)
        # 3. will be used to flag new neighbours: radii in HD will be computed for all observations
        self.flag_all_points_as_having_new_neighbours(optimisation_structures, self.streams.stream_neigh_HD)
        self.streams.stream_neigh_HD.synchronize() 
        # 4. init the sum of LD similarities to 1 (will be used to divide somethings: can't risk it being 0 on the first iteration)
        optimisation_structures.randoms_sumSnorms_LD.resultArr_async[0] = 1.0
        optimisation_structures.randoms_sumSnorms_LD.resultArr_async[0] = 1.0
        # 5. configure the process launch mode
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError: # already set, can ignore this
            pass

    def receive_GUI_messages(self):
        HD_config_changed = False
        # fetch mutex protected variables from GUI
        with self.smem_kernel_alpha.get_lock():
            new_kern_alpha = np.float32(self.smem_kernel_alpha.value)
        with self.smem_attrac_mult.get_lock():
            new_attrac_mult = self.smem_attrac_mult.value
        with self.smem_perplexity.get_lock():
            new_perplexity = self.smem_perplexity.value
        with self.smem_dist_metric.get_lock():
            new_dist_metric = self.smem_dist_metric.value
        with self.smem_LR_shared.get_lock():
            new_lr_multiplier = self.smem_LR_shared.value
        with self.smem_explosion_please.get_lock():
            explosion_request = self.smem_explosion_please.value
            if explosion_request: 
                self.smem_explosion_please.value = False
        with self.smem_save_please.get_lock():
            save_request = self.smem_save_please.value
            if save_request:
                self.smem_save_please.value = False
        with self.smem_reset_please.get_lock():
            reset_request = self.smem_reset_please.value
            if reset_request:
                self.smem_reset_please.value = False
        # check if the HD  configuration has changed: if so, the dists in HD will need to be recomputed
        HD_config_changed = (new_perplexity != self.perplexity or new_dist_metric != self.dist_metric)
        # save the new values
        self.kern_alpha    = np.float32(new_kern_alpha)
        self.attrac_mult   = np.float32(new_attrac_mult)
        self.lr_multiplier = np.float32(new_lr_multiplier)
        self.perplexity    = np.float32(new_perplexity)
        self.dist_metric   = np.uint32(new_dist_metric)
        return HD_config_changed, explosion_request, save_request, reset_request
    
    def save_embedding(self, read_Xld, Y):
        cpu_Xld = read_Xld.get()
        file_suffix = "PP="+str(self.perplexity)+"_KA="+str(self.kern_alpha)+"_AM="+str(self.attrac_mult)+"_DM="+str(self.dist_metric)
        np.save("Xld_"+file_suffix, cpu_Xld)
        np.save("Y_"+file_suffix, Y)
        print("Saved Xld and Y to disk, in files: Xld_"+file_suffix, "  and  Y_"+file_suffix, " as numpy dumps  (readable with np.load())")

    def detemine_warmup_lengths(self, limit_by_time, limit_by_niter, max_n_sec, max_n_iter):
        if not limit_by_time and not limit_by_niter:
            warmup_length_iter = 1000
            warmup_length_sec = None
        elif limit_by_time and not limit_by_niter:
            warmup_length_iter = None
            warmup_length_sec = 0.3 * max_n_sec
        elif limit_by_niter:
            warmup_length_iter = int(0.7 * max_n_iter)
            warmup_length_sec  = None
        return warmup_length_iter, warmup_length_sec

    def is_warmup(self, with_warmup, warmup, warmup_length_iter, warmup_length_sec, iteration, elapsed, optimisation_structures):
        if not with_warmup:
            return False, 1.1
        warmup_ratio = 1.1
        if warmup:
            ratio_iter = (iteration / warmup_length_iter) if (warmup_length_iter is not None and warmup_length_iter>0)  else 1.0
            ratio_time = (elapsed / warmup_length_sec)    if (warmup_length_sec is not None and warmup_length_sec > 0.0001) else 1.0
            warmup_ratio = min(ratio_iter, ratio_time)
            if warmup_length_iter is not None and iteration >= warmup_length_iter:
                warmup = False
            if warmup_length_sec is not None and elapsed >= warmup_length_sec:
                warmup = False
            if not warmup:
                warmup_ratio = 1.1
                self.warmup_end(optimisation_structures)
        return warmup, warmup_ratio

    def is_running(self, running, limit_by_time, limit_by_niter, iteration, elapsed, max_n_sec, max_n_iter):
        if running:
            if limit_by_time:
                running = running and (elapsed < max_n_sec)
            if limit_by_niter:
                running = running and (iteration < max_n_iter)
        return running

    # not efficient, but barely ever called
    def reset_embedding(self, cpu_Xhd_preprocessed, read_Xld, write_Xld, cuda_Xld_mmtm, stream):
        cpu_Xld_mmtm = cuda_Xld_mmtm.get()
        cpu_Xld_mmtm.fill(0.0)
        cuda_Xld_mmtm.set_async(cpu_Xld_mmtm, stream=stream)
        cpu_Xld  = np.dot(cpu_Xhd_preprocessed, np.random.randn(self.Mhd, self.Mld).astype(np.float32)).astype(np.float32)
        std_now  = np.std(cpu_Xld)
        cpu_Xld *= 1e-4 / (std_now + 1e-22)
        write_Xld.set_async(cpu_Xld, stream=stream)
        read_Xld.set_async(cpu_Xld, stream=stream)
        stream.synchronize()

    # not efficient, but barely ever called
    def implosion(self, read_Xld, write_Xld, cuda_Xld_mmtm, stream):
        factor = 1.0 / 100.0
        cpu_Xld = read_Xld.get()
        cpu_Xld = cpu_Xld * factor
        write_Xld.set_async(cpu_Xld, stream=stream)
        read_Xld.set_async(cpu_Xld, stream=stream)
        momentum_cpu = cuda_Xld_mmtm.get()
        momentum_cpu = momentum_cpu * 0.0
        cuda_Xld_mmtm.set_async(momentum_cpu, stream=stream)
        stream.synchronize()

    def transform(self):
        if not self.is_fitted:
            raise Exception("\033[38;2;255;0;0mERROR: " + "fastSNE: transform() called before fit(), or .fit() crashed silently (which would be worrisome)" + "\033[0m")
        return self.Xld

    def flag_all_points_as_having_new_neighbours(self, optimisation_structures, stream):
        self.kernels.flag_new_neighbours_for_all.async_launch(stream, self.N, optimisation_structures.cu_has_new_HD_neighs, optimisation_structures.cu_has_new_HD_neighs_acc)

    def low_dim_updateSim_and_refineKNN(self, Xld_read, knn_LD_read, knn_HD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write,  neighbours_sumSnorms_LD, kern_alpha, stream):
        # 1.  squared dists to LD neighbours, sort neighbours, find farthest dists
        #     compute similarity nominators and first reduction on them for each i
        seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        self.kernels.compute_all_LD_sqdists.async_launch(stream,\
            self.N, self.Mld, Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, seed)
        # 2. candidate neighbours: generate, compute dists, and partial sort
        seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        self.kernels.candidates_LD_generate_and_sort.async_launch(stream,\
            self.N, self.Mld, Xld_read, knn_LD_read, knn_LD_write, sqdists_LD_write, farthest_dist_LD_write, knn_HD_read, seed) 

    def fill_all_sqdists_HD(self, Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, stream):
        kernel = None
        if self.dist_metric == 0:
            kernel = self.kernels.all_HD_sqdists_euclidean
        elif self.dist_metric == 1:
            kernel = self.kernels.all_HD_sqdists_manhattan
        elif self.dist_metric == 2:
            kernel = self.kernels.all_HD_sqdists_cosine
        else:
            kernel = self.kernels.all_HD_sqdists_custom
        seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        kernel.async_launch(stream, self.N, self.Mhd, Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, seed)

    def should_we_refine_HD_neighbourhoods_this_iteration(self, force_KNNsearch, iteration, ema_pct_new_HD_neighs):
        do_HDnnDescent = force_KNNsearch or (iteration < 100)
        if not do_HDnnDescent:
            do_HDnnDescent = np.random.rand() < 0.02 + 2.0 * ema_pct_new_HD_neighs
        return do_HDnnDescent

    def warmup_tweaks(self, optimisation_structures, warmup_ratio, iteration):
        if warmup_ratio > 0.999:
            return
        self.smem_force_new_vals.value = True
        

        phase2_start = 0.75

        phase1_PP = __MAX_PERPLEXITY__

        phase1_LRmul = 100.0
        end_LRmul    = 10.0

        phase1_kernel_alpha = 70.0
        end_kernel_alpha    = self.end_kernel_alpha

        phase1_attrac_mult  = 0.9
        end_attrac_mult     = self.end_attrac_mult

        if warmup_ratio < phase2_start:
            if (iteration % 15) < 5 and warmup_ratio < 0.35*phase2_start:
                self.smem_reset_please.value = True
            self.smem_LR_shared.value     = phase1_LRmul
            self.smem_kernel_alpha.value  = phase1_kernel_alpha
            self.smem_attrac_mult.value   = phase1_attrac_mult
            self.smem_perplexity.value    = phase1_PP
        else:
            ratio = (warmup_ratio - phase2_start) / (1.0 - phase2_start)
            self.smem_LR_shared.value    = phase1_LRmul + (end_LRmul - phase1_LRmul) * ratio
            self.smem_kernel_alpha.value = phase1_kernel_alpha + (end_kernel_alpha - phase1_kernel_alpha) * ratio
            self.smem_attrac_mult.value  = phase1_attrac_mult + (end_attrac_mult - phase1_attrac_mult) * ratio

            if (iteration % 25) == 0:
                mmtm = optimisation_structures.cu_Xld_mmtm.get()
                mmtm *= 2.0 
                optimisation_structures.cu_Xld_mmtm.set(mmtm)
           
    def warmup_end(self, optimisation_structures):
        with self.smem_attrac_mult.get_lock():
            self.smem_attrac_mult.value = self.end_attrac_mult
        with self.smem_kernel_alpha.get_lock():
            self.smem_kernel_alpha.value = self.end_kernel_alpha
        with self.smem_perplexity.get_lock():
            self.smem_perplexity.value = self.end_PP
        with self.smem_force_new_vals.get_lock():
            self.smem_force_new_vals.value = True
        with self.smem_explosion_please.get_lock():
            self.smem_explosion_please.value = True
        
        cpu_momenta = optimisation_structures.cu_Xld_mmtm.get()
        cpu_momenta.fill(0.0)
        optimisation_structures.cu_Xld_mmtm.set(cpu_momenta)

    def launch_gui(self, optimisation_structures, Y, dont_launch=False):
        if self.verbose and not dont_launch:
            print("\033[38;2;255;165;0m \nLaunching the GUI process... (windows warning might trigger, but that's fine, it's just being whiny) \033[0m", end='')

        from .gui import gui_worker
        # 1. Initialise shared memory with GUI (on CPU)
        cpu_shared_mem      = shared_memory.SharedMemory(create=True, size=int(self.N * self.Mld * np.dtype(np.float32).itemsize))
        cpu_Xld_arr_on_smem = np.ndarray((self.N, self.Mld), dtype=np.float32, buffer=cpu_shared_mem.buf)
        optimisation_structures.cu_Xld_true_A.get(cpu_Xld_arr_on_smem) # copy (GPU->CPU) cuda_Xld_true_A to shared memory
        cuda_Xld_temp_Xld   = self.gpu_context.malloc(np.zeros((self.N, self.Mld), dtype=np.float32)) # temp structure related to preprocessing of the embedding for the GUI
        # 2. Shared dynamical hyperparameters
        self.smem_kernel_alpha   = multiprocessing.Value('f', self.kern_alpha)
        self.smem_perplexity     = multiprocessing.Value('f', self.perplexity)
        self.smem_attrac_mult    = multiprocessing.Value('f', self.attrac_mult)
        self.smem_dist_metric    = multiprocessing.Value('i', self.dist_metric)
        self.smem_LR_shared      = multiprocessing.Value('f', self.lr)
        # 3. Shared communications with GUI
        self.smem_gui_closed                 = multiprocessing.Value('b', False)
        self.smem_points_ready_for_rendering = multiprocessing.Value('b', False)
        self.smem_points_rendering_finished  = multiprocessing.Value('b', False)
        self.smem_iteration                  = multiprocessing.Value('i', 0)
        self.smem_explosion_please           = multiprocessing.Value('b', False) 
        self.smem_save_please                = multiprocessing.Value('b', False)
        self.smem_reset_please               = multiprocessing.Value('b', False)
        self.smem_force_new_vals             = multiprocessing.Value('b', False)
        # 4. Launch the GUI process
        process_gui = None
        if not dont_launch:
            process_gui = multiprocessing.Process(target=gui_worker, args=(cpu_shared_mem, Y, self.N, self.Mld, self.smem_kernel_alpha, self.smem_perplexity, self.smem_attrac_mult, self.smem_LR_shared, self.smem_dist_metric, self.smem_gui_closed, self.smem_points_ready_for_rendering, self.smem_points_rendering_finished, self.smem_iteration, self.smem_explosion_please, self.smem_save_please, self.smem_reset_please, self.smem_force_new_vals, __MIN_PERPLEXITY__, __MAX_PERPLEXITY__, __MIN_KERNEL_ALPHA__, __MAX_KERNEL_ALPHA__, __MIN_ATTRACTION_MULTIPLIER__, __MAX_ATTRACTION_MULTIPLIER__))
            process_gui.start()
            # 5. waith for gui init to be finished (points_rendering_finished set to True)
            gui_innit_done = False 
            while not gui_innit_done:
                with self.smem_points_rendering_finished.get_lock():
                    gui_innit_done = self.smem_points_rendering_finished.value
                time.sleep(0.01)
        if self.verbose and not dont_launch:
            print("\033[32m   DONE\033[0m")
        return cuda_Xld_temp_Xld, process_gui, cpu_shared_mem, cpu_Xld_arr_on_smem
    
    def send_to_GUI_pipline_step(self, iteration, optimisation_structures, read_Xld, cuda_Xld_temp_Xld, cpu_Xld_arr_on_smem, gui_data_prep_phase):
        def notify_GUI_that_data_is_ready():
            with self.smem_points_ready_for_rendering.get_lock():
                self.smem_points_ready_for_rendering.value = True
                self.last_frame_time = time.time()
                self.frame_counter += 1
        
        def is_GUI_done_rendering():
            gui_done = False
            with self.smem_points_rendering_finished.get_lock():
                gui_done = self.smem_points_rendering_finished.value
            gui_done = True # meh
            if gui_done:
                current_time = time.time()
                frame_time = current_time - self.last_frame_time
                alpha = 0.2
                self.render_time_ema = (1-alpha) * self.render_time_ema + alpha * frame_time
            return gui_done
        
        # iteration sent to the GUI
        with self.smem_iteration.get_lock():
            self.smem_iteration.value = iteration

        stream_for_scaling = self.streams.stream_minMax2
        if gui_data_prep_phase == 0:
            self.scaling_of_embedding_for_rendering(optimisation_structures, read_Xld, cuda_Xld_temp_Xld, stream_for_scaling, read_Xld)
            return 2
        #      - case 2 :  copy the data to shared memory, and notify the GUI that the data is ready
        elif gui_data_prep_phase == 2:
            # dont do anything while rendering is being performed
            if not is_GUI_done_rendering():
                return gui_data_prep_phase
            # send the scaled embedding to the gui
            cuda_Xld_temp_Xld.get_async(stream=stream_for_scaling, ary=cpu_Xld_arr_on_smem)
            stream_for_scaling.synchronize()
            notify_GUI_that_data_is_ready()
            return 0
        return gui_data_prep_phase

    def scaling_of_embedding_for_rendering(self, optimisation_structures, Xld_read, Xld_scaled, stream_for_scaling, read_Xld):
        # get the min and max, which were launched in the previous iteration
        optimisation_structures.gui_Xld_minFinder.async_reduce_this(gpu_array_to_reduce = read_Xld, stream=self.streams.stream_minMax1)
        optimisation_structures.gui_Xld_maxFinder.async_reduce_this(gpu_array_to_reduce = read_Xld, stream=self.streams.stream_minMax2)
        self.streams.stream_minMax1.synchronize()
        self.streams.stream_minMax2.synchronize()
        global_min = (optimisation_structures.gui_Xld_minFinder.get())
        global_max = (optimisation_structures.gui_Xld_maxFinder.get())
        # the scaling of X proper
        self.kernels.scale_Xld.async_launch(stream_for_scaling,\
            Xld_read, Xld_scaled, np.float32(global_min), np.float32(global_max), np.uint32(self.N), np.uint32(self.Mld))

    def high_dim_refineKNN(self, dist_type, Xhd, knn_HD_read, knn_HD_write, knn_LD_read, sqdists_HD_write, farthest_dist_HD_write, HD_n_new_neighs_sum, has_new_HD_neighs, has_new_HD_neighs_acc, stream):
        # 1.  candidate neighbours: generate, compute dists, and partial sort
        global_seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        self.kernels.candidates_HD_generate.async_launch(stream,\
                    dist_type, self.N, self.Mhd, has_new_HD_neighs, has_new_HD_neighs_acc, Xhd, knn_HD_read, knn_HD_write, sqdists_HD_write, farthest_dist_HD_write, knn_LD_read, global_seed)
        # 2. compute the sum of the obs that have new neighbours
        HD_n_new_neighs_sum.async_reduce_this(gpu_array_to_reduce = has_new_HD_neighs, stream=stream)
        # 3. recompute farthest dists on the obs that have new neighbours
        global_seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        self.kernels.recompute_HD_farthest_distances.async_launch(stream,\
                    self.N, self.Mhd, global_seed, has_new_HD_neighs,  knn_HD_write, sqdists_HD_write, farthest_dist_HD_write)

    def perhaps_recompute_P_matrix(self, read_set, write_set, optimisation_structures, force_recompute, niter_since_recompute_P, HD_config_changed, EMA_pct_new_HD_neighs, bias = 0.02):
        recompute_it = HD_config_changed or force_recompute
        if not recompute_it and niter_since_recompute_P > 25:
            recompute_it = r() < bias + ((0.2 * EMA_pct_new_HD_neighs)**2)
        if recompute_it:
            write_knn_HD, write_sqdists_HD, _, _, _, _, _ = write_set
            read_knn_HD,  read_sqdists_HD,  read_far_dist_HD,  read_Xld_true,  read_knn_LD,  read_sqdists_LD,  read_far_dist_LD  = read_set
            niter_since_recompute_P = -1
            seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
            optimisation_structures.cu_Pasm_sums.fill(0.0)
            self.kernels.radii_P_part1.async_launch(self.streams.stream_neigh_HD,\
                    self.N, self.perplexity, optimisation_structures.cu_has_new_HD_neighs_acc, write_sqdists_HD,\
                    optimisation_structures.cu_invRadii_HD, optimisation_structures.cu_Pasm, optimisation_structures.cu_Pasm_sums, read_knn_HD, seed, read_far_dist_HD)
            self.kernels.radii_P_part2.async_launch(self.streams.stream_neigh_HD,\
                    self.N, optimisation_structures.cu_has_new_HD_neighs_acc, write_knn_HD, optimisation_structures.cu_Psym_knn,\
                    write_sqdists_HD, optimisation_structures.cu_invRadii_HD, optimisation_structures.cu_Psym, optimisation_structures.cu_Pasm, optimisation_structures.cu_Pasm_sums)
            self.streams.stream_neigh_HD.synchronize()
        return niter_since_recompute_P + 1
    
    def compute_LD_simi_denominator(self, optimisation_structures):
        """ random_sum = optimisation_structures.randoms_sumSnorms_LD.get()
        neighs_sum = optimisation_structures.neighbours_sumSnorms_LD.get()
        n_samples_estim = np.float32(self.N) * np.float32(__Khd__ + __Khd__ + __N_INTERACTIONS_FAR__)
        matrix_area     = np.float32(self.N) * np.float32(self.N - 1) / 2.0
        scaling_factor  = matrix_area / n_samples_estim
        return np.float32(scaling_factor * (random_sum + neighs_sum)) """
        random_sum = optimisation_structures.randoms_sumSnorms_LD.get()
        n_samples_estim = np.float32(self.N) * np.float32(__N_INTERACTIONS_FAR__)
        matrix_area     = np.float32(self.N) * np.float32(self.N - 1)
        scaling_factor  = matrix_area / n_samples_estim
        return np.float32(scaling_factor * random_sum)

    def optimise_with_gui(self, cpu_Xhd_preprocessed, optimisation_structures, limit_by_time, limit_by_niter, max_n_sec, max_n_iter, Y, with_warmup):
        # 1. Launch the GUI process
        cuda_Xld_temp_Xld, process_gui, cpu_shared_mem, cpu_Xld_arr_on_smem = self.launch_gui(optimisation_structures, Y)

        if self.end_attrac_mult is not None:
            self.smem_attrac_mult.value  = self.end_attrac_mult
        if self.end_kernel_alpha is not None:
            self.smem_kernel_alpha.value = self.end_kernel_alpha
        if self.end_PP is not None:
            self.smem_perplexity.value   = self.end_PP
        self.smem_force_new_vals.value = True

        # 2. init some local variables
        denominator_simi_LD     = np.float32(self.N * __Kld__ * 0.2)
        running                 = True
        warmup                  = True
        warmup_ratio            = 0.0
        iteration               = 0
        isPhaseA                = True
        gui_data_prep_phase     = 0 # determines the current state in the pipeline giving the current embedding state to the GUI
        busy_copying__for_GUI   = False
        start_t                 = time.time()
        niter_since_recompute_P = 0
        EMA_pct_new_HD_neighs   = 1.0  # pct of points that discovered new neighbours, smoothed across time
        prev_iter_did_HD_knn_search    = True
        prev_iter_had_HD_config_change = True
        # 3. Determine warmup lengths based on limits
        if not self.user_set_warmup_n_iter:
            warmup_length_iter, warmup_length_sec = self.detemine_warmup_lengths(limit_by_time, limit_by_niter, max_n_sec, max_n_iter)
        else:
            warmup_length_iter, warmup_length_sec = self.user_set_warmup_n_iter, None
        # warmup_length_iter, warmup_length_sec = 10, 0 # DEVELOPMENT ONLY: REMOVE THIS AND UNCOMMENT THE LINE ABOVE
        
        # 4. finally, optimise
        while running:
            # 1. Fetch the read/write structures for this phase
            read_set, write_set = optimisation_structures.get_readWrite_variables(is_phaseA=isPhaseA)
            read_knn_HD,  read_sqdists_HD,  read_far_dist_HD,  read_Xld_true,  read_knn_LD,  read_sqdists_LD,  read_far_dist_LD  = read_set
            write_knn_HD, write_sqdists_HD, write_far_dist_HD, write_Xld_true, write_knn_LD, write_sqdists_LD, write_far_dist_LD = write_set
            cu_Xld_mmtm = optimisation_structures.cu_Xld_mmtm
            
            # 2. Update hyperparameters / action request from the GUI
            HD_config_changed, explosion_request, save_request, reset_request = self.receive_GUI_messages()
            
            # 3. Sync all streams (sync in time at each iteration = costly but necessary constraint)
            self.streams.sync_all() # streams: neigh_HD, neigh_LD, minMax, grads
            
            # 4. Save embedding / explosion / reset
            if save_request:
                self.save_embedding(read_Xld_true, Y)
            if reset_request:
                self.reset_embedding(cpu_Xhd_preprocessed, read_Xld_true, write_Xld_true, cu_Xld_mmtm, self.streams.stream_grads)
            if explosion_request: # well technically it's an implosion
                self.implosion(read_Xld_true, write_Xld_true, cu_Xld_mmtm, self.streams.stream_grads)
            
            # 5. Possibly recompute the sparse P matrix (in HD). Do it if HD config changed, or if the gods of randomness will it. The probability increases with EMA_pct_new_HD_neighs with a positive bias of 0.02
            force_recompute_P       = (iteration < 400) and ((iteration % 25) == 0)
            niter_since_recompute_P = self.perhaps_recompute_P_matrix(read_set, write_set, optimisation_structures, force_recompute_P, niter_since_recompute_P, HD_config_changed, EMA_pct_new_HD_neighs, bias = 0.02)
            
            # 6. Warmup particularities
            self.warmup_tweaks(optimisation_structures, warmup_ratio, iteration)

            # 7. Re-sync the HD knn at each iteration, else some HD discovery works would be lost. ("write" now were old "read" and vice versa)
            if prev_iter_did_HD_knn_search or prev_iter_had_HD_config_change:
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_knn_HD,      src_gpuarray=read_knn_HD,      stream=self.streams.generic_stream1)
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_sqdists_HD,  src_gpuarray=read_sqdists_HD,  stream=self.streams.generic_stream2)
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_far_dist_HD, src_gpuarray=read_far_dist_HD, stream=self.streams.generic_stream3)
            self.streams.generic_stream1.synchronize(); self.streams.generic_stream2.synchronize(); self.streams.generic_stream3.synchronize()

            # 8. Recompute all neigh dists on HD hparam change
            if (HD_config_changed) or ((iteration%990) == 0):
                self.fill_all_sqdists_HD(optimisation_structures.cu_Xhd, read_knn_HD,  write_knn_HD, write_sqdists_HD, write_far_dist_HD, self.streams.stream_neigh_HD)
                self.fill_all_sqdists_HD(optimisation_structures.cu_Xhd, write_knn_HD, read_knn_HD,  read_sqdists_HD,  read_far_dist_HD, self.streams.stream_neigh_HD)
                self.flag_all_points_as_having_new_neighbours(optimisation_structures, self.streams.stream_neigh_HD)
                self.streams.stream_neigh_HD.synchronize()

            # 9. Get the sums of for LD simi denominator
            denominator_simi_LD = self.compute_LD_simi_denominator(optimisation_structures)

            # 10. Determine if we should refine the HD neighbourhoods this iteration
            EMA_pct_new_HD_neighs = 0.9 * EMA_pct_new_HD_neighs + 0.1 * float(optimisation_structures.HD_n_new_neighs_sum.get()) / float(self.N)
            do_HDnnDescent = self.should_we_refine_HD_neighbourhoods_this_iteration(self.purpose_is_KNN, iteration, EMA_pct_new_HD_neighs)
            if self.verbose:
                print(f"\r{np.round(EMA_pct_new_HD_neighs, 2)}  ", end=' ')

            # 11. One iteration proper
            self.one_iteration(iteration, warmup, warmup_ratio, do_HDnnDescent, cpu_Xhd_preprocessed, write_set, read_set, optimisation_structures, denominator_simi_LD)

            # 12. sending the embedding to the GUI
            gui_data_prep_phase = self.send_to_GUI_pipline_step(iteration, optimisation_structures, read_Xld_true, cuda_Xld_temp_Xld, cpu_Xld_arr_on_smem, gui_data_prep_phase)

            # 13. end of iteration: prepare next one and check if we're done
            isPhaseA   = not isPhaseA
            iteration += 1
            elapsed    = time.time() - start_t
            prev_iter_did_HD_knn_search    = do_HDnnDescent
            prev_iter_had_HD_config_change = HD_config_changed
            # 13.1 Update running condition
            with self.smem_gui_closed.get_lock():
                running = self.is_running(running, limit_by_time, limit_by_niter, iteration, elapsed, max_n_sec, max_n_iter) and not self.smem_gui_closed.value
            # 13.2 Manage warmup
            warmup, warmup_ratio = self.is_warmup(with_warmup, warmup, warmup_length_iter, warmup_length_sec, iteration, elapsed, optimisation_structures)

        # 4. notify the GUI that we're done (it would still be running if termination was due to time or iteration limit)
        with self.smem_gui_closed.get_lock():
            self.smem_gui_closed.value = True
        process_gui.join()
        
        if self.verbose:
            print("\033[38;2;255;165;0m \nfastSNE: optimisation finished. in ", iteration, " iterations and ", np.round(elapsed, 2), " seconds. \033[0m")

    def optimise_soberly(self, cpu_Xhd_preprocessed, optimisation_structures, limit_by_time, limit_by_niter, max_n_sec, max_n_iter, Y, with_warmup):
        # 0. Fake launch of the GUI process (sets an,d allocates variables)
        cuda_Xld_temp_Xld, process_gui, cpu_shared_mem, cpu_Xld_arr_on_smem = self.launch_gui(optimisation_structures, Y, dont_launch=True)
        
        if self.end_attrac_mult is not None:
            self.smem_attrac_mult.value  = self.end_attrac_mult
        if self.end_kernel_alpha is not None:
            self.smem_kernel_alpha.value = self.end_kernel_alpha
        if self.end_PP is not None:
            self.smem_perplexity.value   = self.end_PP
        self.smem_force_new_vals.value = True

        # 1. init some local variables
        denominator_simi_LD     = np.float32(self.N * __Kld__ * 0.2)
        running                 = True
        warmup                  = True
        warmup_ratio            = 0.0
        iteration               = 0
        isPhaseA                = True
        gui_data_prep_phase     = 0 # determines the current state in the pipeline giving the current embedding state to the GUI
        start_t                 = time.time()
        niter_since_recompute_P = 0
        EMA_pct_new_HD_neighs   = 1.0  # pct of points that discovered new neighbours, smoothed across time
        prev_iter_did_HD_knn_search    = True
        prev_iter_had_HD_config_change = True
        # 3. Determine warmup lengths based on limits
        warmup_length_iter, warmup_length_sec = self.detemine_warmup_lengths(limit_by_time, limit_by_niter, max_n_sec, max_n_iter)
        
        if warmup_length_iter > 300: 
            warmup_length_iter = 300


        

        # 4. finally, optimise
        while running:
            # 1. Fetch the read/write structures for this phase
            read_set, write_set = optimisation_structures.get_readWrite_variables(is_phaseA=isPhaseA)
            read_knn_HD,  read_sqdists_HD,  read_far_dist_HD,  read_Xld_true,  read_knn_LD,  read_sqdists_LD,  read_far_dist_LD  = read_set
            write_knn_HD, write_sqdists_HD, write_far_dist_HD, write_Xld_true, write_knn_LD, write_sqdists_LD, write_far_dist_LD = write_set
            cu_Xld_mmtm = optimisation_structures.cu_Xld_mmtm
            
            # 3. Sync all streams (sync in time at each iteration = costly but necessary constraint)
            self.streams.sync_all() # streams: neigh_HD, neigh_LD, minMax, grads
            
            # 5. Possibly recompute the sparse P matrix (in HD). Do it if HD config changed, or if the gods of randomness will it. The probability increases with EMA_pct_new_HD_neighs with a positive bias of 0.02
            force_recompute_P       = False
            niter_since_recompute_P = self.perhaps_recompute_P_matrix(read_set, write_set, optimisation_structures, force_recompute_P, niter_since_recompute_P, HD_config_changed, EMA_pct_new_HD_neighs, bias = 0.001)
            
            # 6. Warmup particularities
            self.warmup_tweaks(optimisation_structures, warmup_ratio, iteration)

            # 7. Re-sync the HD knn at each iteration, else some HD discovery works would be lost. ("write" now were old "read" and vice versa)
            if prev_iter_did_HD_knn_search:
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_knn_HD,      src_gpuarray=read_knn_HD,      stream=self.streams.generic_stream1)
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_sqdists_HD,  src_gpuarray=read_sqdists_HD,  stream=self.streams.generic_stream2)
                self.gpu_context.copy_gpu2gpu_async(dest_gpuarray=write_far_dist_HD, src_gpuarray=read_far_dist_HD, stream=self.streams.generic_stream3)
            self.streams.generic_stream1.synchronize(); self.streams.generic_stream2.synchronize(); self.streams.generic_stream3.synchronize()

            # 9. Get the sums of for LD simi denominator
            denominator_simi_LD = self.compute_LD_simi_denominator(optimisation_structures)

            # 10. Determine if we should refine the HD neighbourhoods this iteration
            EMA_pct_new_HD_neighs = 0.9 * EMA_pct_new_HD_neighs + 0.1 * float(optimisation_structures.HD_n_new_neighs_sum.get()) / float(self.N)
            do_HDnnDescent = self.should_we_refine_HD_neighbourhoods_this_iteration(self.purpose_is_KNN, iteration, EMA_pct_new_HD_neighs)

            # 11. One iteration proper
            self.one_iteration(iteration, warmup, warmup_ratio, do_HDnnDescent, cpu_Xhd_preprocessed, write_set, read_set, optimisation_structures, denominator_simi_LD)

            # 13. end of iteration: prepare next one and check if we're done
            isPhaseA   = not isPhaseA
            iteration += 1
            elapsed    = time.time() - start_t
            prev_iter_did_HD_knn_search    = do_HDnnDescent
            # 13.1 Update running condition
            with self.smem_gui_closed.get_lock():
                running = self.is_running(running, limit_by_time, limit_by_niter, iteration, elapsed, max_n_sec, max_n_iter)
            # 13.2 Manage warmup
            warmup, warmup_ratio = self.is_warmup(with_warmup, warmup, warmup_length_iter, warmup_length_sec, iteration, elapsed, optimisation_structures)

        if self.verbose:
            print("\033[38;2;255;165;0m \nfastSNE: optimisation finished. in ", iteration, " iterations and ", np.round(elapsed, 2), " seconds. \033[0m")

    def one_iteration(self, iteration, warmup, warmup_ratio, do_HDnnDescent, cpu_Xhd_preprocessed, write_set, read_set, optimisation_structures, denominator_simi_LD):
        read_knn_HD,  read_sqdists_HD,  read_far_dist_HD,  read_Xld_true,  read_knn_LD,  read_sqdists_LD,  read_far_dist_LD  = read_set
        write_knn_HD, write_sqdists_HD, write_far_dist_HD, write_Xld_true, write_knn_LD, write_sqdists_LD, write_far_dist_LD = write_set
        cu_Xld_mmtm, cu_Xld_nest = optimisation_structures.cu_Xld_mmtm, optimisation_structures.cu_Xld_nest

        lr = self.lr * self.lr_multiplier

        # 1. refine the neighbourhoods in LD, and update their similarities
        self.low_dim_updateSim_and_refineKNN(read_Xld_true,  read_knn_LD,  read_knn_HD,  write_knn_LD, write_sqdists_LD, write_far_dist_LD, optimisation_structures.neighbours_sumSnorms_LD, self.kern_alpha,\
                                            self.streams.stream_neigh_LD)
        if do_HDnnDescent:
            self.high_dim_refineKNN(self.dist_metric, optimisation_structures.cu_Xhd, read_knn_HD, write_knn_HD, read_knn_LD, \
                                    write_sqdists_HD, write_far_dist_HD, optimisation_structures.HD_n_new_neighs_sum, optimisation_structures.cu_has_new_HD_neighs, optimisation_structures.cu_has_new_HD_neighs_acc,\
                                    self.streams.stream_neigh_HD)

        # 2. ready the Nesterov parameters
        self.kernels.make_Xnesterov.async_launch(self.streams.stream_grads, \
                    self.N, self.Mld, optimisation_structures.cu_grad_acc_global, write_Xld_true, cu_Xld_nest, cu_Xld_mmtm, lr)
        # 3. gradients to gradient accs
        global_seed = np.uint32(np.random.randint(low = 1, high = (__MAX_UINT32_T__//2)))
        repulsion_multiplier = np.float32(1.0 - self.attrac_mult) * self.repulsion_base
        self.kernels.compute_gradients.async_launch(self.streams.stream_grads, \
                    np.float32(1.0), np.uint32(1), np.float32(1e-12), self.N, self.Mhd, self.Mld, self.kern_alpha, global_seed,\
                    optimisation_structures.cu_grad_acc_global, optimisation_structures.randoms_sumSnorms_LD.lvl1_, optimisation_structures.neighbours_sumSnorms_LD.lvl1_,\
                    cu_Xld_nest, optimisation_structures.cu_Psym_knn, optimisation_structures.cu_Psym, read_knn_LD, repulsion_multiplier, denominator_simi_LD,\
                    read_sqdists_HD, read_sqdists_LD, read_far_dist_HD, read_far_dist_LD)
        #float* neighdists_HD, float* neighdists_LD, float* maxdist_HD, float* maxdist_LD
        
        # 4 gradient accs to momenta, momenta to parameters
        self.kernels.receive_gradients.async_launch(self.streams.stream_grads, \
                    self.N, self.Mld, optimisation_structures.cu_grad_acc_global, write_Xld_true, cu_Xld_mmtm, lr)
        # 5. sum of norms of randoms and neighbours
        optimisation_structures.randoms_sumSnorms_LD.async_reduce(stream=self.streams.stream_grads)
        optimisation_structures.neighbours_sumSnorms_LD.async_reduce(stream=self.streams.stream_grads)
            


