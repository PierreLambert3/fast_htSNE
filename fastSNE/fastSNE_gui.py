# import the classics
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import modernGL (renderer) and pyglet (windowerer)
import moderngl as mgl
import pyglet

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

__TARGET_FPS__ = 20.0

def gen_K_random_colours(K):
    n = K * 200
    dataset = np.random.uniform(0, 1, (n, 3))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, n_init=1, max_iter=25)
    kmeans.fit(dataset)
    return (kmeans.cluster_centers_).astype(np.float32)

def determine_Y_colour(Y):
    cpu_Y_colours = np.zeros((len(Y), 3), dtype=np.float32)
    is_classification  = type(Y[0, 0]) == np.int32
    if is_classification:
        label_colours = gen_K_random_colours(len(np.unique(Y)))
        for i in range(len(Y)):
            cpu_Y_colours[i, :] = label_colours[Y[i, 0]]
    else:
        min_Y = np.min(Y)
        max_Y = np.max(Y)
        span = max_Y - min_Y
        if span < 0.0000001:
            span = 0.0000001
        red0   = 170.0
        green0 = 0.0
        blue0  = 210.0
        red1   = 150.0
        green1 = 210.0
        blue1  = 0.0
        for i in range(len(Y)):
            y_normed = (Y[i, 0] - min_Y) / span
            cpu_Y_colours[i, 0] = np.float32(y_normed*red1 + (1.0-y_normed)*red0)  / 255.0
            cpu_Y_colours[i, 1] = np.float32(y_normed*green1 + (1.0-y_normed)*green0)  / 255.0
            cpu_Y_colours[i, 2] = np.float32(y_normed*blue1 + (1.0-y_normed)*blue0)  / 255.0
    cuda_Y_colours = gpuarray.to_gpu(cpu_Y_colours)
    return cuda_Y_colours


    
class ModernGLWindow(pyglet.window.Window):
    def __init__(self, smem_cpu_Xld_A, smem_cpu_Xld_B, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated, **kwargs):
        super().__init__(**kwargs)
        self.N   = N
        self.Mld = Mld
        # hyperparameters
        self.kernel_alpha = kernel_alpha # multiprocessing Value type
        self.perplexity   = perplexity   # multiprocessing Value type
        self.dist_metric  = dist_metric  # multiprocessing Value type
        # communication with the main process
        self.gui_closed        = gui_closed        # multiprocessing Value type
        self.isPhaseA          = isPhaseA          # multiprocessing Value type
        self.points_updated    = points_updated    # multiprocessing Value type
        # shared memory for the data
        self.cpu_Xld_A = np.ndarray((N, Mld), dtype=np.float32, buffer=smem_cpu_Xld_A.buf)
        self.cpu_Xld_B = np.ndarray((N, Mld), dtype=np.float32, buffer=smem_cpu_Xld_B.buf)

        print("self.cpu_Xld_A:   qsdsqsd ", self.cpu_Xld_A)
        1/0

        # variables on the GPU
        self.cuda_Y_colours   = determine_Y_colour(cpu_Y) # size (N, 3), colour of each observation (removes an "if" in the render function)

        # working variables: these are the ones that will be updated and rendered
        self.cuda_Xld_px = gpuarray.to_gpu(np.zeros((N, self.Mld), dtype=np.float32))
        
        # prepare the data for the first frame
        self.retrieve_and_prepare_data()

        self.ctx  = mgl.create_context()
        # Update VBO for vertex positions
        self.vbo = self.ctx.buffer(self.cuda_Xld_px.astype('f4').tobytes())
        # Create CBO for colors
        self.cbo = self.ctx.buffer(self.cuda_Y_colours.astype('f4').tobytes())
        # Update VAO to include color data
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', self.cbo, 'in_color')
        # Modify shaders to include color handling
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec3 in_color;
                out vec3 color;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 color;
                out vec4 f_color;
                void main() {
                    f_color = vec4(color, 1.0);
                }
            ''',
        )

        
        # # modernGL structures
        # self.ctx  = mgl.create_context()
        # self.prog = self.ctx.program(
        #     vertex_shader='''
        #         #version 330
        #         in vec2 in_vert;
        #         void main() {
        #             gl_Position = vec4(in_vert, 0.0, 1.0);
        #         }
        #     ''',
        #     fragment_shader='''
        #         #version 330
        #         out vec4 f_color;
        #         void main() {
        #             f_color = vec4(1.0, 1.0, 1.0, 1.0);
        #         }
        #     ''',
        # )
        # self.vbo = self.ctx.buffer(reserve=N * 2 * 4)
        # self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def on_draw(self):
        self.clear()
        self.vao.render(mgl.POINTS)
    
    def on_close(self):
        self.vbo.release()
        self.vao.release()
        self.ctx.release()
        super().on_close()
    
    def update(self, dt):
        point_have_moved = False
        with self.points_have_moved.get_lock():
            point_have_moved = self.points_have_moved.value
            self.points_have_moved.value = False
        if point_have_moved:
            print("Update the data in the VBO every frame")
        return

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            print("Left mouse button pressed")
        elif button == pyglet.window.mouse.RIGHT:
            print("Right mouse button pressed")
    
    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            print("Left mouse button released")
        elif button == pyglet.window.mouse.RIGHT:
            print("Right mouse button released")
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            print("Left mouse button dragged")
        elif buttons & pyglet.window.mouse.RIGHT:
            print("Right mouse button dragged")
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            print("Increase perplexity")
        elif scroll_y < 0:
            print("Decrease perplexity")

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.close()

    def retrieve_and_prepare_data(self):
        import ctypes
        ipc_handle_array = (ctypes.c_byte * 64).from_buffer_copy(self.ipc_handle_Xld_B)
        cuda_driver = ctypes.CDLL('libcuda.so')
        gpu_mem_ptr = ctypes.c_void_p()
        cuIpcOpenMemHandle = cuda_driver.cuIpcOpenMemHandle
        cuIpcOpenMemHandle(ctypes.byref(gpu_mem_ptr), ipc_handle_array, 1)
        cuMemcpyDtoD = cuda_driver.cuMemcpyDtoD
        cuMemcpyDtoD(self.cuda_Xld_px.ptr, gpu_mem_ptr, int(self.N * self.Mld * 4))



        """ # 1. get device pointer to the appropriate Xld on the GPU
        Xld_mem = None
        if self.isPhaseA.value:
            Xld_mem = cuda.IPCMemoryHandle(self.ipc_handle_Xld_B)
            # Xld_mem = cuda.ipc_open_mem_handle(self.ipc_handle_Xld_B)
        else:
            Xld_mem = cuda.IPCMemoryHandle(self.ipc_handle_Xld_A)
            # Xld_mem = cuda.ipc_open_mem_handle(self.ipc_handle_Xld_A)
        Xld_ptr = Xld_mem.open(cuda.Context.get_current()) """

        """ # 2. copy the data to self.cuda_Xld_px
        # cuda.memcpy_dtod(self.cuda_Xld_px.gpudata, Xld_ptr, self.N * self.Mld * 4)
        cuda.memcpy_dtod(int(self.cuda_Xld_px.gpudata), int(Xld_ptr), self.N * self.Mld * 4)

        # 3. scale the data with tis min and max values
        # TODO
        print("TODO: scale the data with tis min and max values") """


        """ CUDA ipc: ne marche pas sur windows... 
        carrement retirer multiprocessing et fair multithreading 
        ==> le gui_thread dormira la plupart du temps """

class FastSNE_gui:
    def __init__(self, smem_cpu_Xld_A, smem_cpu_Xld_B, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated, window_w=640, window_h=480):
        self.window = ModernGLWindow(smem_cpu_Xld_A, smem_cpu_Xld_B, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated, width=window_w, height=window_h, caption='fastSNE')
        pyglet.clock.schedule_interval(self.window.update, 1.0/__TARGET_FPS__)
        pyglet.app.run() # blocks until the window is closed, everything is event-driven from there on

def gui_worker(smem_cpu_Xld_A, smem_cpu_Xld_B, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated):
    # ipc for Xld_A and Xld_B
    gui = FastSNE_gui(smem_cpu_Xld_A, smem_cpu_Xld_B, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, isPhaseA, points_updated, window_w=800, window_h=600)

    # notify the main process that the GUI has been closed
    with gui_closed.get_lock():
        pyglet.app.exit()
        gui_closed.value = True
    
    return


