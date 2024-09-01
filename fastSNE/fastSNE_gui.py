# import the classics
import numpy as np
import multiprocessing

# import modernGL (renderer) and pyglet (windowerer)
import moderngl as mgl
import pyglet

__TARGET_FPS__ = 20.0

def gen_K_random_colours(K):
    n = K * 200
    dataset = np.random.uniform(0, 1, (n, 3))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=K, n_init=1, max_iter=25)
    kmeans.fit(dataset)
    return (kmeans.cluster_centers_ * 255.0).astype(np.uint8)

def determine_Y_colour(Y):
    cpu_Y_colours = np.zeros((len(Y), 3), dtype=np.uint8)
    is_classification  = type(Y[0, 0]) == np.int32
    if is_classification:
        label_colours = gen_K_random_colours(len(np.unique(Y)))
        for i in range(len(Y)):
            cpu_Y_colours[i, :] = label_colours[Y[i, 0]]
    else:
        min_Y = np.min(Y)
        max_Y = np.max(Y)
        span = max_Y - min_Y
        if span == 0:
            span = 1.0
        red0   = 170.0
        green0 = 0.0
        blue0  = 210.0
        red1   = 150.0
        green1 = 210.0
        blue1  = 0.0
        for i in range(len(Y)):
            y_normed = (Y[i, 0] - min_Y) / span
            cpu_Y_colours[i, 0] = np.uint8(y_normed*red1 + (1.0-y_normed)*red0)
            cpu_Y_colours[i, 1] = np.uint8(y_normed*green1 + (1.0-y_normed)*green0)
            cpu_Y_colours[i, 2] = np.uint8(y_normed*blue1 + (1.0-y_normed)*blue0)
    from pycuda import gpuarray
    cuda_Y_colours = gpuarray.to_gpu(cpu_Y_colours)
    return cuda_Y_colours

class Shared_CPU_variables:
    def __init__(self, init_kern_alpha, init_perplexity, init_dist_metric):
        # mutliprocessing Value types
        self.kern_alpha  = multiprocessing.Value('f', init_kern_alpha)
        self.perplexity  = multiprocessing.Value('f', init_perplexity)
        self.dist_metric = multiprocessing.Value('i', init_dist_metric)
        self.gui_closed  = multiprocessing.Value('b', False)
        self.isPhaseA    = multiprocessing.Value('b', True)
        self.iterDone1   = multiprocessing.Value('i', 0)
        self.iterDone2   = multiprocessing.Value('i', 0)
        self.points_have_moved = multiprocessing.Value('b', True)
    
    def get_reading_Xld(self):
        return self.cuda_Xld_A if self.isPhaseA.value else self.cuda_Xld_B
    
    def get_writing_Xld(self):
        return self.cuda_Xld_B if self.isPhaseA.value else self.cuda_Xld_A

class ModernGLWindow(pyglet.window.Window):
    def __init__(self, cuda_Xld_A, cuda_Xld_B, Y, shared_CPU_variables, N, Mld, **kwargs):
        super().__init__(**kwargs)
        self.shared_CPU_variables = shared_CPU_variables
        self.cuda_Xld_A = cuda_Xld_A
        self.cuda_Xld_B = cuda_Xld_B
        self.Y = Y
        self.cuda_Y_colours = determine_Y_colour(Y) # size (N, 3), colour for each observation (removes an "if" in the render function)
        self.ctx = mgl.create_context()
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(1.0, 1.0, 1.0, 1.0);
                }
            ''',
        )
        self.vbo = self.ctx.buffer(reserve=N * 2 * 4)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

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
        with self.shared_CPU_variables.points_have_moved.get_lock():
            point_have_moved = self.shared_CPU_variables.points_have_moved.value
            self.shared_CPU_variables.points_have_moved.value = False
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

class FastSNE_gui:
    def __init__(self, cuda_Xld_A, cuda_Xld_B, Y, shared_CPU_variables, N: np.uint32, Mld: np.uint32, window_w: np.uint32=640, window_h: np.uint32=480):
        self.window = ModernGLWindow(cuda_Xld_A, cuda_Xld_B, Y, shared_CPU_variables, N, Mld, width=window_w, height=window_h, caption='fastSNE')
        pyglet.clock.schedule_interval(self.window.update, 1.0/__TARGET_FPS__)
        pyglet.app.run() # blocks until the window is closed, everything is event-driven from there on

def gui_worker(ipc_handles_list, cpu_Y, shared_CPU_variables, N, Mld):
    # import & init pycuda
    import pycuda.driver as cuda
    import pycuda.autoinit
    cuda.init()
    cuda_context = cuda.Device(0).make_context()

    # ipc for Xld_A and Xld_B
    cuda_Xld_A = ipc_handles_list[0]
    cuda_Xld_B = ipc_handles_list[1]
    gui = FastSNE_gui(cuda_Xld_A, cuda_Xld_B, cpu_Y, shared_CPU_variables, N, Mld, window_w=800, window_h=600)

    # notify the main process that the GUI has been closed
    with shared_CPU_variables.gui_closed.get_lock():
        pyglet.app.exit()
        shared_CPU_variables.gui_closed.value = True
    
    # release the pycuda context
    cuda_context.pop()
    return


