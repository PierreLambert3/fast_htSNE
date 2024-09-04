# import the classics
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import modernGL (renderer) and pyglet (windowerer)
import moderngl as mgl
import pyglet

__TARGET_FPS__ = 120.0

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
    return cpu_Y_colours

class ModernGLWindow(pyglet.window.Window):
    def __init__(self, cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, **kwargs):
        # config = pyglet.gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
        config = pyglet.gl.Config(double_buffer=True)
        super().__init__(config=config,vsync=True, **kwargs)

        if(Mld != 2):
            raise ValueError("Gui not implemented for Mld != 2")
        if(N < 5):
            raise ValueError("Just do your embedding manually at this point")

        # -------   data for the GUI   -------
        self.N   = N
        self.Mld = Mld
        # hyperparameters
        self.kernel_alpha = kernel_alpha # multiprocessing Value type
        self.perplexity   = perplexity   # multiprocessing Value type
        self.dist_metric  = dist_metric  # multiprocessing Value type
        # communication with the main process
        self.gui_closed   = gui_closed        # multiprocessing Value type
        self.points_ready_for_rendering = points_ready_for_rendering # multiprocessing Value type
        self.points_rendering_finished  = points_rendering_finished  # multiprocessing Value type
        # colours for each point, on CPU
        self.Y_colours = determine_Y_colour(cpu_Y) # size (N, 3), colour of each observation (removes an "if" in the render function)
        # Xld on CPU
        self.cpu_Xld = np.ndarray((N, Mld), dtype=np.float32, buffer=cpu_shared_mem.buf)
        self.redraw_now = False

        # -------   modernGL structs   -------
        self.ctx = mgl.create_context()
        #  Create VBO for positions
        self.vbo_positions = self.ctx.buffer(self.cpu_Xld.astype('f4').tobytes())
        #  Create VBO for colors
        self.vbo_colors = self.ctx.buffer(self.Y_colours.astype('f4').tobytes())
        #  Create VAO
        self.vao_content = [
            (self.vbo_positions, '2f', 'in_vert'),  # 2 floats per vertex position
            (self.vbo_colors, '3f', 'in_color')     # 3 floats per color (RGB)
        ]
        #  define the shaders & compile them
        vertex_shader, fragment_shader = self.setup_shaders()
        self.shader_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.vao = self.ctx.vertex_array(self.shader_program, self.vao_content)

        # prepare the data for the first frame
        self.retrieve_and_prepare_data()
    
    def setup_shaders(self):
        #TODO cool neon effects
        vertex_shader = '''
        #version 330
        in vec2 in_vert;
        in vec3 in_color;
        out vec3 color;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            color = in_color;
        }
        '''
        fragment_shader = '''
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        '''
        return vertex_shader, fragment_shader

    def on_draw(self):
        # only draw when I want to
        if not self.redraw_now:
            return
        self.redraw_now = False
        
        # render the screen
        self.ctx.clear()
        self.vao.render(mgl.POINTS)

        # notify the main process that the rendering is done
        with self.points_rendering_finished.get_lock():
            self.points_rendering_finished.value = True
        # print("drawing")
    
    def on_close(self):
        self.vbo_positions.release()
        self.vbo_colors.release()
        self.vao.release()
        self.ctx.release()
        super().on_close()

    def retrieve_and_prepare_data(self):
        updated_data = self.cpu_Xld.astype('f4').tobytes()
        self.vbo_positions.write(updated_data)
    
    def update(self, dt):
        # only draw if the points were updated
        with self.points_ready_for_rendering.get_lock():
            points_ready = self.points_ready_for_rendering.value
            self.points_ready_for_rendering.value = False
        if points_ready:
            self.retrieve_and_prepare_data()
            self.redraw_now = True
        

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
    def __init__(self, cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, window_w=640, window_h=480):
        self.window = ModernGLWindow(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, width=window_w, height=window_h, caption='fastSNE')
        pyglet.clock.schedule_interval(self.window.update, 1.0/__TARGET_FPS__)
        pyglet.app.run() # blocks until the window is closed, everything is event-driven from there on

def gui_worker(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished):
    # ipc for Xld_A and Xld_B
    gui = FastSNE_gui(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, window_w=800, window_h=600)

    # notify the main process that the GUI has been closed
    with gui_closed.get_lock():
        pyglet.app.exit()
        gui_closed.value = True
    
    return


