# import the classics
import numpy as np
import multiprocessing
from multiprocessing import shared_memory

# import modernGL (renderer) and pyglet (windowerer)
import moderngl as mgl
import pyglet
from pyglet import shapes

# from .fastSNE import (__MAX_PERPLEXITY__, __MIN_PERPLEXITY__, __MAX_KERNEL_ALPHA__, __MIN_KERNEL_ALPHA__, __MAX_ATTRACTION_MULTIPLIER__, __MIN_ATTRACTION_MULTIPLIER__)

__TARGET_FPS__  = 120.0
__AMBER_LIGHT__ = (255, 191, 0)
__AMBER_DARK__  = (70, 35, 0)

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

class VerticalSlider:
    def __init__(self, x, y, width, height, min_val, max_val, value, window_width, window_height, title):
        # Convert NDC to window coordinates
        self.x = (x + 1) * window_width / 2
        self.y = (y + 1) * window_height / 2
        self.width = width * window_width / 2
        self.height = height * window_height / 2
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.dragging = False
        self.thin_rail = shapes.BorderedRectangle(self.x+0.5*self.width - 2, self.y, 4, self.height, border=1, color=__AMBER_DARK__, border_color=__AMBER_LIGHT__)
        self.handle    = shapes.Rectangle(self.x, self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height, self.width, 6, color=__AMBER_DARK__)
        self.handle_highlight = shapes.BorderedRectangle(self.x, self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height, self.width, 6, border=1, color=(0,0,0), border_color=__AMBER_LIGHT__)
        self.tiny_line = shapes.Line(self.x + 0.3*self.width, 2 + self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height, self.x+ + 0.7*self.width, 2 + self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height, width=1, color=__AMBER_LIGHT__)
        self.label_value = pyglet.text.Label(str(self.value), x=self.x + self.width + 8, y=2 + self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height, anchor_x='center', anchor_y='center', color=__AMBER_LIGHT__, font_size=9)
        self.label_max = pyglet.text.Label(str(self.max_val), x=self.x + self.width / 3, y=self.y + self.height + 7, anchor_x='center', anchor_y='center', color=__AMBER_LIGHT__, font_size=9)
        self.label_min = pyglet.text.Label(str(self.min_val), x=self.x + self.width / 3, y=self.y - 9, anchor_x='center', anchor_y='center', color=__AMBER_LIGHT__, font_size=9)
        self.label_title = pyglet.text.Label(title, x=self.x + self.width / 3, y=self.y + self.height + 26, anchor_x='center', anchor_y='center', color=__AMBER_LIGHT__, font_size=10)

    def draw(self):
        self.thin_rail.draw()
        self.handle.draw()
        self.handle_highlight.draw()
        self.tiny_line.draw()
        self.label_value.draw()
        self.label_max.draw()
        self.label_min.draw()
        self.label_title.draw()
    
    def value_change(self, value):
        self.value = value
        handle_y = self.y + (self.value - self.min_val) / (self.max_val - self.min_val) * self.height
        tiny_line_y = 2 + handle_y
        self.handle.y = handle_y
        self.handle_highlight.y = handle_y
        self.tiny_line.y = tiny_line_y
        self.label_value.y = tiny_line_y
        self.label_value.text = str(self.value)

    def is_inside(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def update(self, x, y):
        if self.dragging:
            self.value = self.min_val + (y - self.y) / self.height * (self.max_val - self.min_val)
            if self.value < self.min_val:
                self.value = self.min_val
            if self.value > self.max_val:
                self.value = self.max_val
            # round to 2 decimals
            self.value = round(self.value, 2)
            self.value_change(self.value)
            print("Value changed to", self.value)

class Button:
    def __init__(self, x, y, width, height, text, window_width, window_height):
        # Convert NDC to window coordinates
        self.x = (x + 1) * window_width / 2
        self.y = (y + 1) * window_height / 2
        self.width = width * window_width / 2
        self.height = height * window_height / 2
        self.text = text
        self.pressed = False
        self.rect  = shapes.Rectangle(self.x, self.y, self.width, self.height)
        self.label = pyglet.text.Label(self.text, x=self.x + self.width / 2, y=self.y + self.height / 2,
                                  anchor_x='center', anchor_y='center')

    def draw(self):
        self.rect.color  = __AMBER_LIGHT__ if self.pressed else __AMBER_DARK__
        self.label.color = __AMBER_DARK__ if self.pressed else __AMBER_LIGHT__
        self.rect.draw()
        self.label.draw()

    def is_inside(self, x, y):
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

class ModernGLWindow(pyglet.window.Window):
    def __init__(self, cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please,\
                 min_perplexity, max_perplexity, min_kernel_alpha, max_kernel_alpha, min_attraction_mul, max_attraction_mul, **kwargs):
        config = pyglet.gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4)
        # config = pyglet.gl.Config(double_buffer=True)
        # super().__init__(config=config,vsync=True, **kwargs)
        super().__init__(config=config, vsync=True, **kwargs)

        if(Mld != 2):
            raise ValueError("Gui not implemented for Mld != 2")
        if(N < 5):
            raise ValueError("Just do your embedding manually at this point")
        
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity
        self.min_kernel_alpha = min_kernel_alpha
        self.max_kernel_alpha = max_kernel_alpha
        self.min_attraction_mul = min_attraction_mul
        self.max_attraction_mul = max_attraction_mul

        # -------   data for the GUI   -------
        self.N   = N
        self.Mld = Mld
        # hyperparameters
        self.kernel_alpha = kernel_alpha # multiprocessing Value type
        self.perplexity   = perplexity   # multiprocessing Value type
        self.attrac_mult  = attrac_mult
        self.dist_metric  = dist_metric  # multiprocessing Value type
        # communication with the main process
        self.gui_closed   = gui_closed        # multiprocessing Value type
        self.points_ready_for_rendering = points_ready_for_rendering # multiprocessing Value type
        self.points_rendering_finished  = points_rendering_finished  # multiprocessing Value type
        self.iteration = iteration # multiprocessing Value type
        self.explosion_request = explosion_please # multiprocessing Value type
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
        
        self.ctrl_held = False

        window_width, window_height = self.get_size()
        self.buttons = [
            Button(-0.99, -0.99, 0.23, 0.098, "Euclidean", window_width, window_height),
            Button(-0.99, -0.89, 0.23, 0.098, "Manhattan", window_width, window_height),
            Button(-0.99, -0.79, 0.23, 0.098, "Cosine", window_width, window_height),
            Button(-0.99, -0.69, 0.23, 0.098, "Custom", window_width, window_height)
        ]
        self.buttons[self.dist_metric.value].pressed = True

        self.explosion_request_button = Button(0.55, -0.99, 0.44, 0.098, "Explosion please", window_width, window_height)

        self.slider_perplexity   = VerticalSlider(-0.9, 0.4, 0.05, 0.3, self.min_perplexity, self.max_perplexity, self.perplexity.value, window_width, window_height, "Perplexity")
        self.slider_kernel_alpha = VerticalSlider(0.85, 0.4, 0.05, 0.3, self.min_kernel_alpha, self.max_kernel_alpha, self.kernel_alpha.value, window_width, window_height, "Kernel alpha")
        self.slider_attrac_mult  = VerticalSlider(0.85, -0.2, 0.05, 0.3, self.min_attraction_mul, self.max_attraction_mul, self.attrac_mult.value, window_width, window_height, "Attraction")
        # label containing the iteration number
        self.label_iteration = pyglet.text.Label(f"Iteration: {self.iteration.value}", x=window_width - 10, y=window_height - 10, anchor_x='right', anchor_y='top', color=__AMBER_DARK__, font_size=12)

        slider_01_value = 0.5
        nonlinear_01_value = (((slider_01_value-0.5)*np.abs(slider_01_value-0.5))/0.25)*0.5 + 0.5
        alpha_value = 1.0
        if nonlinear_01_value < 0.5:
            alpha_value = self.min_kernel_alpha + 2.0 * slider_01_value * (1.0 - self.min_kernel_alpha)
        else:
            alpha_value = 1.0 + 2.0 * slider_01_value * (self.max_kernel_alpha - 1.0)
        self.slider_kernel_alpha.value = round(alpha_value, 2)
        self.value = alpha_value
        handle_y = self.slider_kernel_alpha.y + slider_01_value * self.slider_kernel_alpha.height
        tiny_line_y = 2 + handle_y
        self.slider_kernel_alpha.handle.y = handle_y
        self.slider_kernel_alpha.handle_highlight.y = handle_y
        self.slider_kernel_alpha.tiny_line.y = tiny_line_y
        self.slider_kernel_alpha.label_value.y = tiny_line_y
        self.slider_kernel_alpha.label_value.text = str(self.value)

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

        for button in self.buttons:
            button.draw()
        self.explosion_request_button.draw()
        
        self.slider_perplexity.draw()
        self.slider_kernel_alpha.draw()
        self.slider_attrac_mult.draw()

        self.label_iteration.draw()

        # notify the main process that the rendering is done
        with self.points_rendering_finished.get_lock():
            self.points_rendering_finished.value = True
    
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
            self.label_iteration.text = f"Iteration: {self.iteration.value}"

    def on_mouse_press(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            if self.slider_perplexity.is_inside(x, y):
                self.slider_perplexity.dragging = True
            if self.slider_kernel_alpha.is_inside(x, y):
                self.slider_kernel_alpha.dragging = True
            if self.slider_attrac_mult.is_inside(x, y):
                self.slider_attrac_mult.dragging = True
    
    def on_mouse_release(self, x, y, button, modifiers):
        if button == pyglet.window.mouse.LEFT:
            if button == pyglet.window.mouse.LEFT:
                self.slider_perplexity.dragging = False
                self.slider_kernel_alpha.dragging = False
                self.slider_attrac_mult.dragging = False

            btn_idx = -1
            for idx, btn in enumerate(self.buttons):
                if btn.is_inside(x, y):
                    btn_idx = idx
            if btn_idx != -1:
                for idx, btn in enumerate(self.buttons):
                    if idx != btn_idx:
                        btn.pressed = False
                    else:
                        btn.pressed = True
                        with self.dist_metric.get_lock():
                            self.dist_metric.value = btn_idx
            else:
                if self.explosion_request_button.is_inside(x, y):
                    with self.explosion_request.get_lock():
                        self.explosion_request.value = True
            
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.LEFT:
            if self.slider_perplexity.dragging:
                self.slider_perplexity.update(x, y)
                with self.perplexity.get_lock():
                    self.perplexity.value = self.slider_perplexity.value
            if self.slider_kernel_alpha.dragging:
                #self.slider_kernel_alpha.update(x, y)
                
                slider_01_value = (y - self.slider_kernel_alpha.y) / self.slider_kernel_alpha.height
                if(slider_01_value < 0.0):
                    slider_01_value = 0.0
                if(slider_01_value > 1.0):
                    slider_01_value = 1.0
                nonlinear_01_value = (((slider_01_value-0.5)*np.abs(slider_01_value-0.5))/0.25)*0.5 + 0.5

                alpha_value = 1.0
                if nonlinear_01_value < 0.5:
                    alpha_value = self.min_kernel_alpha + 2.0 * slider_01_value * (1.0 - self.min_kernel_alpha)
                else:
                    alpha_value = ((slider_01_value - 0.5) * 2.0) * (self.max_kernel_alpha - 1.0) + 1.0

                self.slider_kernel_alpha.value = round(alpha_value, 2)
                self.value = alpha_value
                handle_y = self.slider_kernel_alpha.y + slider_01_value * self.slider_kernel_alpha.height
                tiny_line_y = 2 + handle_y
                self.slider_kernel_alpha.handle.y = handle_y
                self.slider_kernel_alpha.handle_highlight.y = handle_y
                self.slider_kernel_alpha.tiny_line.y = tiny_line_y
                self.slider_kernel_alpha.label_value.y = tiny_line_y
                self.slider_kernel_alpha.label_value.text = str(self.value)

                with self.kernel_alpha.get_lock():
                    self.kernel_alpha.value = self.slider_kernel_alpha.value
            if self.slider_attrac_mult.dragging:
                self.slider_attrac_mult.update(x, y)
                with self.attrac_mult.get_lock():
                    self.attrac_mult.value = self.slider_attrac_mult.value
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            increment = 1 if not self.ctrl_held else 6
            prev_perplexity = self.perplexity.value
            new_perplexity = prev_perplexity + increment
            if new_perplexity > self.max_perplexity:
                new_perplexity = self.max_perplexity
            if new_perplexity < self.min_perplexity:
                new_perplexity = self.min_perplexity
            with self.perplexity.get_lock():
                self.perplexity.value = new_perplexity
            self.slider_perplexity.value_change(new_perplexity)
        elif scroll_y < 0:
            increment = -1 if not self.ctrl_held else -6
            prev_perplexity = self.perplexity.value
            new_perplexity = prev_perplexity + increment
            if new_perplexity > self.max_perplexity:
                new_perplexity = self.max_perplexity
            if new_perplexity < self.min_perplexity:
                new_perplexity = self.min_perplexity
            with self.perplexity.get_lock():
                self.perplexity.value = new_perplexity
            self.slider_perplexity.value_change(new_perplexity)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        if symbol == pyglet.window.key.LCTRL:
            self.ctrl_held = True
    
    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.LCTRL:
            self.ctrl_held = False

class FastSNE_gui:
    def __init__(self, cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please, min_perplexity, max_perplexity, min_kernel_alpha, max_kernel_alpha, min_attraction_mul, max_attraction_mul,window_w=640, window_h=480):
        self.window = ModernGLWindow(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please, min_perplexity, max_perplexity, min_kernel_alpha, max_kernel_alpha, min_attraction_mul, max_attraction_mul, width=window_w, height=window_h, caption='fastSNE')
        pyglet.clock.schedule_interval(self.window.update, 1.0/__TARGET_FPS__)
        pyglet.app.run() # blocks until the window is closed, everything is event-driven from there on

def gui_worker(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please, min_perplexity, max_perplexity, min_kernel_alpha, max_kernel_alpha, min_attraction_mul, max_attraction_mul):
    # ipc for Xld_A and Xld_B
    gui = FastSNE_gui(cpu_shared_mem, cpu_Y, N, Mld, kernel_alpha, perplexity, attrac_mult, dist_metric, gui_closed, points_ready_for_rendering, points_rendering_finished, iteration, explosion_please, min_perplexity, max_perplexity, min_kernel_alpha, max_kernel_alpha, min_attraction_mul, max_attraction_mul, window_w=800, window_h=800)

    # notify the main process that the GUI has been closed
    with gui_closed.get_lock():
        pyglet.app.exit()
        gui_closed.value = True
    
    return


