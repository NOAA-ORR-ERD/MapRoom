import os
import time

import wx
import math
import numpy as np

from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_cube

from renderer_driver import RendererDriver

import logging
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)

vertex = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_color;

attribute vec3 position;
// texcoord removed
attribute vec3 normal;
attribute vec4 color;

varying vec4 v_color;
void main()
{
    v_color = u_color * color;
    gl_Position = u_projection * u_view * u_model * vec4(position,1.0);
}
"""

fragment = """
varying vec4 v_color;
void main()
{
    gl_FragColor = v_color;
}
"""

class BaseCanvas(app.Canvas):

    """
    The core rendering class for MapRoom app.
    """
    
    shared_context = None
    
    @classmethod
    def init_context(cls, canvas):
        # Only one GLContext is needed for the entire application -- this way,
        # textures can be shared among views.
        if cls.shared_context is None:
            pass

    def __init__(self, *args, **kwargs):
        app.Canvas.__init__(self, app="wx", parent=args[0])

        self.init_context(self)
        self.vispy_renderer = None
    
    def get_native_control(self):
        return self.native

    def on_initialize(self, event):
        # Build cube data
        V, I, O = create_cube()
        # Each item in the vertex data V is a tuple of lists, e.g.:
        #
        # ([1.0, 1.0, 1.0], [0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0])
        #
        # where each list corresponds to one of the attributes in the vertex
        # shader.  (See vispy.geometry.create_cube).  To experiment with this,
        # I've removed the 2nd tuple (the texcoord) and changed the vertex
        # shader from vispy5.  We'll see if this works. UPDATE: yes, it does!
        v2type = [('position', np.float32, 3),
                  ('normal', np.float32, 3),
                  ('color',    np.float32, 4)]
        v2 = np.zeros(V.shape[0], v2type)
        v2['position'] = V['position']
        v2['normal'] = V['normal']
        v2['color'] = V['color']
        print V[0]
        print v2[0]
        vertices = VertexBuffer(v2)
        self.faces = IndexBuffer(I)
        self.outline = IndexBuffer(O)

        # Build program
        # --------------------------------------
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)

        # Build view, model, projection & normal
        # --------------------------------------
        view = np.eye(4, dtype=np.float32)
        model = np.eye(4, dtype=np.float32)
        translate(view, 0, 0, -5)
        self.program['u_model'] = model
        self.program['u_view'] = view
 
        # OpenGL initalization
        # --------------------------------------
        gloo.set_state(clear_color=(1.0, 1.0, 1.0, 1.00), depth_test=True,
                       polygon_offset=(1, 1), line_width=0.75,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)

        # Filled cube
        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True)
        self.program['u_color'] = 1, 1, 1, 1
        self.program.draw('triangles', self.faces)

        # Outlined cube
        gloo.set_state(blend=True, depth_mask=False, polygon_offset_fill=False)
        self.program['u_color'] = 0, 0, 0, 1
        self.program.draw('lines', self.outline)
        gloo.set_state(depth_mask=True)

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)
        projection = perspective(45.0, event.size[0] / float(event.size[1]),
                                 2.0, 10.0)
        self.program['u_projection'] = projection

    def on_mouse_down(self, event):
        # self.SetFocus() # why would it not be focused?
        mouselog.debug("in on_mouse_down: event=%s" % event)
        self.get_effective_tool_mode(event)  # update alt key state
        self.forced_cursor = None
        self.mouse_is_down = True
        self.selection_box_is_being_defined = False
        self.mouse_down_position = event.GetPosition()
        self.mouse_move_position = self.mouse_down_position

        self.mouse_handler.process_mouse_down(event)
        self.set_cursor()

    def on_mouse_motion(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        if self.mouse_is_down:
            self.mouse_handler.process_mouse_motion_down(event)
        else:
            self.mouse_handler.process_mouse_motion_up(event)
        self.set_cursor()

    def on_mouse_up(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        self.forced_cursor = None
        
        self.mouse_handler.process_mouse_up(event)
        self.set_cursor()

    def on_mouse_wheel_scroll(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        
        self.mouse_handler.process_mouse_wheel_scroll(event)
        self.set_cursor()

    def on_mouse_enter(self, event):
        self.set_cursor()

    def on_mouse_leave(self, event):
        self.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        
        self.mouse_handler.process_mouse_leave(event)

    def on_key_down(self, event):
        self.get_effective_tool_mode(event)
        
        self.mouse_handler.process_key_down(event)
        self.set_cursor()
        
        event.Skip()

    def on_key_up(self, event):
        self.get_effective_tool_mode(event)
        
        self.mouse_handler.process_key_up(event)
        self.set_cursor()
        
        event.Skip()

    def on_key_char(self, event):
        self.get_effective_tool_mode(event)
        self.set_cursor()
        
        self.mouse_handler.process_key_char(event)

    def render(self, event=None):
        # Get interactive console here:
#        import traceback
#        traceback.print_stack();
#        import code; code.interact( local = locals() )
        if not self.native.IsShownOnScreen():
            # log.debug("layer_control_wx.render: not shown yet, so skipping render")
            return
        
        if True:
            if not hasattr(self, 'program'):
                self.on_initialize(None)
            self.on_draw(event)
            return

        t0 = time.clock()
        self.SetCurrent(self.context)
        # this has to be here because the window has to exist before making the renderer
        if (self.vispy_renderer is None):
            self.vispy_renderer = RendererDriver(True)
        self.update_renderers()

        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)

        if not self.vispy_renderer.prepare_to_render(p_r, s_r):
            return

        """
        self.root_renderer.render()
        self.set_screen_projection_matrix()
        self.box_overlay.render()
        self.set_render_projection_matrix()
        """

        ## fixme -- why is this a function defined in here??
        ##   so that it can be called with and without pick-mode turned on
        ##   but it seems to be in the wrong place -- make it a regular  method?
        def render_layers(pick_mode=False):
            list = self.layer_manager.flatten()
            length = len(list)
            self.layer_manager.pick_layer_index_map = {} # make sure it's cleared
            pick_layer_index = -1
            for i, layer in enumerate(reversed(list)):
                if pick_mode:
                    if layer.pickable:
                        pick_layer_index += 1
                        self.layer_manager.pick_layer_index_map[pick_layer_index] = (length - 1 - i) # looping reversed...
                        renderer = self.layer_renderers[layer]
                        layer.render(self.vispy_renderer,
                                     renderer,
                                     w_r, p_r, s_r,
                                     self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                     pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                     pick_mode)
                else: # not in pick-mode
                    renderer = self.layer_renderers[layer]
                    layer.render(self.vispy_renderer,
                                 renderer,
                                 w_r, p_r, s_r,
                                 self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                 pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                 pick_mode)

        render_layers()

        self.vispy_renderer.prepare_to_render_screen_objects()
        if (self.bounding_boxes_shown):
            self.draw_bounding_boxes()
        
        self.mouse_handler.render_overlay()

        self.SwapBuffers()

        self.vispy_renderer.prepare_to_render_projected_objects()
        self.vispy_renderer.prepare_to_render_picker(s_r)
        render_layers(pick_mode=True)
        self.vispy_renderer.done_rendering_picker()

        elapsed = time.clock() - t0

        def update_status(message):
            self.project.task.status_bar.debug = message
        wx.CallAfter(update_status, "Render complete, took %f seconds." % elapsed)

        if (event is not None):
            event.Skip()

    def resize_render_pane(self, event):
        if not self.GetContext():
            return

        event.Skip()
        self.render(event)

    # functions related to world coordinates, projected coordinates, and screen coordinates

    def get_screen_size(self):
        return self.native.GetClientSize()

    def get_canvas_as_image(self):
        window_size = self.native.GetClientSize()

        gl.glReadBuffer(gl.GL_FRONT)

        raw_data = gl.glReadPixels(
            x=0,
            y=0,
            width=window_size[0],
            height=window_size[1],
            format=gl.GL_RGB,
            type=gl.GL_UNSIGNED_BYTE,
            outputType=str,
        )

        bitmap = wx.BitmapFromBuffer(
            width=window_size[0],
            height=window_size[1],
            dataBuffer=raw_data,
        )

        image = wx.ImageFromBitmap(bitmap)

        # Flip the image vertically, because glReadPixel()'s y origin is at
        # the bottom and wxPython's y origin is at the top.
        screenshot = image.Mirror(horizontally=False)
        return screenshot
