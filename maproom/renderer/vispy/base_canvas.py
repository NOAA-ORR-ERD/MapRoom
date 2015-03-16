import os
import time

import wx
import math
import numpy as np

from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import ortho, translate, rotate
from vispy.geometry import create_cube

from renderer import VispyRenderer
import maproom.library.rect as rect
import Picker

import logging
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)

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

        self.overlay = VispyRenderer(self, None)
        self.picker = Picker.Picker()

        self.screen_rect = rect.EMPTY_RECT
        

        # Build view, model, projection & normal
        # --------------------------------------
        
        self.model_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix = np.eye(4, dtype=np.float32)
        translate(self.view_matrix, 0, 0, -5)
        self.projection_matrix = None
    
    def get_native_control(self):
        return self.native

    def on_initialize(self, event):
 
        # OpenGL initalization
        # --------------------------------------
        gloo.set_state(
            clear_color=(1.0, 1.0, 1.0, 1.00),
            depth_test=True,
            polygon_offset=(1, 1),
            lighting=False,
            line_width=0.75,
            cull_face=False, # Don't cull polygons that are wound the wrong way.
            blend_func=('src_alpha', 'one_minus_src_alpha'))
        gloo.clear(color=True, depth=True)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.render()

    def on_resize(self, event):
        self.s_w = event.size[0]
        self.s_h = event.size[1]
        gloo.set_viewport(0, 0, self.s_w, self.s_h)
        self.prepare_to_render_projected_objects()

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
    
    def get_renderer(self, layer):
        r = VispyRenderer(self, layer)
        return r
    
    def prepare_to_render(self, projected_rect, screen_rect):
        self.screen_rect = screen_rect
        self.s_w = rect.width(screen_rect)
        self.s_h = rect.height(screen_rect)
        self.projected_rect = projected_rect
        p_w = rect.width(projected_rect)
        p_h = rect.height(projected_rect)

        if (self.s_w <= 0 or self.s_h <= 0 or p_w <= 0 or p_h <= 0):
            return False
        
        gloo.set_viewport(0, 0, self.s_w, self.s_h)
        gloo.clear(color=True, depth=True)
        
        aspect = float(self.s_w) / self.s_h
        if aspect > 1:
            self.projection_matrix = ortho(-2 * aspect, 2 * aspect, -2, 2, 2.0, 10.0)
        else:
            self.projection_matrix = ortho(-2, 2, -2 / aspect, 2 / aspect, 2.0, 10.0)
        
        return True

    def prepare_to_render_picker(self, screen_rect):
        self.picker.prepare_to_render(screen_rect)
        self.set_up_for_picker_rendering()
        gloo.set_state(clear_color=(1.0, 1.0, 1.0, 1.00), depth_test=True)

    def done_rendering_picker(self):
        self.picker.done_rendering()
        self.set_up_for_regular_rendering()

    def set_up_for_regular_rendering(self):
        gloo.set_state(blend=True, line_smooth=True, point_smooth=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    def set_up_for_picker_rendering(self):
        gloo.set_state(blend=False, line_smooth=False, point_smooth=False)
#        gl.glDisable(gl.GL_POINT_SMOOTH)
#        gl.glDisable(gl.GL_LINE_SMOOTH)
#        # gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE )
#        gl.glDisable(gl.GL_BLEND)
#        # gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

    #
    # the methods below are used to render simple objects one at a time, in screen coordinates

    def render(self, event=None):
        # Get interactive console here:
#        import traceback
#        traceback.print_stack();
#        import code; code.interact( local = locals() )
        if not self.native.IsShownOnScreen():
            # log.debug("layer_control_wx.render: not shown yet, so skipping render")
            return

        t0 = time.clock()
        # this has to be here because the window has to exist before making the renderer
        self.update_renderers()

        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)

        if not self.prepare_to_render(p_r, s_r):
            return

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
                        layer.render(self,
                                     w_r, p_r, s_r,
                                     self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                     pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                     pick_mode)
                else: # not in pick-mode
                    layer.render(self,
                                 w_r, p_r, s_r,
                                 self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                 pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                 pick_mode)

        render_layers()

        self.overlay.prepare_to_render_screen_objects()
        if (self.bounding_boxes_shown):
            self.draw_bounding_boxes()
        
        self.mouse_handler.render_overlay()

        #self.SwapBuffers()

#        self.vispy_renderer.prepare_to_render_projected_objects()
#        self.vispy_renderer.prepare_to_render_picker(s_r)
#        render_layers(pick_mode=True)
#        self.vispy_renderer.done_rendering_picker()

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
