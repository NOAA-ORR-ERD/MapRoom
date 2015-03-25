import os
import time

import wx
import wx.glcanvas as glcanvas

import math
import numpy as np

import OpenGL.GL as gl
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import OpenGL.GLU as glu

from renderer import ImmediateModeRenderer
import maproom.library.rect as rect
import Picker

from ..gl.font import load_font_texture_with_alpha

import logging
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)


class BaseCanvas(glcanvas.GLCanvas):

    """
    The core rendering class for MapRoom app.
    """
    
    shared_context = None
    
    @classmethod
    def init_context(cls, canvas):
        # Only one GLContext is needed for the entire application -- this way,
        # textures can be shared among views.
        if cls.shared_context is None:
            cls.shared_context = glcanvas.GLContext(canvas)

    def __init__(self, *args, **kwargs):
        kwargs['attribList'] = (glcanvas.WX_GL_RGBA,
                                glcanvas.WX_GL_DOUBLEBUFFER,
                                glcanvas.WX_GL_MIN_ALPHA, 8, )

        glcanvas.GLCanvas.__init__(self, *args, **kwargs)

        self.init_context(self)

        self.overlay = ImmediateModeRenderer(self, None)
        self.picker = Picker.Picker()

        self.screen_rect = rect.EMPTY_RECT

        (self.font_texture, self.font_texture_size, self.font_extents) = self.load_font_texture()
    
        #self.frame.Bind( wx.EVT_MOVE, self.refresh )
        #self.frame.Bind( wx.EVT_IDLE, self.on_idle )
        # self.frame.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )

        self.Bind(wx.EVT_PAINT, self.on_draw)
        # Prevent flashing on Windows by doing nothing on an erase background event.
        ## fixme -- I think you can pass a flag to the Window instead...
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
        self.Bind(wx.EVT_SIZE, self.on_resize)
        
        # mouse handler events
        self.mouse_handler = None  # defined in subclass
        
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
        self.Bind(wx.EVT_ENTER_WINDOW, self.on_mouse_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_CHAR, self.on_key_char)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_up)
        
        self.native = self.get_native_control()

    def get_native_control(self):
        return self

    def on_draw(self, event):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.render()

    def on_resize(self, event):
        if not self.GetContext():
            return

        event.Skip()
        self.render(event)

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
        r = ImmediateModeRenderer(self, layer)
        return r
    
    def load_font_texture(self):
        buffer_with_alpha, extents = load_font_texture_with_alpha()
        width = buffer_with_alpha.shape[0]
        height = buffer_with_alpha.shape[1]

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            buffer_with_alpha.tostring(),
        )
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        # gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )

        return (texture, (width, height), extents)

    def prepare_to_render(self, projected_rect, screen_rect):
        self.screen_rect = screen_rect
        self.s_w = rect.width(screen_rect)
        self.s_h = rect.height(screen_rect)
        self.projected_rect = projected_rect
        p_w = rect.width(projected_rect)
        p_h = rect.height(projected_rect)

        if (self.s_w <= 0 or self.s_h <= 0 or p_w <= 0 or p_h <= 0):
            return False
        
        gl.glDisable(gl.GL_LIGHTING)
        # Don't cull polygons that are wound the wrong way.
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glViewport(0, 0, self.s_w, self.s_h)
        self.set_up_for_regular_rendering()
        gl.glClearColor(1.0, 1.0, 1.0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        
        return True

    def prepare_to_render_picker(self, screen_rect):
        self.picker.prepare_to_render(screen_rect)
        self.set_up_for_picker_rendering()
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

    def done_rendering_picker(self):
        self.picker.done_rendering()
        self.set_up_for_regular_rendering()

    def set_up_for_regular_rendering(self):
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def set_up_for_picker_rendering(self):
        gl.glDisable(gl.GL_POINT_SMOOTH)
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glDisable(gl.GL_BLEND)

    def get_object_at_mouse_position(self, screen_point):
        if not self.native.IsShownOnScreen():
            return None
        return self.picker.get_object_at_mouse_position(screen_point)

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
        self.SetCurrent(self.shared_context)
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
        def render_layers(picker=None):
            list = self.layer_manager.flatten()
            length = len(list)
            self.layer_manager.pick_layer_index_map = {} # make sure it's cleared
            pick_layer_index = -1
            for i, layer in enumerate(reversed(list)):
                if picker is not None:
                    if layer.pickable:
                        pick_layer_index += 1
                        self.layer_manager.pick_layer_index_map[pick_layer_index] = (length - 1 - i) # looping reversed...
                        layer.render(self,
                                     w_r, p_r, s_r,
                                     self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                     pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                     picker)
                else: # not in pick-mode
                    layer.render(self,
                                 w_r, p_r, s_r,
                                 self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                 pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                 picker)

        render_layers()

        self.overlay.prepare_to_render_screen_objects()
        if (self.bounding_boxes_shown):
            self.draw_bounding_boxes()
        
        self.mouse_handler.render_overlay(self.overlay)

        self.SwapBuffers()

        self.prepare_to_render_picker(s_r)
        render_layers(picker=self.picker)
        self.done_rendering_picker()

        elapsed = time.clock() - t0

        def update_status(message):
            self.project.task.status_bar.debug = message
        wx.CallAfter(update_status, "Render complete, took %f seconds." % elapsed)

        if (event is not None):
            event.Skip()

    # functions related to world coordinates, projected coordinates, and screen coordinates

    def get_screen_size(self):
        return self.native.GetClientSize()

    def get_canvas_as_image(self):
        window_size = self.get_screen_size()

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
