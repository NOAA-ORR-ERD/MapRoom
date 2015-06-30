import sys
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

import OpenGL.GL.EXT.framebuffer_object as gl_fbo

import maproom.library.rect as rect

from ..gl.color import *
# import Image, ImageDraw

MAX_PICKER_OFFSET = 4

POINTS_PICKER_OFFSET = 0
LINES_PICKER_OFFSET = 1
FILL_PICKER_OFFSET = 2

def get_picker_index_base(layer_index):
    return layer_index * MAX_PICKER_OFFSET

class Picker(object):
    """
    An off-screen buffer for detecting what object the mouse is over.
    """

    CHANNELS = 4  # i.e., RGBA
    BACKGROUND_COLOR = color_to_int(1, 1, 1, 1)
    BACKGROUND_COLOR_STRING = chr(255) * 3
    frame_buffer = None  # the container frame buffer object
    render_buffer = None  # the render buffer within the frame buffer for offscreen drawing
    screen_rect = rect.EMPTY_RECT
    vbo_colors = None
    
    is_active = True

    def __init__(self):
        # self.colored_objects = {} # renderer -> ( index, original color )
        gl_fbo.glInitFramebufferObjectEXT()

    def destroy(self):
        self.vbo_colors = None

    def prepare_to_render(self, screen_rect):
        if (screen_rect != self.screen_rect and self.frame_buffer is not None):
            gl_fbo.glDeleteFramebuffersEXT(
                1, np.array([self.frame_buffer], dtype=np.uint32),
            )
            gl_fbo.glDeleteRenderbuffersEXT(
                1, np.array([self.render_buffer], dtype=np.uint32),
            )
            self.frame_buffer = None
            self.render_buffer = None

        if (self.frame_buffer is None):
            self.frame_buffer = int(gl_fbo.glGenFramebuffersEXT(1))
            self.bind_frame_buffer()

            self.render_buffer = int(gl_fbo.glGenRenderbuffersEXT(1))
            gl_fbo.glBindRenderbufferEXT(
                gl_fbo.GL_RENDERBUFFER_EXT,
                self.render_buffer
            )
            gl_fbo.glRenderbufferStorageEXT(
                gl_fbo.GL_RENDERBUFFER_EXT,
                gl.GL_RGBA,
                rect.width(screen_rect),
                rect.height(screen_rect)
            )
            gl_fbo.glFramebufferRenderbufferEXT(
                gl_fbo.GL_FRAMEBUFFER_EXT,
                gl_fbo.GL_COLOR_ATTACHMENT0_EXT,
                gl_fbo.GL_RENDERBUFFER_EXT,
                self.render_buffer
            )
        else:
            self.bind_frame_buffer()

        self.screen_rect = screen_rect

    def done_rendering(self):
        self.unbind_frame_buffer()

    def bind_frame_buffer(self):
        gl.glDrawBuffer(gl.GL_NONE)
        gl_fbo.glBindFramebufferEXT(gl_fbo.GL_FRAMEBUFFER_EXT, self.frame_buffer)

    def unbind_frame_buffer(self):
        gl_fbo.glBindFramebufferEXT(gl_fbo.GL_FRAMEBUFFER_EXT, 0)
        gl.glDrawBuffer(gl.GL_BACK)

    def bind_picker_colors_for_lines(self, layer_index, object_count):
        self.bind_picker_colors(layer_index + LINES_PICKER_OFFSET,
                                object_count, True)

    def bind_picker_colors_for_points(self, layer_index, object_count):
        self.bind_picker_colors(layer_index + POINTS_PICKER_OFFSET,
                                object_count, False)

    def bind_picker_colors(self, layer_index, object_count, doubled=False):
        """
        bind the colors in the OpenGL context (right word?)

        for a bunch of object in a layer

        :param layer_index: the first index to use -- this will be pick-layer index plus the sublayer index

        :param object_count: how many objects need colors

        :param doubled = False: whether to double the array (not sure why you would!) 
        """
        if (layer_index > 255):
            raise ValueError("invalid layer_index: %s"%layer_index)

        # fill the color buffer with a different color for each object
        start_color = layer_index << 24
        color_data = np.arange(start_color, start_color+object_count, dtype=np.uint32)

        ## fix me: why would you want it doubled?
        if doubled:
            color_data = np.c_[color_data, color_data].reshape(-1)

        self.vbo_colors = gl_vbo.VBO( color_data.view(dtype=np.uint8) )

        gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
        self.vbo_colors.bind()
        gl.glColorPointer(self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated

    def unbind_picker_colors(self):
        self.vbo_colors.unbind()
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated

    def get_polygon_picker_colors(self, layer_index, object_count):
        start_color = (layer_index + FILL_PICKER_OFFSET) << 24
        active_colors = np.arange(start_color, start_color + object_count, dtype=np.uint32)
        return active_colors

    def get_object_at_mouse_position(self, screen_point):
        """

            returns ( layer_index, object_index ) or None

        """

        self.bind_frame_buffer()
        if screen_point[1] == 0:
            # work around an odd bug in which the color is not read correctly from the first row of
            # pixels in the window (y coordinate 0)
            color_string = None
        else:
            color_string = gl.glReadPixels(x=screen_point[0],
                                           # glReadPixels() expects coordinates with a lower-left origin
                                           y=rect.height(self.screen_rect) - screen_point[1],
                                           width=1,
                                           height=1,
                                           format=gl.GL_RGBA,
                                           type=gl.GL_UNSIGNED_BYTE,
                                           outputType=str,
                                           )

        if (color_string is not None) and (color_string[0: 3] != self.BACKGROUND_COLOR_STRING):

            full_string = gl.glReadPixels(x=0,
                                          # glReadPixels() expects coordinates with a lower-left origin
                                          y=0,
                                          width=rect.width(self.screen_rect),
                                          height=rect.height(self.screen_rect),
                                          format=gl.GL_RGBA,
                                          type=gl.GL_UNSIGNED_BYTE,  # gl.GL_UNSIGNED_INT_8_8_8_8
                                          outputType=str,
                                          )

        self.unbind_frame_buffer()

        if color_string is None or color_string[0: 3] == self.BACKGROUND_COLOR_STRING:
            return None

        # im = Image.fromstring( "RGBA", ( rect.width( self.screen_rect ), rect.height( self.screen_rect ) ), full_string )
        # im.save( "x.png" )
        # print color_string.__repr__()
        if (sys.byteorder == "big"):
            # most-significant byte first
            b = np.frombuffer(color_string, dtype=np.uint8)
            layer_index = b[0]
            object_index = (b[1] << 16) + (b[2] << 8) + b[3]
        else:
            # least-significant byte first
            b = np.frombuffer(color_string, dtype=np.uint8)
            layer_index = b[3]
            object_index = (b[2] << 16) + (b[1] << 8) + b[0]

        return (layer_index, object_index)
    
    @staticmethod
    def is_ugrid_point(obj):
        (layer_index, type, subtype, object_index) = Picker.parse_clickable_object(obj)
        #
        return type == POINTS_PICKER_OFFSET
    
    @staticmethod
    def is_ugrid_point_type(type):
        return type == POINTS_PICKER_OFFSET

    @staticmethod
    def is_ugrid_line(obj):
        (layer_index, type, subtype, object_index) = Picker.parse_clickable_object(obj)
        #
        return type == LINES_PICKER_OFFSET

    @staticmethod
    def is_polygon_fill(obj):
        (layer_index, type, subtype, object_index) = Picker.parse_clickable_object(obj)
        #
        return type == FILL_PICKER_OFFSET

    @staticmethod
    def parse_clickable_object(o):
        if (o is None):
            return (None, None, None, None)

        # see Layer.py for layer types
        # see Point_and_line_set_renderer.py and Polygon_set_renderer.py for subtypes
        ## fixme: OMG! I can't believe how hard-coded this is!!!
        (layer_pick_index, object_index) = o
        type = layer_pick_index % MAX_PICKER_OFFSET
        subtype = None
        layer_pick_index = layer_pick_index // MAX_PICKER_OFFSET
        # print str( obj ) + "," + str( ( layer_index, type, subtype ) )
        #
        return (layer_pick_index, type, subtype, object_index)

"""
        if self.frame_buffer is None:
            gl.glClear( gl.GL_COLOR_BUFFER_BIT )
        

        self.window.SetCursor( wx.StockCursor( wx.CURSOR_HAND ) )
        

        # If the control/command or shift key is down, then add to the
        # selection rather than replacing it (see mouse_released() below).
        # This allows multi-selection.
        if event.CmdDown() or event.ShiftDown():
            return
        

        self.window.Bind( wx.EVT_MOTION, self.mouse_moved )
        self.window.Bind( wx.EVT_LEFT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_LEFT_UP, self.mouse_released )
        self.window.Bind( wx.EVT_RIGHT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_RIGHT_UP, self.mouse_right_released )
        self.window.Bind( wx.EVT_LEFT_DCLICK, self.mouse_double_clicked )
        self.window.Bind( wx.EVT_LEAVE_WINDOW, self.mouse_left )
"""
