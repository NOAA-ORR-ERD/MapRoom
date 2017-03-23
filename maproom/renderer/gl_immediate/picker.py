import sys
import bisect

import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

import OpenGL.GL.EXT.framebuffer_object as gl_fbo

import maproom.library.rect as rect

from ..gl.color import color_floats_to_int

import logging
log = logging.getLogger(__name__)

POINTS_PICKER_OFFSET = 0
LINES_PICKER_OFFSET = 1
FILL_PICKER_OFFSET = 2


class Picker(object):
    """
    An off-screen buffer for detecting what object the mouse is over.
    """

    CHANNELS = 4  # i.e., RGBA
    BACKGROUND_COLOR = color_floats_to_int(1, 1, 1, 1)
    BACKGROUND_COLOR_STRING = chr(255) * 3
    frame_buffer = None  # the container frame buffer object
    render_buffer = None  # the render buffer within the frame buffer for offscreen drawing
    screen_rect = rect.EMPTY_RECT
    vbo_colors = None

    is_active = True

    def __init__(self):
        # self.colored_objects = {} # renderer -> ( index, original color )
        gl_fbo.glInitFramebufferObjectEXT()
        self.picker_map = None

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
        self.picker_map = []
        self.picker_blocks = []
        self.picker_block_start = 1

    def done_rendering(self):
        self.unbind_frame_buffer()

    def bind_frame_buffer(self):
        gl.glDrawBuffer(gl.GL_NONE)
        gl_fbo.glBindFramebufferEXT(gl_fbo.GL_FRAMEBUFFER_EXT, self.frame_buffer)

    def unbind_frame_buffer(self):
        gl_fbo.glBindFramebufferEXT(gl_fbo.GL_FRAMEBUFFER_EXT, 0)
        gl.glDrawBuffer(gl.GL_BACK)

    def render_picker_to_screen(self):
        if self.frame_buffer is None:
            return
        gl_fbo.glBindFramebufferEXT(gl.GL_READ_FRAMEBUFFER, self.frame_buffer)
        gl_fbo.glBindFramebufferEXT(gl.GL_DRAW_FRAMEBUFFER, 0)
        w = rect.width(self.screen_rect)
        h = rect.height(self.screen_rect)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR)
        gl_fbo.glBindFramebufferEXT(gl.GL_READ_FRAMEBUFFER, 0)

    def bind_picker_colors_for_lines(self, layer, object_count):
        self.bind_picker_colors(layer, LINES_PICKER_OFFSET, object_count, True)

    def bind_picker_colors_for_points(self, layer, object_count):
        self.bind_picker_colors(layer, POINTS_PICKER_OFFSET, object_count, False)

    def bind_picker_colors(self, layer, object_type, object_count, doubled=False):
        """
        bind the colors in the OpenGL context (right word?)

        for a bunch of object in a layer

        :param layer: the first index to use -- this will be pick-layer index plus the sublayer index

        :param object_type: flag indicating the type of object

        :param object_count: how many objects need colors

        :param doubled = False: whether to double the array (used for drawing line segments) 
        """
        # Get range of picker colors for this layer and object type
        color_data = self.get_next_color_block(layer, object_type, object_count)

        # NOTE: the color data array is doubled for lines because there are two
        # copies of each endpoint in the line data. See set_lines in
        # renderer/gl_immediate/renderer.py for more info
        if doubled:
            color_data = np.c_[color_data, color_data].reshape(-1)

        self.vbo_colors = gl_vbo.VBO( color_data.view(dtype=np.uint8) )

        gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
        self.vbo_colors.bind()
        gl.glColorPointer(self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated

    def unbind_picker_colors(self):
        self.vbo_colors.unbind()
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated

    def get_next_color_block(self, layer, object_type, object_count):
        start_range = self.picker_block_start
        end_range = self.picker_block_start + object_count
        self.picker_block_start = end_range
        color_data = np.arange(start_range, end_range, dtype=np.uint32)
        self.picker_map.append((layer, object_type, start_range))
        self.picker_blocks.append(start_range)
        log.debug("Generating color block for [%s] type=%d, #%d: %d-%d" % (layer, object_type, object_count, start_range, end_range))
        return color_data

    def get_polygon_picker_colors(self, layer, object_count):
        active_colors = self.get_next_color_block(layer, FILL_PICKER_OFFSET, object_count)
        return active_colors

    def get_object_at_mouse_position(self, screen_point):
        """

            returns ( layer, object_index ) or None

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

        if (sys.byteorder == "big"):
            picked_color = np.frombuffer(color_string, dtype='>i4')
        else:
            picked_color = np.frombuffer(color_string, dtype='<i4')

        # bisect returns where to insert a value, but we want the index right
        # before that since that points to the starting color
        index = bisect.bisect(self.picker_blocks, picked_color) - 1
        if index < 0:
            return None
        layer, object_type, start_color = self.picker_map[index]
        object_index = int(picked_color - start_color)

        return (layer, object_type, object_index)

    @staticmethod
    def is_ugrid_point(obj):
        (layer, object_type, object_index) = obj
        return object_type == POINTS_PICKER_OFFSET

    @staticmethod
    def is_ugrid_point_type(object_type):
        return object_type == POINTS_PICKER_OFFSET

    @staticmethod
    def is_ugrid_line(obj):
        (layer, object_type, object_index) = obj
        return object_type == LINES_PICKER_OFFSET

    @staticmethod
    def is_ugrid_line_type(object_type):
        return object_type == LINES_PICKER_OFFSET

    @staticmethod
    def is_polygon_fill(obj):
        (layer, object_type, object_index) = obj
        return type == FILL_PICKER_OFFSET

    @staticmethod
    def is_polygon_fill_type(object_type):
        return object_type == FILL_PICKER_OFFSET
