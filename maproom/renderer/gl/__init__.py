import os
import sys
import wx
import numpy as np
import OpenGL
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import OpenGL.GLU as glu
import maproom.library.rect as rect
import Picker
import Font_extents


class Opengl_renderer():

    QUAD_VERTEX_DTYPE = np.dtype(
        [("x_lb", np.float32), ("y_lb", np.float32),
         ("x_lt", np.float32), ("y_lt", np.float32),
         ("x_rt", np.float32), ("y_rt", np.float32),
         ("x_rb", np.float32), ("y_rb", np.float32)]
    )
    TEXTURE_COORDINATE_DTYPE = np.dtype(
        [("u_lb", np.float32), ("v_lb", np.float32),
         ("u_lt", np.float32), ("v_lt", np.float32),
         ("u_rt", np.float32), ("v_rt", np.float32),
         ("u_rb", np.float32), ("v_rb", np.float32)]
    )

    picker = None

    def __init__(self, create_picker=False):
        if gl.glGetInteger( gl.GL_RED_BITS ) != 8 or \
            gl.glGetInteger( gl.GL_GREEN_BITS ) != 8 or \
            gl.glGetInteger( gl.GL_BLUE_BITS ) != 8 or \
                gl.glGetInteger(gl.GL_ALPHA_BITS) != 8:
            raise Exception("Your display must support 32-bit color.")

        if not __debug__ or hasattr(sys, "frozen"):
            OpenGL.ERROR_CHECKING = False
            OpenGL.ERROR_LOGGING = False
        OpenGL.ERROR_ON_COPY = True

        self.set_up_for_regular_rendering()

        gl.glDisable(gl.GL_LIGHTING)
        # Don't cull polygons that are wound the wrong way.
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glClearColor(1.0, 1.0, 1.0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        (self.font_texture, self.font_texture_size) = self.load_font_texture()
        self.font_extents = Font_extents.FONT_EXTENTS

        if (create_picker):
            self.picker = Picker.Picker()

        self.screen_rect = rect.EMPTY_RECT

    def destroy(self):
        if self.font_texture:
            gl.glDeleteTextures(
                np.array([self.font_texture], np.uint32),
            )
            self.font_texture = None
    
    def prepare_to_render(self, projected_rect, screen_rect):
        self.screen_rect = screen_rect
        self.s_w = rect.width(screen_rect)
        self.s_h = rect.height(screen_rect)
        self.projected_rect = projected_rect
        p_w = rect.width(projected_rect)
        p_h = rect.height(projected_rect)

        if (self.s_w <= 0 or self.s_h <= 0 or p_w <= 0 or p_h <= 0):
            return False
        
        gl.glViewport(0, 0, self.s_w, self.s_h)

        gl.glClearColor(1.0, 1.0, 1.0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self.prepare_to_render_projected_objects()
        
        return True

    def prepare_to_render_projected_objects(self):
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glMatrixMode(gl.GL_PROJECTION)  # TODO: deprecated
        gl.glLoadIdentity()

        glu.gluOrtho2D(self.projected_rect[0][0], self.projected_rect[1][0],
                       self.projected_rect[0][1], self.projected_rect[1][1])

        gl.glMatrixMode(gl.GL_MODELVIEW)  # TODO: deprecated
        gl.glLoadIdentity()

    def load_font_texture(self):
        frozen = getattr(sys, 'frozen', False)
        font_path = os.path.join(os.path.dirname(__file__), "font.png")
        if frozen and frozen in ('macosx_app'):
            root = os.environ['RESOURCEPATH']
            zippath, modpath = font_path.split(".zip/")
            font_path = os.path.join(root, modpath)
        image = wx.Image(font_path, wx.BITMAP_TYPE_PNG)
        width = image.GetWidth()
        height = image.GetHeight()
        buffer = np.frombuffer(image.GetDataBuffer(), np.uint8).reshape(
            (width, height, 3),
        )

        # Make an alpha channel that is opaque where the pixels are black
        # and semi-transparent where the pixels are white.
        buffer_with_alpha = np.empty((width, height, 4), np.uint8)
        buffer_with_alpha[:, :, 0: 3] = buffer
        buffer_with_alpha[:, :, 3] = (
            255 - buffer[:, :, 0: 3].sum(axis=2) / 3
        ).clip(180, 255)

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

        return (texture, (width, height))

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
        # gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE )
        gl.glDisable(gl.GL_BLEND)
        # gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

    #
    # the methods below are used to render simple objects one at a time, in screen coordinates

    def prepare_to_render_screen_objects(self):
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.font_texture)

        # set up an orthogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.s_w, 0, self.s_h, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def get_drawn_string_dimensions(self, s):
        width = 0
        height = 0
        for c in s:
            if c not in self.font_extents:
                c = "?"

            width += self.font_extents[c][2]
            height = max(height, self.font_extents[c][3])

        return (width, height)

    def draw_screen_line(self, point_a, point_b, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        # flip y to treat point as normal screen coordinates
        point_a = (point_a[0], rect.height(self.screen_rect) - point_a[1])
        point_b = (point_b[0], rect.height(self.screen_rect) - point_b[1])

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(width)
        # don't let texture colors override the line color
        gl.glColor(red, green, blue, alpha)
        gl.glLineStipple(stipple_factor, stipple_pattern)
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glBegin(gl.GL_LINE_STRIP)
        gl.glVertex(point_a[0], point_a[1], 0)
        gl.glVertex(point_b[0], point_b[1], 0)
        gl.glEnd()
        gl.glDisable(gl.GL_LINE_STIPPLE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)

    def draw_screen_box(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0, width=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(self.screen_rect) - r[0][1]),
             (r[1][0], rect.height(self.screen_rect) - r[1][1]))

        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(width)
        # don't let texture colors override the line color
        gl.glColor(red, green, blue, alpha)
        gl.glLineStipple(stipple_factor, stipple_pattern)
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glBegin(gl.GL_LINE_LOOP)
        gl.glVertex(r[0][0], r[0][1], 0)
        gl.glVertex(r[1][0], r[0][1], 0)
        gl.glVertex(r[1][0], r[1][1], 0)
        gl.glVertex(r[0][0], r[1][1], 0)
        gl.glEnd()
        gl.glDisable(gl.GL_LINE_STIPPLE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)

    def draw_screen_rect(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(self.screen_rect) - r[0][1]),
             (r[1][0], rect.height(self.screen_rect) - r[1][1]))

        gl.glDisable(gl.GL_TEXTURE_2D)
        # don't let texture colors override the fill color
        gl.glColor(red, green, blue, alpha)
        gl.glBegin(gl.GL_QUADS)
        gl.glVertex(r[0][0], r[0][1], 0)
        gl.glVertex(r[1][0], r[0][1], 0)
        gl.glVertex(r[1][0], r[1][1], 0)
        gl.glVertex(r[0][0], r[1][1], 0)
        gl.glEnd()
        gl.glColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(gl.GL_TEXTURE_2D)

    def draw_screen_string(self, point, s):
        # flip y to treat point as normal screen coordinates
        point = (point[0], rect.height(self.screen_rect) - point[1])

        screen_vertex_data = np.zeros(
            (len(s), ),
            dtype=self.QUAD_VERTEX_DTYPE,
        ).view(np.recarray)

        texcoord_data = np.zeros(
            (len(s), ),
            dtype=self.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        
        # Create views into recarray data.  PyOpenGL 3.1 with PyOpenGL-
        # accelerate doesn't allow VBO data in recarray form
        screen_vertex_raw = screen_vertex_data.view(dtype=np.float32).reshape(-1,8)
        texcoord_raw = texcoord_data.view(dtype=np.float32).reshape(-1,8)

        # these are used just because it seems to be the fastest way to full numpy arrays
        screen_vertex_accumulators = [[], [], [], [], [], [], [], []]
        tex_coord_accumulators = [[], [], [], [], [], [], [], []]

        texture_width = float(self.font_texture_size[0])
        texture_height = float(self.font_texture_size[1])
        x_offset = 0

        for c in s:
            if c not in self.font_extents:
                c = "?"

            x = self.font_extents[c][0]
            y = self.font_extents[c][1]
            w = self.font_extents[c][2]
            h = self.font_extents[c][3]

            # again, flip y to treat point as normal screen coordinates
            screen_vertex_accumulators[0].append(point[0] + x_offset)
            screen_vertex_accumulators[1].append(point[1] - h)
            screen_vertex_accumulators[2].append(point[0] + x_offset)
            screen_vertex_accumulators[3].append(point[1])
            screen_vertex_accumulators[4].append(point[0] + w + x_offset)
            screen_vertex_accumulators[5].append(point[1])
            screen_vertex_accumulators[6].append(point[0] + w + x_offset)
            screen_vertex_accumulators[7].append(point[1] - h)
            x_offset += w

            tex_coord_accumulators[0].append(x / texture_width)
            tex_coord_accumulators[1].append((y + h) / texture_height)
            tex_coord_accumulators[2].append(x / texture_width)
            tex_coord_accumulators[3].append(y / texture_height)
            tex_coord_accumulators[4].append((x + w) / texture_width)
            tex_coord_accumulators[5].append(y / texture_height)
            tex_coord_accumulators[6].append((x + w) / texture_width)
            tex_coord_accumulators[7].append((y + h) / texture_height)

        screen_vertex_data.x_lb = screen_vertex_accumulators[0]
        screen_vertex_data.y_lb = screen_vertex_accumulators[1]
        screen_vertex_data.x_lt = screen_vertex_accumulators[2]
        screen_vertex_data.y_lt = screen_vertex_accumulators[3]
        screen_vertex_data.x_rt = screen_vertex_accumulators[4]
        screen_vertex_data.y_rt = screen_vertex_accumulators[5]
        screen_vertex_data.x_rb = screen_vertex_accumulators[6]
        screen_vertex_data.y_rb = screen_vertex_accumulators[7]

        texcoord_data.u_lb = tex_coord_accumulators[0]
        texcoord_data.v_lb = tex_coord_accumulators[1]
        texcoord_data.u_lt = tex_coord_accumulators[2]
        texcoord_data.v_lt = tex_coord_accumulators[3]
        texcoord_data.u_rt = tex_coord_accumulators[4]
        texcoord_data.v_rt = tex_coord_accumulators[5]
        texcoord_data.u_rb = tex_coord_accumulators[6]
        texcoord_data.v_rb = tex_coord_accumulators[7]

        vbo_screen_vertexes = gl_vbo.VBO(screen_vertex_raw)
        vbo_texture_coordinates = gl_vbo.VBO(texcoord_raw)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glColor(1.0, 1.0, 1.0, 1.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        vbo_screen_vertexes.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        vbo_texture_coordinates.bind()
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        vertex_count = len(s) * 4
        gl.glDrawArrays(gl.GL_QUADS, 0, vertex_count)

        vbo_screen_vertexes.unbind()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        vbo_texture_coordinates.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        gl.glDisable(gl.GL_TEXTURE_2D)

        vbo_screen_vertexes = None
        vbo_texture_coordinates = None
