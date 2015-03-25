import os
import sys
import wx
import numpy as np

import OpenGL
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import OpenGL.GLU as glu

from peppy2 import get_image_path

import maproom.library.rect as rect
import Picker


class ImmediateModeRenderer():

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

    NUM_COLOR_CHANNELS = 4 #i.e. RGBA

    def __init__(self, canvas, layer):
        self.canvas = canvas
        self.layer = layer
        
        self.vbo_point_xys = None
        self.vbo_point_colors = None
        self.world_line_segment_points = None
        self.vbo_line_segment_point_xys = None
        self.vbo_line_segment_colors = None
        self.vbo_triangle_point_indexes = None
        self.vbo_triangle_point_colors = None

    def prepare_to_render_projected_objects(self):
        c = self.canvas
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glMatrixMode(gl.GL_PROJECTION)  # TODO: deprecated
        gl.glLoadIdentity()

        glu.gluOrtho2D(c.projected_rect[0][0], c.projected_rect[1][0],
                       c.projected_rect[0][1], c.projected_rect[1][1])

        gl.glMatrixMode(gl.GL_MODELVIEW)  # TODO: deprecated
        gl.glLoadIdentity()

    def prepare_to_render_screen_objects(self):
        c = self.canvas
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, c.font_texture)

        # set up an orthogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, c.s_w, 0, c.s_h, -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
    
    def set_points(self, xy, depths, color):
        if self.vbo_point_xys is None:
            storage = np.zeros((len(xy), 2), dtype=np.float32)
            self.vbo_point_xys = gl_vbo.VBO(storage)
        self.vbo_point_colors = gl_vbo.VBO(color)
        self.vbo_point_xys[: np.alen(xy)] = xy
    
    def set_lines(self, xy, indexes, color):
        # OpenGL doesn't allow a given vertex to have multiple colors
        # simultaneously. So vbo_line_segment_point_xys is needed to color each
        # line segment individually.
        self.world_line_segment_points = xy[indexes.reshape(-1)].astype(np.float32).reshape(-1, 2)  # .view( self.SIMPLE_POINT_DTYPE ).copy()
        if self.vbo_line_segment_point_xys is None:
            storage = np.zeros((len(self.world_line_segment_points), 2), dtype=np.float32)
            self.vbo_line_segment_point_xys = gl_vbo.VBO(storage)
        self.vbo_line_segment_point_xys[: np.alen(self.world_line_segment_points)] = self.world_line_segment_points
        if (color is not None):
            # double the colors since each segment has two vertexes
            segment_colors = np.c_[color, color].reshape(-1)
            self.vbo_line_segment_colors = gl_vbo.VBO(segment_colors.view(dtype=np.uint8))
    
    def draw_points_and_lines(self, layer,
                              layer_index_base,
                              picker,
                              point_size,
                              line_width,
                              draw_points=True,
                              draw_line_segments=True,
                              draw_triangles=True,
                              triangle_line_width=1,
                              selected_point_indexes=[],
                              flagged_point_indexes=[],
                              selected_line_segment_indexes=[],
                              flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        """
        layer_index_base = the base number of this layer renderer for pick buffer purposes
        picker = True if we are drawing to the off-screen pick buffer
        """
        #log.debug("in Point_and_line_set_renderer, layer_index_base:%s"%layer_index_base)
        if draw_line_segments:
            self.draw_lines(layer_index_base, picker, point_size, line_width,
                            selected_line_segment_indexes, flagged_line_segment_indexes)

        if draw_points:
            self.draw_points(layer_index_base, picker, point_size,
                               selected_point_indexes, flagged_point_indexes)

    def draw_lines(self,
                   layer_index_base,
                   picker,
                   point_size,
                   line_width,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        if (self.vbo_line_segment_point_xys is not None and len(self.vbo_line_segment_point_xys.data) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if (picker is None):
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_line_segment_colors.bind()
                gl.glColorPointer(self.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glLineWidth(line_width)
            else:
                picker.bind_picker_colors_for_lines(layer_index_base,
                                          len(self.world_line_segment_points))
                gl.glLineWidth(6)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

            gl.glDrawArrays(gl.GL_LINES, 0, np.alen(self.vbo_line_segment_point_xys.data))

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if (picker is None):
                self.vbo_line_segment_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            else:
                picker.unbind_picker_colors()
            self.vbo_line_segment_point_xys.unbind()

            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def draw_selected_lines(self, line_width, selected_line_segment_indexes=[]):
        if (len(selected_line_segment_indexes) != 0):
            gl.glLineWidth(line_width + 10)
            gl.glColor(1, 0.6, 0, 0.75)
            gl.glBegin(gl.GL_LINES)
            for i in selected_line_segment_indexes:
                gl.glVertex(self.vbo_line_segment_point_xys.data[i * 2, 0], self.vbo_line_segment_point_xys.data[i * 2, 1], 0)
                gl.glVertex(self.vbo_line_segment_point_xys.data[i * 2 + 1, 0], self.vbo_line_segment_point_xys.data[i * 2 + 1, 1], 0)
            gl.glEnd()
            gl.glColor(1, 1, 1, 1)

    def draw_points(self,
                    layer_index_base,
                    picker,
                    point_size,
                    selected_point_indexes=[],
                    flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        #log.debug("in Point_and_line_set_renderer.render_points, layer_index_base:%s, picker:%s"%(layer_index_base, picker) )

        if (self.vbo_point_xys is not None and len(self.vbo_point_xys) > 0):
            if (picker is None and len(flagged_point_indexes) != 0):
                gl.glPointSize(point_size + 15)
                gl.glColor(0.2, 0, 1, 0.75)
                gl.glBegin(gl.GL_POINTS)
                for i in flagged_point_indexes:
                    gl.glVertex(self.vbo_point_xys.data[i, 0], self.vbo_point_xys.data[i, 1], 0)
                gl.glEnd()
                gl.glColor(1, 1, 1, 1)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if (picker is None):
                # To make the points stand out better, especially when rendered on top
                # of line segments, draw translucent white rings under them.
                gl.glPointSize(point_size + 4)
                gl.glColor(1, 1, 1, 0.75)
                gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
                gl.glColor(1, 1, 1, 1)
                
                # Now set actual color
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_point_colors.bind()
                gl.glColorPointer(self.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glPointSize(point_size)
            else:
                picker.bind_picker_colors_for_points(layer_index_base,
                                                     len(self.vbo_point_xys.data))
                gl.glPointSize(point_size + 8)

            gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            if (picker is None):
                self.vbo_point_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            else:
                picker.unbind_picker_colors()

            self.vbo_point_xys.unbind()

    def draw_selected_points(self, point_size, selected_point_indexes=[]):
        if (len(selected_point_indexes) != 0):
            gl.glPointSize(point_size + 10)
            gl.glColor(1, 0.6, 0, 0.75)
            gl.glBegin(gl.GL_POINTS)
            for i in selected_point_indexes:
                gl.glVertex(self.vbo_point_xys.data[i, 0], self.vbo_point_xys.data[i, 1], 0)
            gl.glEnd()
            gl.glColor(1, 1, 1, 1)

    def draw_screen_line(self, point_a, point_b, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        c = self.canvas
        # flip y to treat point as normal screen coordinates
        point_a = (point_a[0], rect.height(c.screen_rect) - point_a[1])
        point_b = (point_b[0], rect.height(c.screen_rect) - point_b[1])

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
        c = self.canvas
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(c.screen_rect) - r[0][1]),
             (r[1][0], rect.height(c.screen_rect) - r[1][1]))

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
        c = self.canvas
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(c.screen_rect) - r[0][1]),
             (r[1][0], rect.height(c.screen_rect) - r[1][1]))

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

    def get_drawn_string_dimensions(self, text):
        c = self.canvas
        width = 0
        height = 0
        for char in text:
            if char not in c.font_extents:
                char = "?"

            width += c.font_extents[char][2]
            height = max(height, c.font_extents[char][3])

        return (width, height)

    def draw_screen_string(self, point, text):
        ##fixme: Is this is the right place?
        ##fixme: This should be done with shaders anyway.
        ##fixme:  if not shaders, Cython could help a lot, too

        c = self.canvas
        # flip y to treat point as normal screen coordinates
        point = (point[0], rect.height(c.screen_rect) - point[1])

        screen_vertex_data = np.zeros(
            (len(text), ),
            dtype=self.QUAD_VERTEX_DTYPE,
        ).view(np.recarray)

        texcoord_data = np.zeros(
            (len(text), ),
            dtype=self.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        
        # Create views into recarray data.  PyOpenGL 3.1 with PyOpenGL-
        # accelerate doesn't allow VBO data in recarray form
        screen_vertex_raw = screen_vertex_data.view(dtype=np.float32).reshape(-1,8)
        texcoord_raw = texcoord_data.view(dtype=np.float32).reshape(-1,8)

        # these are used just because it seems to be the fastest way to full numpy arrays
        # fixme: -- yes, but if you know how big the arrays are going to be
        #           better to build the array once. 
        screen_vertex_accumulators = [[], [], [], [], [], [], [], []]
        tex_coord_accumulators = [[], [], [], [], [], [], [], []]

        texture_width = float(c.font_texture_size[0])
        texture_height = float(c.font_texture_size[1])
        x_offset = 0

        for char in text:
            if char not in c.font_extents:
                char = "?"

            x = c.font_extents[char][0]
            y = c.font_extents[char][1]
            w = c.font_extents[char][2]
            h = c.font_extents[char][3]

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
        gl.glBindTexture(gl.GL_TEXTURE_2D, c.font_texture)
        gl.glColor(1.0, 1.0, 1.0, 1.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        vbo_screen_vertexes.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        vbo_texture_coordinates.bind()
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        vertex_count = len(text) * 4
        gl.glDrawArrays(gl.GL_QUADS, 0, vertex_count)

        vbo_screen_vertexes.unbind()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        vbo_texture_coordinates.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

        vbo_screen_vertexes = None
        vbo_texture_coordinates = None
