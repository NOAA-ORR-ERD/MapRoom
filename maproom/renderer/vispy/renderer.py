import os
import sys
import wx
import numpy as np

from vispy import app, gloo
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import ortho, translate, rotate
from vispy.geometry import create_cube

from omnimon import get_image_path

import maproom.library.rect as rect
import Picker
import Font_extents

vertex = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec4 u_color;

attribute vec3 position;
attribute vec4 color;

varying vec4 v_color;
void main()
{
    v_color = u_color * color;
    gl_Position = u_projection * u_view * u_model * vec4(position,1.0);
}
"""

vertex_type = [
    ('position', np.float32, 3),
    ('color', np.float32, 4)]

xy_depth_type = [
    ('xy', np.float32, 2),
    ('depth', np.float32, 1),
    ('color', np.float32, 4)]

fragment = """
varying vec4 v_color;
void main()
{
    //gl_FragColor = v_color;
    gl_FragColor = vec4(0.5, 0.5, 1.0, 1.0);
}
"""

class VispyRenderer():

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
        
#        # Build cube data
#        V, I, O = create_cube()
#        # Each item in the vertex data V is a tuple of lists, e.g.:
#        #
#        # ([1.0, 1.0, 1.0], [0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0])
#        #
#        # where each list corresponds to one of the attributes in the vertex
#        # shader.  (See vispy.geometry.create_cube).  To experiment with this,
#        # I've removed the 2nd tuple (the texcoord) and changed the vertex
#        # shader from vispy5.  We'll see if this works. UPDATE: yes, it does!
#        v2 = np.zeros(V.shape[0], vertex_type)
#        v2['position'] = V['position']
#        v2['color'] = V['color']
#        print V[0]
#        print v2[0]
#        self.vertices = VertexBuffer(v2)
#        self.faces = IndexBuffer(I)
#        self.outline = IndexBuffer(O)
        v = np.zeros(0, vertex_type)
        faces = []
        outline = []
        self.vertices = VertexBuffer(v)
        self.faces = IndexBuffer(faces)
        self.outline = IndexBuffer(outline)

        # Build program
        # --------------------------------------
        self.program = Program(vertex, fragment)
        self.program.bind(self.vertices)

    def prepare_to_render_projected_objects(self):
        print "prepare_to_render_projected_objects"
#        gloo.set_state(texture_2d=False)
        self.program['u_model'] = self.canvas.model_matrix
        print self.canvas.model_matrix
        self.program['u_view'] = self.canvas.view_matrix
        print self.canvas.view_matrix
        self.program['u_projection'] = self.canvas.projection_matrix
        print self.canvas.projection_matrix

    def prepare_to_render_screen_objects(self):
#        gl.glEnable(gl.GL_TEXTURE_2D)
#        gl.glBindTexture(gl.GL_TEXTURE_2D, self.font_texture)

#        # set up an orthogonal projection
#        gl.glMatrixMode(gl.GL_PROJECTION)
#        gl.glLoadIdentity()
#        gl.glOrtho(0, self.s_w, 0, self.s_h, -1, 1)
#        gl.glMatrixMode(gl.GL_MODELVIEW)
#        gl.glLoadIdentity()
        pass
    
    def set_points(self, xy, depths):
        v = np.zeros(xy.shape[0], dtype=xy_depth_type)
        print v
        v['xy'] = xy
        v['depth'] = depths
        print v
        self.vertices.set_data(v)
    
    def set_lines(self, indexes, dtype):
        print indexes
        self.outline.set_data(indexes[['point1', 'point2']].view(np.uint32))
#        self.outline.set_data([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6])
    
    def draw_points_and_lines(self, layer,
                              layer_index_base,
                              pick_mode,
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
        pick_mode = True if we are drawing to the off-screen pick buffer
        """
        #log.debug("in Point_and_line_set_renderer, layer_index_base:%s"%layer_index_base)
        if draw_line_segments:
            print "VISPY: calling render_lines"
            self.draw_lines(layer_index_base, pick_mode, point_size, line_width,
                            selected_line_segment_indexes, flagged_line_segment_indexes)

#        if draw_points:
#            self.draw_points(layer_index_base, pick_mode, point_size,
#                               selected_point_indexes, flagged_point_indexes)

    def draw_lines(self,
                   layer_index_base,
                   pick_mode,
                   point_size,
                   line_width,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        # Filled cube
        print "draw lines"
#        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=True,
#                       line_width=line_width)
        gl.glLineWidth(line_width)
#        self.program['u_color'] = 1, 1, 1, 1
#        self.program.draw('triangles', self.faces)
#
#        gloo.set_state(blend=False, depth_test=True, polygon_offset_fill=False)
        self.program['u_color'] = 1, 1, 0, 1
        self.program.draw('lines', self.outline)
        gloo.set_state(depth_mask=True)

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
                    pick_mode,
                    point_size,
                    selected_point_indexes=[],
                    flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        #log.debug("in Point_and_line_set_renderer.render_points, layer_index_base:%s, pick_mode:%s"%(layer_index_base, pick_mode) )

        if (self.vbo_point_xys is not None and len(self.vbo_point_xys) > 0):
            if (not pick_mode and len(flagged_point_indexes) != 0):
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

            if (not pick_mode):
                # To make the points stand out better, especially when rendered on top
                # of line segments, draw translucent white rings under them.
                gl.glPointSize(point_size + 4)
                gl.glColor(1, 1, 1, 0.75)
                gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
                gl.glColor(1, 1, 1, 1)

            if (pick_mode):
                self.vispy.picker.bind_picker_colors(layer_index_base + POINTS_SUB_LAYER_PICKER_OFFSET,
                                                    len(self.vbo_point_xys.data))
                gl.glPointSize(point_size + 8)
            else:
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_point_colors.bind()
                gl.glColorPointer(self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glPointSize(point_size)

            gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            if (pick_mode):
                self.vispy.picker.unbind_picker_colors()
            else:
                self.vbo_point_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated

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
        # flip y to treat point as normal screen coordinates
        point_a = (point_a[0], rect.height(self.screen_rect) - point_a[1])
        point_b = (point_b[0], rect.height(self.screen_rect) - point_b[1])

#        gl.glDisable(gl.GL_TEXTURE_2D)
#        gl.glDisable(gl.GL_LINE_SMOOTH)
#        gl.glLineWidth(width)
#        # don't let texture colors override the line color
#        gl.glColor(red, green, blue, alpha)
#        gl.glLineStipple(stipple_factor, stipple_pattern)
#        gl.glEnable(gl.GL_LINE_STIPPLE)
#        gl.glBegin(gl.GL_LINE_STRIP)
#        gl.glVertex(point_a[0], point_a[1], 0)
#        gl.glVertex(point_b[0], point_b[1], 0)
#        gl.glEnd()
#        gl.glDisable(gl.GL_LINE_STIPPLE)
#        gl.glEnable(gl.GL_LINE_SMOOTH)
#        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)

    def draw_screen_box(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0, width=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(self.screen_rect) - r[0][1]),
             (r[1][0], rect.height(self.screen_rect) - r[1][1]))

#        gl.glDisable(gl.GL_TEXTURE_2D)
#        gl.glDisable(gl.GL_LINE_SMOOTH)
#        gl.glLineWidth(width)
#        # don't let texture colors override the line color
#        gl.glColor(red, green, blue, alpha)
#        gl.glLineStipple(stipple_factor, stipple_pattern)
#        gl.glEnable(gl.GL_LINE_STIPPLE)
#        gl.glBegin(gl.GL_LINE_LOOP)
#        gl.glVertex(r[0][0], r[0][1], 0)
#        gl.glVertex(r[1][0], r[0][1], 0)
#        gl.glVertex(r[1][0], r[1][1], 0)
#        gl.glVertex(r[0][0], r[1][1], 0)
#        gl.glEnd()
#        gl.glDisable(gl.GL_LINE_STIPPLE)
#        gl.glEnable(gl.GL_LINE_SMOOTH)
#        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)

    def draw_screen_rect(self, r, red=0.0, green=0.0, blue=0.0, alpha=1.0):
        # flip y to treat rect as normal screen coordinates
        r = ((r[0][0], rect.height(self.screen_rect) - r[0][1]),
             (r[1][0], rect.height(self.screen_rect) - r[1][1]))

#        gl.glDisable(gl.GL_TEXTURE_2D)
#        # don't let texture colors override the fill color
#        gl.glColor(red, green, blue, alpha)
#        gl.glBegin(gl.GL_QUADS)
#        gl.glVertex(r[0][0], r[0][1], 0)
#        gl.glVertex(r[1][0], r[0][1], 0)
#        gl.glVertex(r[1][0], r[1][1], 0)
#        gl.glVertex(r[0][0], r[1][1], 0)
#        gl.glEnd()
#        gl.glColor(1.0, 1.0, 1.0, 1.0)
#        gl.glEnable(gl.GL_TEXTURE_2D)

    def draw_screen_string(self, point, s):
        ##fixme: Is this is the right place?
        ##fixme: This should be done with shaders anyway.
        ##fixme:  if not shaders, Cython could help a lot, too

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
        # fixme: -- yes, but if you know how big the arrays are going to be
        #           better to build the array once. 
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

#        vbo_screen_vertexes = gl_vbo.VBO(screen_vertex_raw)
#        vbo_texture_coordinates = gl_vbo.VBO(texcoord_raw)
#
#        gl.glEnable(gl.GL_TEXTURE_2D)
#        gl.glColor(1.0, 1.0, 1.0, 1.0)
#
#        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
#        vbo_screen_vertexes.bind()
#        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated
#
#        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
#        vbo_texture_coordinates.bind()
#        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated
#
#        vertex_count = len(s) * 4
#        gl.glDrawArrays(gl.GL_QUADS, 0, vertex_count)
#
#        vbo_screen_vertexes.unbind()
#        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
#        vbo_texture_coordinates.unbind()
#        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
#
#        gl.glDisable(gl.GL_TEXTURE_2D)
#
#        vbo_screen_vertexes = None
#        vbo_texture_coordinates = None
