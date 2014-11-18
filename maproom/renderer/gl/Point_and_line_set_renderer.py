import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

POINTS_SUB_LAYER_PICKER_OFFSET = 0
LINES_SUB_LAYER_PICKER_OFFSET = 1

import logging
log = logging.getLogger(__name__)

class Point_and_line_set_renderer:

    CHANNELS = 4  # i.e., RBGA

    """
    SIMPLE_POINT_DTYPE = np.dtype(
        [ ( "x", np.float64 ), ( "y", np.float64 ) ],
    )
    """
    def __init__(self,
                 opengl_renderer,
                 points,
                 point_colors,
                 line_segment_indexes,
                 line_segment_colors,
                 triangle_point_indexes,
                 triangle_point_colors,
                 projection):
        """
            points = 2 x np.float64, i.e., "2f4"
            point_colors = np array of np.uint32, one per point
            line_segment_indexes = 2 x np.uint32, i.e., "2u4"
            line_segment_colors = np array of np.uint32, one per line segment
            projection = a pyproj-style projection callable object, such that
                            projection( world_x, world_y ) = ( projected_x, projected_y )
        """

        self.oglr = opengl_renderer
        # self.world_points = points.copy() # .view( self.SIMPLE_POINT_DTYPE ) # .view( np.recarray )
        self.vbo_point_xys = None
        self.vbo_point_colors = None
        self.world_line_segment_points = None
        self.vbo_line_segment_point_xys = None
        self.vbo_line_segment_colors = None
        self.vbo_triangle_point_indexes = None
        self.vbo_triangle_point_colors = None

        if (points == None or len(points) == 0):
            return

        projected_point_data = np.zeros(
            (len(points), 2),
            dtype=np.float32
        )

        self.vbo_point_xys = gl_vbo.VBO(
            projected_point_data
        )
        self.vbo_point_colors = gl_vbo.VBO(
            point_colors
        )

        if line_segment_indexes is not None:
            self.build_line_segment_buffers(points, line_segment_indexes, line_segment_colors)

        if triangle_point_indexes is not None:
            self.build_triangle_buffers(points, triangle_point_indexes, triangle_point_colors)

        self.reproject(points, projection )

    def build_line_segment_buffers(self, points, line_segment_indexes, line_segment_colors):
        if (line_segment_indexes == None or np.alen(line_segment_indexes) == 0):
            self.world_line_segment_points = None
            self.vbo_line_segment_point_xys = None
            self.vbo_line_segment_colors = None
            #
            return

        # OpenGL doesn't allow a given vertex to have multiple colors
        # simultaneously. So vbo_line_segment_point_xys is needed to color each
        # line segment individually.
        self.world_line_segment_points = points[
            line_segment_indexes.reshape(-1)
        ].astype(np.float32).reshape(-1, 2)  # .view( self.SIMPLE_POINT_DTYPE ).copy()
        projected_line_segment_point_data = np.zeros(
            (len(self.world_line_segment_points), 2),
            dtype=np.float32
        )
        self.vbo_line_segment_point_xys = gl_vbo.VBO(
            projected_line_segment_point_data
        )
        if (line_segment_colors != None):
            # double the colors since each segment has two vertexes
            segment_colors = np.c_[line_segment_colors, line_segment_colors].reshape(-1)
            self.vbo_line_segment_colors = gl_vbo.VBO(
                segment_colors.view(dtype=np.uint8)
            )

    def build_triangle_buffers(self, points, triangle_point_indexes, triangle_point_colors):
        """
        projected_triangle_point_data = np.zeros(
            ( len( points ), 2 ),
            dtype = np.float64
        )
        self.vbo_triangle_point_xys = gl_vbo.VBO(
            projected_triangle_point_data
        )
        """
        self.vbo_triangle_point_indexes = gl_vbo.VBO(
            triangle_point_indexes.view(np.uint32),
            usage="GL_STATIC_DRAW",
            target="GL_ELEMENT_ARRAY_BUFFER"
        )
        if (triangle_point_colors != None):
            self.vbo_triangle_point_colors = gl_vbo.VBO(triangle_point_colors)

    def reproject(self, points, projection ):
        if (points == None or len(points) == 0):
            self.vbo_point_xys = None
            self.vbo_point_colors = None
            self.world_line_segment_points = None
            self.vbo_line_segment_point_xys = None
            self.vbo_line_segment_colors = None

            return

        ##fixme -- this could probably be optimized -- proj can take a Nx2 array
        projected_point_data = self.vbo_point_xys.data
        projected_point_data[:, 0], projected_point_data[:, 1] = projection(points[:, 0].astype(np.float32), points[:, 1].astype(np.float32))
        self.vbo_point_xys[: np.alen(projected_point_data)] = projected_point_data

        if (self.vbo_line_segment_point_xys != None and len(self.vbo_line_segment_point_xys.data) > 0):
            projected_line_segment_point_data = self.vbo_line_segment_point_xys.data
            if (projection.srs.find("+proj=longlat") != -1):
                projected_line_segment_point_data[:, 0] = self.world_line_segment_points[:, 0]
                projected_line_segment_point_data[:, 1] = self.world_line_segment_points[:, 1]
            else:
                projected_line_segment_point_data[:, 0], projected_line_segment_point_data[:, 1] = projection(self.world_line_segment_points[:, 0], self.world_line_segment_points[:, 1])
            self.vbo_line_segment_point_xys[: np.alen(projected_line_segment_point_data)] = projected_line_segment_point_data

    def render(self,
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
            self.render_lines(layer_index_base, pick_mode, point_size, line_width,
                              selected_line_segment_indexes, flagged_line_segment_indexes)

        if draw_points:
            self.render_points(layer_index_base, pick_mode, point_size,
                               selected_point_indexes, flagged_point_indexes)

        if draw_triangles and self.vbo_triangle_point_indexes is not None:
            self.render_triangles(pick_mode, triangle_line_width)

    def render_selected_line_segments(self, line_width, selected_line_segment_indexes=[]):
        if (self.vbo_line_segment_point_xys != None and len(self.vbo_line_segment_point_xys.data) > 0):
            if (len(selected_line_segment_indexes) != 0):
                gl.glLineWidth(line_width + 10)
                gl.glColor(1, 0.6, 0, 0.75)
                gl.glBegin(gl.GL_LINES)
                for i in selected_line_segment_indexes:
                    gl.glVertex(self.vbo_line_segment_point_xys.data[i * 2, 0], self.vbo_line_segment_point_xys.data[i * 2, 1], 0)
                    gl.glVertex(self.vbo_line_segment_point_xys.data[i * 2 + 1, 0], self.vbo_line_segment_point_xys.data[i * 2 + 1, 1], 0)
                gl.glEnd()
                gl.glColor(1, 1, 1, 1)

    def render_selected_points(self, point_size, selected_point_indexes=[]):
        if (self.vbo_point_xys != None and len(self.vbo_point_xys) > 0):
            if (len(selected_point_indexes) != 0):
                gl.glPointSize(point_size + 10)
                gl.glColor(1, 0.6, 0, 0.75)
                gl.glBegin(gl.GL_POINTS)
                for i in selected_point_indexes:
                    gl.glVertex(self.vbo_point_xys.data[i, 0], self.vbo_point_xys.data[i, 1], 0)
                gl.glEnd()
                gl.glColor(1, 1, 1, 1)

    def render_points(self,
                      layer_index_base,
                      pick_mode,
                      point_size,
                      selected_point_indexes=[],
                      flagged_point_indexes=[]):  # flagged_line_segment_indexes not yet used
        #log.debug("in Point_and_line_set_renderer.render_points, layer_index_base:%s, pick_mode:%s"%(layer_index_base, pick_mode) )

        if (self.vbo_point_xys != None and len(self.vbo_point_xys) > 0):
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
                self.oglr.picker.bind_picker_colors(layer_index_base + POINTS_SUB_LAYER_PICKER_OFFSET,
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
                self.oglr.picker.unbind_picker_colors()
            else:
                self.vbo_point_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated

            self.vbo_point_xys.unbind()

    def render_lines(self,
                     layer_index_base,
                     pick_mode,
                     point_size,
                     line_width,
                     selected_line_segment_indexes=[],
                     flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        # the line segments
        # log.debug("in Point_and_line_set_renderer.render_lines, layer_index_base:%s, pick_mode:%s"%(layer_index_base, pick_mode) )

        if (self.vbo_line_segment_point_xys != None and len(self.vbo_line_segment_point_xys.data) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if (pick_mode):
                self.oglr.picker.bind_picker_colors(layer_index_base + LINES_SUB_LAYER_PICKER_OFFSET,
                                                    len(self.world_line_segment_points),
                                                    True)
                gl.glLineWidth(6)
            else:
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_line_segment_colors.bind()
                gl.glColorPointer(self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glLineWidth(line_width)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

            gl.glDrawArrays(gl.GL_LINES, 0, np.alen(self.vbo_line_segment_point_xys.data))

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if (pick_mode):
                self.oglr.picker.unbind_picker_colors()
            else:
                self.vbo_line_segment_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_point_xys.unbind()

            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)


    def render_triangles(self, pick_mode, line_width):
        if pick_mode:
            return

        # the line segments
        if (self.vbo_triangle_point_indexes != None and len(self.vbo_triangle_point_indexes.data) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            gl.glEnableClientState(gl.GL_INDEX_ARRAY)  # FIXME: deprecated
            self.vbo_triangle_point_indexes.bind()

            gl.glColor(0.5, 0.5, 0.5, 0.75)
            gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            self.vbo_triangle_point_colors.bind()
            gl.glColorPointer(self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDrawElements(gl.GL_TRIANGLES, np.alen(self.vbo_triangle_point_indexes.data) * 3, gl.GL_UNSIGNED_INT, None)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )

            gl.glColor(0.5, 0.5, 0.5, 0.75)
            gl.glLineWidth(line_width)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glDrawElements(gl.GL_TRIANGLES, np.alen(self.vbo_triangle_point_indexes.data) * 3, gl.GL_UNSIGNED_INT, None)

            gl.glColor(1, 1, 1, 1)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            gl.glDisableClientState(gl.GL_INDEX_ARRAY)  # FIXME: deprecated
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
#            self.vbo_triangle_point_colors.unbind()
            self.vbo_triangle_point_indexes.unbind()
            self.vbo_point_xys.unbind()

    def destroy(self):
        self.world_line_segment_points = None
        self.vbo_point_xys = None
        self.vbo_point_colors = None
        self.vbo_line_segment_point_xys = None
        self.vbo_line_segment_colors = None
        self.vbo_triangle_point_indexes = None
        self.vbo_triangle_point_colors = None
