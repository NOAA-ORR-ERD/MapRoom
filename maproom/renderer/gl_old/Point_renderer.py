"""
Renderer for just points

Used for GNOME particles for now

    This was copied from the Point_and_line_set_renderer, with extra stuff removed.
"""


import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
from.data_types import POINT_COORD_VIEW_DTYPE

import logging
log = logging.getLogger(__name__)


POINTS_SUB_LAYER_PICKER_OFFSET = 0
LINES_SUB_LAYER_PICKER_OFFSET = 1


class Point_renderer(object):
    """
    OpenGL renderer for simple sets of points: 
    no lines, no selections, no anything but points.
    """

    # SIMPLE_POINT_DTYPE = np.dtype(
    #     [ ( "x", np.float64 ), ( "y", np.float64 ) ],
    # )

    def __init__(self,
                 opengl_renderer,
                 points,
                 point_colors,
                 projection ):
        """
        initilize a Point_renderer

        :param opengl_renderer: the opgl render object -- holds all the gl context state

        :param point: coordinates of the points
        :type point: 2 x np.float64, i.e., "2f4"

        :param point_colors: colors of the points
        :type point_colors: np array of np.uint8, four per point (RGBA)

        :param projection: current projection to use
        :type projection: a pyproj-style projection callable object, such that
                            projection( world_x, world_y ) = ( projected_x, projected_y )

        """
        self.oglr = opengl_renderer

        if ( points is None or len(points) == 0 ):
            self.vbo_point_xys = None
            self.vbo_point_colors = None
        else:
            self.vbo_point_xys = gl_vbo.VBO( np.zeros( points.shape, dtype=POINT_COORD_VIEW_DTYPE ) )
            self.vbo_point_colors = gl_vbo.VBO( point_colors )
            self.reproject(points, projection )

    def reproject(self, points, projection ):
        if (points is None or len(points) == 0):
            self.vbo_point_xys = None
            self.vbo_point_colors = None
            return

        projected_point_data = self.vbo_point_xys.data
        ## fixme -- why each axis individually --  proj takes an Nx2 array
        projected_point_data[:, 0], projected_point_data[:, 1] = projection(points[:, 0].astype(POINT_COORD_VIEW_DTYPE), points[:, 1].astype(POINT_COORD_VIEW_DTYPE))

        self.vbo_point_xys[:] = projected_point_data

    def render(self,
               layer_index_base,
               pick_mode,
               point_size,
               draw_points=True,
               ):
        """
        layer_index_base = the base number of this layer renderer for pick buffer purposes
        pick_mode = True if we are drawing to the off-screen pick buffer
        """
        if draw_points:
            self.render_points(layer_index_base,
                               pick_mode,
                               point_size,
                               )

    def render_points(self,
                      layer_index_base,
                      pick_mode,
                      point_size,
                      ):

        if (self.vbo_point_xys is not None and len(self.vbo_point_xys) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if (pick_mode):
                self.oglr.picker.bind_picker_colors(layer_index_base + POINTS_SUB_LAYER_PICKER_OFFSET,
                                                    len(self.vbo_point_xys.data))
                gl.glPointSize(point_size + 8)
            else:
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_point_colors.bind()
                gl.glColorPointer(self.oglr.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glPointSize(point_size)

            gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            if (pick_mode):
                self.oglr.picker.unbind_picker_colors()
            else:
                self.vbo_point_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated

            self.vbo_point_xys.unbind()

    def destroy(self):
        self.world_line_segment_points = None
        self.vbo_point_xys = None
        self.vbo_point_colors = None
        self.vbo_line_segment_point_xys = None
        self.vbo_line_segment_colors = None
        self.vbo_triangle_point_indexes = None
        self.vbo_triangle_point_colors = None
