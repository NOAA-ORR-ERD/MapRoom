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
from .. import data_types, int_to_color
from ..gl.textures import ImageTextures
from ..gl.Tessellator import init_vertex_buffers, tessellate
from ..gl.Render import render_buffers_with_colors, render_buffers_with_one_color

from .gldc import GLDC


class ImmediateModeRenderer():
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
        self.image_textures = None
        self.image_projected_rects = []

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
    
    def set_points(self, xy, depths, color, num_points=-1):
        if num_points == -1:
            num_points = np.alen(xy)
        if self.vbo_point_xys is None or np.alen(self.vbo_point_xys.data) != num_points:
            storage = np.zeros((num_points, 2), dtype=np.float32)
            self.vbo_point_xys = gl_vbo.VBO(storage)
        self.vbo_point_colors = gl_vbo.VBO(color)
        self.vbo_point_xys[:num_points] = xy[:num_points]
    
    def set_lines(self, xy, indexes, color):
        # OpenGL doesn't allow a given vertex to have multiple colors
        # simultaneously. So vbo_line_segment_point_xys is needed to color each
        # line segment individually.
        self.world_line_segment_points = xy[indexes.reshape(-1)].astype(np.float32).reshape(-1, 2)  # .view( self.SIMPLE_POINT_DTYPE ).copy()
        if self.vbo_line_segment_point_xys is None or np.alen(self.vbo_line_segment_point_xys.data) != np.alen(self.world_line_segment_points):
            storage = np.zeros((len(self.world_line_segment_points), 2), dtype=np.float32)
            self.vbo_line_segment_point_xys = gl_vbo.VBO(storage)
        self.vbo_line_segment_point_xys[: np.alen(self.world_line_segment_points)] = self.world_line_segment_points
        if (color is not None):
            # double the colors since each segment has two vertexes
            segment_colors = np.c_[color, color].reshape(-1)
            self.vbo_line_segment_colors = gl_vbo.VBO(segment_colors.view(dtype=np.uint8))
    
    def draw_lines(self,
                   layer_index_base,
                   picker,
                   style,
                   selected_line_segment_indexes=[],
                   flagged_line_segment_indexes=[]):  # flagged_line_segment_indexes not yet used
        if (self.vbo_line_segment_point_xys is not None and len(self.vbo_line_segment_point_xys.data) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if (picker.is_active):
                picker.bind_picker_colors_for_lines(layer_index_base,
                                          len(self.world_line_segment_points))
                gl.glLineWidth(6)
            else:
                gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
                self.vbo_line_segment_colors.bind()
                gl.glColorPointer(self.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
                gl.glLineWidth(style.line_width)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

            gl.glDrawArrays(gl.GL_LINES, 0, np.alen(self.vbo_line_segment_point_xys.data))

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            if (picker.is_active):
                picker.unbind_picker_colors()
            else:
                self.vbo_line_segment_colors.unbind()
                gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_point_xys.unbind()

            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def draw_selected_lines(self, style, selected_line_segment_indexes=[]):
        if (len(selected_line_segment_indexes) != 0):
            gl.glLineWidth(style.line_width + 10)
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
            if (not picker.is_active and len(flagged_point_indexes) != 0):
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

            if (picker.is_active):
                picker.bind_picker_colors_for_points(layer_index_base,
                                                     len(self.vbo_point_xys.data))
                gl.glPointSize(point_size + 8)
            else:
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

            gl.glDrawArrays(gl.GL_POINTS, 0, np.alen(self.vbo_point_xys.data))
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

            if (picker.is_active):
                picker.unbind_picker_colors()
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

    def draw_labels_at_points(self, values, screen_rect, projected_rect):
        c = self.canvas

        projected_points = self.vbo_point_xys.data
        n, tex_id = c.prepare_string_texture_for_labels(values, projected_points, projected_rect)
        if (n == 0):
            return

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        c.vbo_screen_vertexes.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        c.vbo_texture_coordinates.bind()
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        # set up an orthogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, rect.width(screen_rect), 0, rect.height(screen_rect), -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        # vertex_count = np.alen( self.character_coordinates_data ) * 4
        vertex_count = n * 4
        gl.glDrawArrays(gl.GL_QUADS, 0, vertex_count)

        # undo the orthogonal projection
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()

        c.vbo_texture_coordinates.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        c.vbo_screen_vertexes.unbind()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

    def set_triangles(self, triangle_point_indexes, triangle_point_colors):
        self.vbo_triangle_point_indexes = gl_vbo.VBO(
            triangle_point_indexes.view(np.uint32),
            usage="GL_STATIC_DRAW",
            target="GL_ELEMENT_ARRAY_BUFFER"
        )
        self.vbo_triangle_point_colors = gl_vbo.VBO(triangle_point_colors)

    def draw_triangles(self, line_width):
        # the line segments
        if (self.vbo_triangle_point_indexes is not None and len(self.vbo_triangle_point_indexes.data) > 0):
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            gl.glEnableClientState(gl.GL_INDEX_ARRAY)  # FIXME: deprecated
            self.vbo_triangle_point_indexes.bind()

            gl.glColor(0.5, 0.5, 0.5, 0.75)
            gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            self.vbo_triangle_point_colors.bind()
            gl.glColorPointer(self.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
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

    def set_image_projection(self, image_data, projection):
        if self.image_textures is None:
            self.image_textures = ImageTextures(image_data)
            # Release the raw image data to free up memory.
            image_data.release_images()
            
        self.image_textures.set_projection(image_data, projection)

    def use_world_rects_as_screen_rects(self, image_data):
        # FIXME: not optimized at all! Currently renegerates everything at
        # every update!
        if self.image_textures is not None:
            self.image_textures.destroy()
            self.image_textures = None
        if self.image_textures is None:
            self.image_textures = ImageTextures(image_data)
            # Release the raw image data to free up memory.
            image_data.release_images()
            
        self.image_textures.use_world_rects_as_screen_rects()

    def draw_image(self, layer_index_base, picker, alpha=1.0):
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        texture = not picker.is_active
        if texture:
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
            gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
            gl.glColor(1.0, 1.0, 1.0, alpha)
        else:
            fill_color = picker.get_polygon_picker_colors(layer_index_base, 1)[0]
            r, g, b, a = int_to_color(fill_color)
            gl.glColor(r, g, b, a)
        for i, vbo in enumerate(self.image_textures.vbo_vertexes):
            # have to bind texture before VBO
            if texture:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.image_textures.textures[i])
            vbo.bind()
            gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            if texture:
                self.image_textures.vbo_texture_coordinates.bind()
                gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

            gl.glDrawArrays(gl.GL_QUADS, 0, 4)

            if texture:
                self.image_textures.vbo_texture_coordinates.unbind()

            vbo.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

    def set_invalid_polygons(self, polygons, polygon_count):
        # Invalid polygons are those that couldn't be tessellated and thus
        # have zero fill triangles. But don't consider hole polygons as
        # invalid polygons.
        invalid_indices_including_holes = np.where(
            self.triangle_vertex_counts[: polygon_count] == 0
        )[0]
        invalid_indices = []

        for index in invalid_indices_including_holes:
            if index > 0 and \
               polygons.group[index] != polygons.group[index - 1]:
                invalid_indices.append(index)

        # this is a mechanism to inform the calling program of invalid polygons
        # TODO: make this a pull (call to a get_invalid_polygons() method) instead
        # of a push (message)
        """
        self.layer.inbox.send(
            request = "set_invalid_polygons",
            polygon_indices = np.array( invalid_indices, np.uint32 ),
        )
        """

    def set_polygons(self, polygons, point_adjacency_array):
        self.point_adjacency_array = point_adjacency_array.copy()
        self.polygons = polygons.copy()
        self.polygon_count = np.alen(polygons)
        self.line_vertex_counts = polygons.count.copy()
        self.triangle_vertex_buffers = np.ndarray(
            self.polygon_count,
            dtype=np.uint32
        )
        self.triangle_vertex_counts = np.ndarray(
            self.polygon_count,
            dtype=np.uint32
        )
        self.line_vertex_buffers = np.ndarray(
            self.polygon_count,
            dtype=np.uint32
        )
        self.line_nan_counts = np.zeros(
            self.polygon_count,
            dtype=np.uint32
        )

        init_vertex_buffers(
            self.triangle_vertex_buffers,  # out parameter -- init_vertex_buffers() builds a vbo buffer for each polygon and stores them in this handle
            self.line_vertex_buffers,  # out parameter -- init_vertex_buffers() builds a vbo buffer for each polygon and stores them in this handle
            start_index=0,
            count=self.polygon_count,
            pygl=gl
        )

        projected_points = self.vbo_point_xys.data
        tessellate(
            projected_points,  # used to be: self.points
            self.point_adjacency_array.next,
            self.point_adjacency_array.polygon,
            self.polygons.start,
            self.polygons.count,  # per-polygon point count
            self.line_nan_counts,  # out parameter -- how many nan/deleted points in each polygon
            self.polygons.group,
            self.triangle_vertex_buffers,  # out parameter -- fills in the triangle vertex points
            self.triangle_vertex_counts,  # out parameter -- how many triangle points for each polygon?
            self.line_vertex_buffers,  # out parameter -- fills in the line vertex points
            gl
        )

        # print "total line_nan_counts = " + str( self.line_nan_counts.sum() )
        self.set_invalid_polygons(self.polygons, self.polygon_count)

    def draw_polygons(self, layer_index_base, picker,
                      polygon_colors, line_color, line_width,
                      broken_polygon_index=None):
        if self.triangle_vertex_buffers is None or self.polygon_count == 0:
            return

        # the fill triangles

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if (picker.is_active):
            active_colors = picker.get_polygon_picker_colors(layer_index_base, self.polygon_count)
        else:
            active_colors = polygon_colors

        render_buffers_with_colors(
            self.triangle_vertex_buffers[: self.polygon_count],
            active_colors,
            self.triangle_vertex_counts[: self.polygon_count],
            gl.GL_TRIANGLES,
            gl
        )

        # the lines

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

        if (picker.is_active):
            gl.glLineWidth(6)
            # note that all of the lines of each polygon get the color of the polygon as a whole
            render_buffers_with_colors(
                self.line_vertex_buffers[: self.polygon_count],
                active_colors,
                self.line_vertex_counts[: self.polygon_count],
                gl.GL_LINE_LOOP,
                gl
            )
        else:
            gl.glLineWidth(line_width)
            render_buffers_with_one_color(
                self.line_vertex_buffers[: self.polygon_count],
                line_color,
                self.line_vertex_counts[: self.polygon_count],
                gl.GL_LINE_LOOP,
                gl,
                0 if broken_polygon_index is None else broken_polygon_index,
                # If needed, render with one polygon border popped open.
                gl.GL_LINE_LOOP if broken_polygon_index is None else gl.GL_LINE_STRIP
            )

        # TODO: drawt the points if the polygon is selected for editing

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


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

    def draw_screen_lines(self, points, width=1.0, red=0.0, green=0.0, blue=0.0, alpha=1.0, stipple_factor=1, stipple_pattern=0xFFFF):
        c = self.canvas
        h = rect.height(c.screen_rect)
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glDisable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(width)
        # don't let texture colors override the line color
        gl.glColor(red, green, blue, alpha)
        gl.glLineStipple(stipple_factor, stipple_pattern)
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glBegin(gl.GL_LINE_STRIP)
        for p in points:
            # flip y to treat point as normal screen coordinates
            gl.glVertex(p[0], h - p[1], 0)
        gl.glEnd()
        gl.glDisable(gl.GL_LINE_STIPPLE)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE)

    def draw_screen_markers(self, markers, style):
        """Draws a list of markers on screen.
        
        Each entry in markers is a 3-tuple; the point to center the marker, a
        point on the other end of the line, and the marker type
        """ 
        c = self.canvas
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(style.line_width)
        gl.glColor4ubv(np.uint32(style.line_color).tostring())
        for p1, p2, symbol in markers:
            marker_points, filled = style.get_marker_data(symbol)
            # Compute the angles in screen coordinates, because using world
            # coordinates for the angles results in the projection being applied,
            # which shows distortion as it moves away from the equator
            point = c.get_numpy_screen_point_from_world_point(p1)
            d = point - c.get_numpy_screen_point_from_world_point(p2)
            mag = np.linalg.norm(d)
            if mag > 0.0:
                d = d / np.linalg.norm(d)
            else:
                d[:] = (1, 0)
            r = np.array(((d[0], d[1]), (d[1], -d[0])), dtype=np.float32)
            points = (np.dot(marker_points, r) * style.line_width) + point
            #self.renderer.draw_screen_lines(a, self.style.line_width, smooth=True, color4b=self.style.line_color)
            h = rect.height(c.screen_rect)
            if filled:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glEnable(gl.GL_POLYGON_SMOOTH)
                gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_DONT_CARE)
                gl.glBegin(gl.GL_TRIANGLE_FAN)
            else:
                gl.glBegin(gl.GL_LINE_LOOP)
            for p in points:
                # flip y to treat point as normal screen coordinates
                gl.glVertex(p[0], h - p[1], 0)
            gl.glEnd()
        gl.glDisable(gl.GL_POLYGON_SMOOTH)

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

        str_len, tex_id = c.prepare_string_texture(point[0], point[1], text)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glColor(1.0, 1.0, 1.0, 1.0)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        c.vbo_screen_vertexes.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        c.vbo_texture_coordinates.bind()
        gl.glTexCoordPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        vertex_count = str_len * 4
        gl.glDrawArrays(gl.GL_QUADS, 0, vertex_count)

        c.vbo_screen_vertexes.unbind()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        c.vbo_texture_coordinates.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

    def get_emulated_dc(self):
        return GLDC(self)


    # Vector object drawing routines

    def get_fill_properties(self, style):
        if style.fill_style == 0:
            return None
        return style.fill_color

    def fill_object(self, layer_index_base, picker, style):
        if (self.vbo_line_segment_point_xys is None or len(self.vbo_line_segment_point_xys.data) == 0):
            return
        
        if (picker.is_active):
            fill_color = picker.get_polygon_picker_colors(layer_index_base, 1)[0]
        else:
            fill_color = self.get_fill_properties(style)
        if fill_color is None:
            return
        
        if (not picker.is_active):
            fill_stipple = style.get_fill_stipple()
            if fill_stipple is not None:
                gl.glEnable(gl.GL_POLYGON_STIPPLE)
                gl.glPolygonStipple(fill_stipple)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        self.vbo_line_segment_point_xys.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        r, g, b, a = int_to_color(fill_color)
        gl.glColor(r, g, b, a)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glEnable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, np.alen(self.vbo_line_segment_point_xys.data))
        gl.glDisable( gl.GL_POLYGON_OFFSET_FILL )
        gl.glDisable(gl.GL_POLYGON_STIPPLE)

        self.vbo_line_segment_point_xys.unbind()

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def outline_object(self, layer_index_base, picker, style):
        if (self.vbo_line_segment_point_xys is None or len(self.vbo_line_segment_point_xys.data) == 0):
            return
        
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        self.vbo_line_segment_point_xys.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        if (picker.is_active):
            picker.bind_picker_colors_for_lines(layer_index_base,
                                      len(self.world_line_segment_points))
            gl.glLineWidth(6)
        else:
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
            self.vbo_line_segment_colors.bind()
            gl.glColorPointer(self.NUM_COLOR_CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None)  # FIXME: deprecated
            gl.glLineWidth(style.line_width)
            gl.glLineStipple(style.line_stipple_factor, style.line_stipple)
            gl.glEnable(gl.GL_LINE_STIPPLE)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_LINES, 0, np.alen(self.vbo_line_segment_point_xys.data))
        gl.glDisable(gl.GL_LINE_STIPPLE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        if (picker.is_active):
            picker.unbind_picker_colors()
        else:
            self.vbo_line_segment_colors.unbind()
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)  # FIXME: deprecated
        self.vbo_line_segment_point_xys.unbind()
        
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
