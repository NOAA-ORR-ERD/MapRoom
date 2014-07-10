import os
import wx
import numpy as np
import time
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproom.library.rect as rect
from maproom.library.accumulator import accumulator


class Label_set_renderer:
    oglr = None
    screen_vertexes_data = None
    texture_coordinates_data = None

    # these both are set dynamically on each render, using only the first n
    # records, where n is the number of characters actually to be drawn
    vbo_screen_vertexes = None
    vbo_texture_coordinates = None

    def __init__(self, opengl_renderer, max_label_characters):
        """
            points = 2 x np.float32, i.e., "2f4"
            strings = array of strings, one per point
            projection = a pyproj-style projection callable object, such that
                            projection( world_x, world_y ) = ( projected_x, projected_y )
        """

        self.oglr = opengl_renderer

        self.screen_vertexes_data = np.zeros(
            (max_label_characters, ),
            dtype=self.oglr.QUAD_VERTEX_DTYPE,
        ).view(np.recarray)
        self.screen_vertexes_raw = self.screen_vertexes_data.view(dtype=np.float32).reshape(-1,8)
        
        self.texture_coordinates_data = np.zeros(
            (max_label_characters, ),
            dtype=self.oglr.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        self.texture_coordinates_raw = self.texture_coordinates_data.view(dtype=np.float32).reshape(-1,8)

        # note that the data for these vbo arrays is not yet set; it is set on
        # each render and depends on the number of points being labeled
        #
        # Also note that PyOpenGL 3.1 doesn't allow VBO data to be updated
        # later when using a recarray, so force the VBO to use the raw view
        # into the recarray
        self.vbo_screen_vertexes = gl_vbo.VBO(self.screen_vertexes_raw)
        self.vbo_texture_coordinates = gl_vbo.VBO(self.texture_coordinates_raw)

    def render(self, layer_index_base, pick_mode, screen_rect, max_label_characters, depths, projected_points, projected_rect, projected_units_per_pixel):
        if (self.vbo_screen_vertexes == None or len(self.vbo_screen_vertexes.data) == 0):
            return

        # labels can't be selected with mouse click
        if (pick_mode):
            return

        n = self.prepare_to_render(max_label_characters, depths, projected_points, projected_rect, projected_units_per_pixel)
        if (n == 0):
            return

        t0 = time.clock()

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.oglr.font_texture)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)  # FIXME: deprecated
        self.vbo_screen_vertexes.bind()
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)  # FIXME: deprecated

        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)
        self.vbo_texture_coordinates.bind()
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
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

        self.vbo_texture_coordinates.unbind()
        gl.glDisableClientState(gl.GL_TEXTURE_COORD_ARRAY)

        self.vbo_screen_vertexes.unbind()
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "draw labels in {0} seconds".format( t )

    def prepare_to_render(self, max_label_characters, depths, projected_points, projected_rect, projected_units_per_pixel):
        t0 = time.clock()

        # (0) check for too many points to label (i.e., would take too long to position the labels)
        r1 = projected_points[:, 0] >= projected_rect[0][0]
        r2 = projected_points[:, 0] <= projected_rect[1][0]
        r3 = projected_points[:, 1] >= projected_rect[0][1]
        r4 = projected_points[:, 1] <= projected_rect[1][1]
        mask = np.logical_and(np.logical_and(r1, r2), np.logical_and(r3, r4))
        relevant_indexes = np.where(mask)[0]
        n = np.alen(relevant_indexes)
        relevant_points = projected_points[relevant_indexes]
#        
#        print "%d depths: %s" % (len(depths), str(depths))
#        print "%d relevant points: %s" % (len(relevant_points), str(relevant_points))

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "check a in {0} seconds".format( t )
        t0 = time.clock()

        # print "going to draw {0} labels".format( n )
        if (n == 0 or n > max_label_characters):
            return 0

        relevant_depths = depths[relevant_indexes]
        n = sum(map(len, map(str, relevant_depths)))
        # import code; code.interact( local = locals() )

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "check b in {0} seconds".format( t )

        # print "going to draw {0} characters".format( n )
        if (n == 0 or n > max_label_characters):
            return 0

        if (n == 0 or n > max_label_characters):
            return 0

        #

        labels = map(str, relevant_depths)

        t0 = time.clock()

        character_coord_accumulators = [[], [], [], [], [], [], [], []]
        tex_coord_accumulators = [[], [], [], [], [], [], [], []]

        texture_width = float(self.oglr.font_texture_size[0])
        texture_height = float(self.oglr.font_texture_size[1])

        for index, depth in enumerate(relevant_depths):
            s = str(depth)
            # determine the width of the label
            width = 0
            for c in s:
                if c not in self.oglr.font_extents:
                    c = "?"
                width += self.oglr.font_extents[c][2]
            x_offset = -width / 2

            projected_point = relevant_points[index]
            base_screen_x = (projected_point[0] - projected_rect[0][0]) / projected_units_per_pixel
            base_screen_y = (projected_point[1] - projected_rect[0][1]) / projected_units_per_pixel
            # print str( base_screen_x ) + "," + str( base_screen_y ) + "," + str( x_offset )

            for c in s:
                if c not in self.oglr.font_extents:
                    c = "?"

                x = self.oglr.font_extents[c][0]
                y = self.oglr.font_extents[c][1]
                w = self.oglr.font_extents[c][2]
                h = self.oglr.font_extents[c][3]

                # lb
                character_coord_accumulators[0].append(base_screen_x + x_offset)
                character_coord_accumulators[1].append(base_screen_y - 2 - h)
                # lt
                character_coord_accumulators[2].append(base_screen_x + x_offset)
                character_coord_accumulators[3].append(base_screen_y - 2)
                # rb
                character_coord_accumulators[4].append(base_screen_x + x_offset + w)
                character_coord_accumulators[5].append(base_screen_y - 2)
                # rt
                character_coord_accumulators[6].append(base_screen_x + x_offset + w)
                character_coord_accumulators[7].append(base_screen_y - 2 - h)

                # lb
                tex_coord_accumulators[0].append(x / texture_width)
                tex_coord_accumulators[1].append((y + h) / texture_height)
                # lt
                tex_coord_accumulators[2].append(x / texture_width)
                tex_coord_accumulators[3].append(y / texture_height)
                # rt
                tex_coord_accumulators[4].append((x + w) / texture_width)
                tex_coord_accumulators[5].append(y / texture_height)
                # rb
                tex_coord_accumulators[6].append((x + w) / texture_width)
                tex_coord_accumulators[7].append((y + h) / texture_height)

                x_offset += w

        #

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "accumulate coords in {0} seconds".format( t )
        t0 = time.clock()

        self.screen_vertexes_data.x_lb[0: n] = character_coord_accumulators[0]
        self.screen_vertexes_data.y_lb[0: n] = character_coord_accumulators[1]
        self.screen_vertexes_data.x_lt[0: n] = character_coord_accumulators[2]
        self.screen_vertexes_data.y_lt[0: n] = character_coord_accumulators[3]
        self.screen_vertexes_data.x_rt[0: n] = character_coord_accumulators[4]
        self.screen_vertexes_data.y_rt[0: n] = character_coord_accumulators[5]
        self.screen_vertexes_data.x_rb[0: n] = character_coord_accumulators[6]
        self.screen_vertexes_data.y_rb[0: n] = character_coord_accumulators[7]

        self.texture_coordinates_data.u_lb[0: n] = tex_coord_accumulators[0]
        self.texture_coordinates_data.v_lb[0: n] = tex_coord_accumulators[1]
        self.texture_coordinates_data.u_lt[0: n] = tex_coord_accumulators[2]
        self.texture_coordinates_data.v_lt[0: n] = tex_coord_accumulators[3]
        self.texture_coordinates_data.u_rt[0: n] = tex_coord_accumulators[4]
        self.texture_coordinates_data.v_rt[0: n] = tex_coord_accumulators[5]
        self.texture_coordinates_data.u_rb[0: n] = tex_coord_accumulators[6]
        self.texture_coordinates_data.v_rb[0: n] = tex_coord_accumulators[7]

        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        # print "assign in {0} seconds".format( t )
        t0 = time.clock()

        # print self.screen_vertexes_data[ 0 : n ]
        # print self.texture_coordinates_data[ 0 : n ]

        self.vbo_screen_vertexes[0: n] = self.screen_vertexes_raw[0: n]
        self.vbo_texture_coordinates[0: n] = self.texture_coordinates_raw[0: n]

        return n

    def destroy(self):
        self.vbo_screen_vertexes = None
        self.vbo_texture_coordinates = None
