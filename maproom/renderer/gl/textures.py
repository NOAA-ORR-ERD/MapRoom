import os
import wx
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

from maproom.library.accumulator import flatten

import data_types

class ImageTextures(object):
    """Class to allow sharing of textures between views
    
    """
    def __init__(self, image_data):
        image_list = flatten(image_data.images)
        self.image_sizes = flatten(image_data.image_sizes)
        self.image_world_rects = flatten(image_data.image_world_rects)

        self.blank = np.array([128, 128, 128, 128], 'B')
        self.textures = []
        self.vbo_vertexes = []
        self.vbo_texture_coordinates = None  # just one, same one for all images
        self.load(image_list)

    def load(self, image_list):
        texcoord_data = np.zeros(
            (1, ),
            dtype=data_types.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        texcoord_raw = texcoord_data.view(dtype=np.float32).reshape(-1,8)

        n = 0
        for i in xrange(len(self.image_sizes)):
            if image_list:
                image_data = image_list[i]
            else:
                image_data = None
            self.textures.append(gl.glGenTextures(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[i])
            # Mipmap levels: half-sized, quarter-sized, etc.
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 4)
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE)
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameter(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
            # gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR )
            # gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST )
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexEnvf(gl.GL_TEXTURE_FILTER_CONTROL, gl.GL_TEXTURE_LOD_BIAS, -0.5)
            
            if image_data is not None:
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,  # level
                    gl.GL_RGBA8,
                    image_data.shape[1],  # width
                    image_data.shape[0],  # height
                    0,  # border
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    image_data
                )
            else:
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,  # level
                    gl.GL_RGBA8,
                    1,  # width
                    1,  # height
                    0,  # border
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    self.blank
                )

            vertex_data = np.zeros(
                (1, ),
                dtype=data_types.QUAD_VERTEX_DTYPE,
            ).view(np.recarray)
            vertex_raw = vertex_data.view(dtype=np.float32).reshape(-1,8)
            # we fill the vbo_vertexes data in reproject() below
            self.vbo_vertexes.append(gl_vbo.VBO(vertex_raw))

        texcoord_data.u_lb = 0
        texcoord_data.v_lb = 1.0
        texcoord_data.u_lt = 0
        texcoord_data.v_lt = 0
        texcoord_data.u_rt = 1.0
        texcoord_data.v_rt = 0
        texcoord_data.u_rb = 1.0
        texcoord_data.v_rb = 1.0

        self.vbo_texture_coordinates = gl_vbo.VBO(texcoord_raw)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def update_texture(self, progress_report):
        # ImageDataProgressReport
        if not hasattr(progress_report, "texture_index"):
            return
#        print "ImageData: loading texture index %d" % progress_report.texture_index
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[progress_report.texture_index])
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,  # level
            gl.GL_RGBA8,
            progress_report.size[0],  # width
            progress_report.size[1],  # height
            0,  # border
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            progress_report.image
        )
    
    def set_projection(self, projection):
        image_projected_rects = []
        for lb, lt, rt, rb in self.image_world_rects:
            left_bottom_projected = projection(lb[0], lb[1])
            left_top_projected = projection(lt[0], lt[1])
            right_top_projected = projection(rt[0], rt[1])
            right_bottom_projected = projection(rb[0], rb[1])
            image_projected_rects.append( (left_bottom_projected,
                                           left_top_projected,
                                           right_top_projected,
                                           right_bottom_projected) )

        for i, projected_rect in enumerate(image_projected_rects):
            raw = self.vbo_vertexes[i].data
            vertex_data = raw.view(dtype=data_types.QUAD_VERTEX_DTYPE, type=np.recarray)
            vertex_data.x_lb = projected_rect[0][0]
            vertex_data.y_lb = projected_rect[0][1]
            vertex_data.x_lt = projected_rect[1][0]
            vertex_data.y_lt = projected_rect[1][1]
            vertex_data.x_rt = projected_rect[2][0]
            vertex_data.y_rt = projected_rect[2][1]
            vertex_data.x_rb = projected_rect[3][0]
            vertex_data.y_rb = projected_rect[3][1]

            self.vbo_vertexes[i][: np.alen(vertex_data)] = raw

    def destroy(self):
        for texture in self.textures:
            gl.glDeleteTextures(np.array([texture], np.uint32))
        self.vbo_vertexes = None
        self.vbo_texture_coordinates = None
