import os
import wx
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

from maproom.library.accumulator import flatten
import maproom.library.rect as rect

import data_types


def apply_transform(point, transform):
    return (
        transform[0] +
        point[0] * transform[1] +
        point[1] * transform[2],
        transform[3] +
        point[0] * transform[4] +
        point[1] * transform[5],
    )


class ImageData(object):
    """ Temporary storage object to hold raw image data before converted to GL
    textures.
    
    images = list of lists, where each sublist is a row of images
                and each image is a numpy array [ 0 : max_y, 0 : max_x, 0 : num_bands ]
                where:
                    num_bands = 4
                    max_x and max_y = 1024,
                        except for the last image in each row (may be narrower) and
                        the images in the last row (may be shorter)
    image_sizes = list of lists, the same shape as images,
                  but where each item gives the ( width, height ) pixel size
                  of the corresponding image
    image_world_rects = list of lists, the same shape as images,
                        but where each item gives the world rect
                        of the corresponding image
    """

    NORTH_UP_TOLERANCE = 0.002
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.projection = None
        
        self.images = []
        self.image_sizes = []
        self.image_world_rects = []
    
    def is_threaded(self):
        return False
    
    def release_images(self):
        """Free image data after renderer is done converting to textures.
        
        This has no effect when using the background loader because each image
        chunk is sent to the main thread through a callback.  When using the
        normal non-threaded loader, the entire image is loaded into memory and
        can be freed after GL converts it to textures.
        """
        # release images by allowing garbage collector to collect the now
        # unrefcounted images.
        self.images = True
    
    def set_projection(self, projection=None):
        if projection is None:
            # no projection, assume latlong:
            projection = pyproj.Proj("+proj=latlong")
        self.projection = projection
    
    def get_bounds(self):
        bounds = rect.NONE_RECT

        if (self.image_world_rects):
            world_rect_flat_list = flatten(self.image_world_rects)
            b = world_rect_flat_list[0]
            for r in world_rect_flat_list[1:]:
                b = rect.accumulate_rect(b, r)
            bounds = rect.accumulate_rect(bounds, b)
        
        return bounds
    
    def calc_textures(self, texture_size):
        num_cols = self.x / texture_size
        if ((self.x % texture_size) != 0):
            num_cols += 1
        num_rows = self.y / texture_size
        if ((self.y % texture_size) != 0):
            num_rows += 1
        
        return num_cols, num_rows
    
    def calc_world_rect(self, selection_origin, selection_width, selection_height):
        # we invert the y in going to projected coordinates
        left_bottom_projected = apply_transform(
            (selection_origin[0],
             selection_origin[1] + selection_height),
            self.pixel_to_projected_transform)
        left_top_projected = apply_transform(
            (selection_origin[0],
             selection_origin[1]),
            self.pixel_to_projected_transform)
        right_top_projected = apply_transform(
            (selection_origin[0] + selection_width,
             selection_origin[1]),
            self.pixel_to_projected_transform)
        right_bottom_projected = apply_transform(
            (selection_origin[0] + selection_width,
             selection_origin[1] + selection_height),
            self.pixel_to_projected_transform)
        if (self.projection.srs.find("+proj=longlat") != -1):
            # for longlat projection, apparently someone decided that since the projection
            # is the identity, it might as well do something and so it returns the coordinates as
            # radians instead of degrees; so here we avoid using the projection altogether
            left_bottom_world = left_bottom_projected
            left_top_world = left_top_projected
            right_top_world = right_top_projected
            right_bottom_world = right_bottom_projected
        else:
            left_bottom_world = self.projection(left_bottom_projected[0], left_bottom_projected[1], inverse=True)
            left_top_world = self.projection(left_top_projected[0], left_top_projected[1], inverse=True)
            right_top_world = self.projection(right_top_projected[0], right_top_projected[1], inverse=True)
            right_bottom_world = self.projection(right_bottom_projected[0], right_bottom_projected[1], inverse=True)
        
        return left_bottom_world, left_top_world, right_top_world, right_bottom_world

    def load_texture_data(self, texture_size, subimage_loader):
        num_cols, num_rows = self.calc_textures(texture_size)

        subimage_loader.prepare(num_cols, num_rows)
        for r in xrange(num_rows):
            images_row = []
            image_sizes_row = []
            image_world_rects_row = []
            selection_height = texture_size
            if (((r + 1) * texture_size) > self.y):
                selection_height -= (r + 1) * texture_size - self.y
            for c in xrange(num_cols):
                selection_origin = (c * texture_size, r * texture_size)
                selection_width = texture_size
                if (((c + 1) * texture_size) > self.x):
                    selection_width -= (c + 1) * texture_size - self.x
                world_rect = self.calc_world_rect(selection_origin, selection_width, selection_height)
                image = subimage_loader.load(selection_origin,
                                             (selection_width, selection_height),
                                             world_rect)
                images_row.append(image)
                image_sizes_row.append((selection_width, selection_height))
                image_world_rects_row.append(world_rect)
            self.images.append(images_row)
            self.image_sizes.append(image_sizes_row)
            self.image_world_rects.append(image_world_rects_row)


class SubImageLoader(object):
    def prepare(self, num_cols, num_rows):
        pass
    
    def load(self, origin, size, w_r):
        pass


class ImageScreenData(ImageData):
    def calc_world_rect(self, selection_origin, selection_width, selection_height):
        # "world" coordinates in screen mode are just the screen coordinates
        # we invert the y in going to projected coordinates
        left_bottom = (selection_origin[0],
                       selection_origin[1] + selection_height)
        left_top = (selection_origin[0],
                    selection_origin[1])
        right_top = (selection_origin[0] + selection_width,
                     selection_origin[1])
        right_bottom = (selection_origin[0] + selection_width,
                        selection_origin[1] + selection_height)
        return left_bottom, left_top, right_top, right_bottom


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
    
    def update_texture(self, texture_index, w, h, image):
#        print "ImageData: loading texture index %d" % texture_index
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_index])
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,  # level
            gl.GL_RGBA8,
            w,
            h,
            0,  # border
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            image
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
