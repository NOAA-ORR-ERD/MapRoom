import os
import wx
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

import pyproj

from maproom.library.accumulator import flatten
import maproom.library.rect as rect

import data_types

import logging
log = logging.getLogger(__name__)


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
    
    def __init__(self, x, y, texture_size=1024):
        self.x = x
        self.y = y
        self.texture_size = texture_size
        self.projection = None
        self.pixel_to_projected_transform = np.array((0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        
        self.images = []
        self.image_sizes = []
        self.image_world_rects = []
        
        self.calc_textures(self.texture_size)
    
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
        
        # UPDATE: need to keep raw images around for PDF rendering
        #self.images = True
        pass
    
    def set_projection(self, projection=None):
        if projection is None:
            # no projection, assume latlong:
            projection = pyproj.Proj("+proj=latlong")
        self.projection = projection
        self.calc_image_world_rects()
    
    def get_bounds(self):
        bounds = rect.NONE_RECT

        if (self.image_world_rects):
            b = self.image_world_rects[0]
            for r in self.image_world_rects[1:]:
                b = rect.accumulate_rect(b, r)
            bounds = rect.accumulate_rect(bounds, b)
        
        return bounds
    
    def calc_textures(self, texture_size):
        self.texture_size = texture_size
        num_cols = self.x / texture_size
        if ((self.x % texture_size) != 0):
            num_cols += 1
        num_rows = self.y / texture_size
        if ((self.y % texture_size) != 0):
            num_rows += 1
        self.image_sizes = []
        for r in xrange(num_rows):
            selection_height = texture_size
            if (((r + 1) * texture_size) > self.y):
                selection_height -= (r + 1) * texture_size - self.y
            for c in xrange(num_cols):
                selection_origin = (c * texture_size, r * texture_size)
                selection_width = texture_size
                if (((c + 1) * texture_size) > self.x):
                    selection_width -= (c + 1) * texture_size - self.x
                self.image_sizes.append((selection_origin, (selection_width, selection_height)))
    
    def calc_world_rect(self, selection_origin, selection_size):
        # we invert the y in going to projected coordinates
        left_bottom_projected = apply_transform(
            (selection_origin[0],
             selection_origin[1] + selection_size[1]),
            self.pixel_to_projected_transform)
        left_top_projected = apply_transform(
            (selection_origin[0],
             selection_origin[1]),
            self.pixel_to_projected_transform)
        right_top_projected = apply_transform(
            (selection_origin[0] + selection_size[0],
             selection_origin[1]),
            self.pixel_to_projected_transform)
        right_bottom_projected = apply_transform(
            (selection_origin[0] + selection_size[0],
             selection_origin[1] + selection_size[1]),
            self.pixel_to_projected_transform)
        log.debug("calc_world_rect: projection=%s" % self.projection.srs)
        log.debug("  before: %s" % str((left_bottom_projected, left_top_projected, right_top_projected, right_bottom_projected)))
        if self.projection.srs.find("+proj=longlat") != -1 or self.projection.srs.find("+proj=latlong") != -1:
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
        log.debug("  after: %s" % str((left_bottom_world, left_top_world, right_top_world, right_bottom_world)))
        
        return left_bottom_world, left_top_world, right_top_world, right_bottom_world
    
    def calc_image_world_rects(self):
        self.image_world_rects = []
        for entry in self.image_sizes:
            selection_origin, selection_size = entry
            world_rect = self.calc_world_rect(selection_origin, selection_size)
            log.debug("image size: %s" % str(entry))
            log.debug("world rect: %s" % str(world_rect))
            self.image_world_rects.append(world_rect)

    def load_texture_data(self, subimage_loader):
        subimage_loader.prepare(len(self.image_sizes))
        images = []
        for entry in self.image_sizes:
            selection_origin, selection_size = entry
            image = subimage_loader.load(selection_origin, selection_size)
            images.append(image)
        self.images = images
    
    def set_control_points(self, cp, projection):
        xoffset = cp[0][0]
        yoffset = cp[0][1]
        xscale = (cp[1][0] - cp[0][0])/self.x
        yscale = (cp[3][1] - cp[0][1])/self.y
        self.pixel_to_projected_transform = np.array((xoffset, xscale, 0.0, yoffset, 0.0, yscale))
        self.set_projection(projection)
    
    def set_rect(self, rect, projection):
        xoffset = rect[0][0]
        yoffset = rect[0][1]
        xscale = (rect[1][0] - rect[0][0])/self.x
        yscale = (rect[1][1] - rect[0][1])/self.y
        self.pixel_to_projected_transform = np.array((xoffset, xscale, 0.0, yoffset, 0.0, yscale))
        self.set_projection(projection)
    
    def load_numpy_array(self, cp, array, projection=None):
        if projection is not None:
            self.set_control_points(cp, projection)
        loader = RawSubImageLoader(array)
        self.load_texture_data(loader)


class SubImageLoader(object):
    def prepare(self, num_sub_images):
        pass
    
    def load(self, origin, size):
        pass


class RawSubImageLoader(SubImageLoader):
    def __init__(self, array):
        self.array = array
    
    def load(self, origin, size):
        # numpy image coords are reversed!
        return self.array[origin[1]:origin[1] + size[1],
                          origin[0]:origin[0] + size[0],:]


class ImageTextures(object):
    """Class to allow sharing of textures between views
    
    """
    def __init__(self, image_data):
        self.blank = np.array([128, 128, 128, 128], 'B')
        self.textures = []
        self.vbo_vertexes = []
        self.vbo_texture_coordinates = None  # just one, same one for all images
        self.load(image_data)

    def load(self, image_data):
        texcoord_data = np.zeros(
            (1, ),
            dtype=data_types.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        texcoord_raw = texcoord_data.view(dtype=np.float32).reshape(-1,8)

        n = 0
        image_list = image_data.images
        for i in xrange(len(image_data.image_sizes)):
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
    
    def set_projection(self, image_data, projection):
        image_projected_rects = []
        log.debug("set_projection: world rects=%s" % str(image_data.image_world_rects))
        for lb, lt, rt, rb in image_data.image_world_rects:
            left_bottom_projected = projection(lb[0], lb[1])
            left_top_projected = projection(lt[0], lt[1])
            right_top_projected = projection(rt[0], rt[1])
            right_bottom_projected = projection(rb[0], rb[1])
            image_projected_rects.append( (left_bottom_projected,
                                           left_top_projected,
                                           right_top_projected,
                                           right_bottom_projected) )
            log.debug("  world -> proj: %s -> %s" % (str((lb, lt, rt, rb)), str(image_projected_rects[-1])))

        for i, projected_rect in enumerate(image_projected_rects):
            log.debug("  projected #%d: %s" % (i, str(projected_rect)))
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
    
    def use_screen_rect(self, image_data, r):
        for i, entry in enumerate(image_data.image_sizes):
            selection_origin, selection_size = entry
            x = selection_origin[0]
            y = selection_origin[1]
            w = selection_size[0]
            h = selection_size[1]
            raw = self.vbo_vertexes[i].data
            vertex_data = raw.view(dtype=data_types.QUAD_VERTEX_DTYPE, type=np.recarray)
            vertex_data.x_lb = x + r[0][0]
            vertex_data.y_lb = y + r[0][1]
            vertex_data.x_lt = x + r[0][0]
            vertex_data.y_lt = y + r[0][1] + h
            vertex_data.x_rt = x + r[0][0] + w
            vertex_data.y_rt = y + r[0][1] + h
            vertex_data.x_rb = x + r[0][0] + w
            vertex_data.y_rb = y + r[0][1]

            self.vbo_vertexes[i][: np.alen(vertex_data)] = raw
    
    def center_at_screen_point(self, image_data, point, screen_height):
        left = int(point[0] - image_data.x/2)
        bottom = int(point[1] + image_data.y/2)
        right = left + image_data.x
        top = bottom + image_data.y
        # flip y to treat rect as normal opengl coordinates
        r = ((left, screen_height - bottom),
             (right, screen_height - top))
        self.use_screen_rect(image_data, r)

    def destroy(self):
        for texture in self.textures:
            gl.glDeleteTextures(np.array([texture], np.uint32))
        self.vbo_vertexes = None
        self.vbo_texture_coordinates = None
