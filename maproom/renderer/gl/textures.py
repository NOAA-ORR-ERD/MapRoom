import Queue
import weakref

import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import pyproj

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


class Image(object):
    def __init__(self, origin, size):
        self.origin = origin  # Origin relative to original image if this is a subset
        self.size = size  # width, height in pixels
        self.world_rect = None
        self.data = None


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

        self.image_list = []

        self.calc_textures(self.texture_size)

    def __iter__(self):
        return iter(self.image_list)

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

    def set_projection(self, projection=None):
        if projection is None:
            # no projection, assume latlong:
            projection = pyproj.Proj("+proj=latlong")
        self.projection = projection
        self.calc_image_world_rects()

    def get_bounds(self):
        bounds = rect.NONE_RECT

        if (self.image_list):
            b = self.image_list[0].world_rect
            for r in self.image_list[1:]:
                b = rect.accumulate_rect(b, r.world_rect)
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
        self.image_list = []
        for r in xrange(num_rows):
            selection_height = texture_size
            if (((r + 1) * texture_size) > self.y):
                selection_height -= (r + 1) * texture_size - self.y
            for c in xrange(num_cols):
                selection_origin = (c * texture_size, r * texture_size)
                selection_width = texture_size
                if (((c + 1) * texture_size) > self.x):
                    selection_width -= (c + 1) * texture_size - self.x

                image = Image(selection_origin, (selection_width, selection_height))
                self.image_list.append(image)

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
        """ Includes a simple dateline check to move images that cross the
        dateline or are in far east latitudes to move to the west latitude (US
        Centric) side of the map.
        """
        for entry in self.image_list:
            wr = self.calc_world_rect(entry.origin, entry.size)
            world_rect = (((wr[0][0] % 360) - 360, wr[0][1]), ((wr[1][0] % 360) - 360, wr[1][1]), ((wr[2][0] % 360) - 360, wr[2][1]), ((wr[3][0] % 360) - 360, wr[3][1]))
            log.debug("image size: %s" % str(entry))
            log.debug("world rect: %s" % str(world_rect))
            entry.world_rect = world_rect

    def load_texture_data(self, subimage_loader):
        subimage_loader.prepare(len(self.image_list))
        for entry in self.image_list:
            entry.data = subimage_loader.load(entry.origin, entry.size)

    def set_control_points(self, cp, projection):
        xoffset = cp[0][0]
        yoffset = cp[0][1]
        xscale = (cp[1][0] - cp[0][0]) / self.x
        yscale = (cp[3][1] - cp[0][1]) / self.y
        self.pixel_to_projected_transform = np.array((xoffset, xscale, 0.0, yoffset, 0.0, yscale))
        self.set_projection(projection)

    def set_rect(self, rect, projection):
        xoffset = rect[0][0]
        yoffset = rect[0][1]
        xscale = (rect[1][0] - rect[0][0]) / self.x
        yscale = (rect[1][1] - rect[0][1]) / self.y
        self.pixel_to_projected_transform = np.array((xoffset, xscale, 0.0, yoffset, 0.0, yscale))
        self.set_projection(projection)

    def load_numpy_array(self, cp, array, projection=None):
        if projection is not None:
            self.set_control_points(cp, projection)
        loader = RawSubImageLoader(array)
        self.load_texture_data(loader)

    @property
    def is_blank(self):
        for entry in self.image_list:
            # check alpha channel for each block
            blank = not entry.data[:,:,3].any()
            if not blank:
                return False
        return True


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


class TileImage(Image):
    def __init__(self, xy, zoom_level, texture_size, world_lb_rt):
        Image.__init__(self, None, (texture_size, texture_size))
        self.xy = xy
        self.z = zoom_level
        lb, rt = world_lb_rt
        lt = (lb[0], rt[1])
        rb = (rt[0], lb[1])
        self.world_rect = (lb, lt, rt, rb)


class TileImageData(ImageData):
    def __init__(self, projection, downloader, texture_size=256):
        ImageData.__init__(self, 0, 0, texture_size)
        self.zoom_level = -1
        self.last_zoom_level = -1
        self.projection = projection
        self.downloader_ref = weakref.ref(downloader)
        self.last_requested = None
        self.requested = dict()  # (x, y): Image

    def __iter__(self):
        return self.requested.itervalues()

    def calc_textures(self, texture_size):
        pass

    def update_tiles(self, zoom, w_r, manager, event_data):
        downloader = self.downloader_ref()
        if downloader is None:
            log.warning("Download thread has been stopped")
            return
        host = downloader.host
        if self.zoom_level != zoom:
            self.set_zoom_level(zoom)
        top_left = host.world_to_tile_num(self.zoom_level, w_r[0][0], w_r[1][1])
        bot_right = host.world_to_tile_num(self.zoom_level, w_r[1][0], w_r[0][1])
        print "UPDATE_TILES:", top_left, bot_right
        tile_list = self.calc_center_tiles(top_left, bot_right)
        print "CENTER TILES:", tile_list
        self.request_tiles(tile_list, manager, event_data)
        tile_list = self.calc_border_tiles(top_left, bot_right)
        print "BORDER TILES", tile_list
        self.request_tiles(tile_list, manager, event_data)

    def set_zoom_level(self, zoom):
        zoom = int(zoom)
        if zoom == self.last_zoom_level:
            # if zoom levels are just swapped, don't throw anything away, just
            # change the pointers around
            self.zoom_level, self.last_zoom_level = self.last_zoom_level, self.zoom_level
            self.requested, self.last_requested = self.last_requested, self.requested
        elif zoom != self.zoom_level:
            self.release_tiles()
            self.last_zoom_level = self.zoom_level
            self.last_requested = self.requested
            self.zoom_level = zoom
            self.requested = dict()

    def release_tiles(self):
        print "RELEASING TILES FOR ZOOM=%d: %s" % (self.last_zoom_level, self.last_requested)

    def calc_center_tiles(self, tl, br):
        needed = []
        x1, y1 = tl
        x2, y2 = br
        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                tile = (x, y)
                if not tile in self.requested:
                    needed.append(tile)
        return needed

    def calc_border_tiles(self, tl, br):
        needed = []
        z = self.zoom_level
        x1, y1 = tl
        x2, y2 = br
        x1 = max(x1 - 1, 0)
        y1 = max(y1 - 1, 0)
        n = (2 << (z - 1)) - 1
        x2 = min(x2 + 1, n)
        y2 = min(y2 + 1, n)
        for x in [x1, x2]:
            for y in range(y1, y2 + 1):
                tile = (x, y)
                if not tile in self.requested:
                    needed.append(tile)
        for y in [y1, y2]:
            for x in range(x1 + 1, x2):
                tile = (x, y)
                if not tile in self.requested:
                    needed.append(tile)
        return needed

    def request_tiles(self, tiles, manager, event_data):
        downloader = self.downloader_ref()
        if downloader is None:
            log.warning("Download thread has been stopped")
            return
        for tile in tiles:
            if tile not in self.requested:
                print "REQUESTING TILE:", tile
                req = downloader.request_tile(self.zoom_level, tile[0], tile[1], manager, event_data)
                self.requested[tile] = TileImage(tile, self.zoom_level, self.texture_size, req.world_lb_rt)

    def add_tiles(self, queue, image_textures):
        if image_textures.static_renderer:
            # short circuit to skip for PDF renderer or other renderers that
            # don't dynamically update the screen
            return
        try:
            while True:
                tile_request = queue.get_nowait()
                print "GOT TILE:", tile_request
                tile = (tile_request.x, tile_request.y)
                if tile not in self.requested:
                    # Hmmm, got tile info for something that's not currently in
                    # the request list.  Maybe a really late server response?
                    # Or the user zoomed in and out really quickly? If it's
                    # for the same zoom level, let's use it.
                    if tile_request.zoom == self.zoom_level:
                        print "  Using tile received but not requested:", tile
                        self.requested[tile] = TileImage(tile, self.zoom_level, self.texture_size, tile_request.world_lb_rt)
                    else:
                        print "  Ignoring tile received but not requested:", tile
                if tile in self.requested:
                    tile_image = self.requested[tile]
                    tile_image.data = tile_request.get_image_array()
                    image_textures.add_tile(tile_image, self.projection)
        except Queue.Empty:
            pass


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
        for i, image in enumerate(image_data):
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

            if image.data is not None:
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D,
                    0,  # level
                    gl.GL_RGBA8,
                    image.data.shape[1],  # width
                    image.data.shape[0],  # height
                    0,  # border
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    image.data
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
        log.debug("set_projection: image_list=%s" % str(list(image_data)))
        for i, image in enumerate(image_data):
            log.debug("  world rect #%d: %s" % (i, str(image.world_rect)))
            lb, lt, rt, rb = image.world_rect
            lb_projected = projection(lb[0], lb[1])
            lt_projected = projection(lt[0], lt[1])
            rt_projected = projection(rt[0], rt[1])
            rb_projected = projection(rb[0], rb[1])

            log.debug("  projected #%d: %s" % (i, str((lb_projected, lt_projected, rt_projected, rb_projected))))
            raw = self.vbo_vertexes[i].data
            vertex_data = raw.view(dtype=data_types.QUAD_VERTEX_DTYPE, type=np.recarray)
            vertex_data.x_lb = lb_projected[0]
            vertex_data.y_lb = lb_projected[1]
            vertex_data.x_lt = lt_projected[0]
            vertex_data.y_lt = lt_projected[1]
            vertex_data.x_rt = rt_projected[0]
            vertex_data.y_rt = rt_projected[1]
            vertex_data.x_rb = rb_projected[0]
            vertex_data.y_rb = rb_projected[1]

            self.vbo_vertexes[i][: np.alen(vertex_data)] = raw

    def use_screen_rect(self, image_data, r, scale=1.0):
        for i, image in enumerate(image_data):
            x = image.origin[0] * scale
            y = image.origin[1] * scale
            w = image.size[0] * scale
            h = image.size[1] * scale
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

    def center_at_screen_point(self, image_data, point, screen_height, scale=1.0):
        left = int(point[0] - (image_data.x / 2) * scale)
        bottom = int(point[1] + (image_data.y / 2) * scale)
        right = left + (image_data.x * scale)
        top = bottom + (image_data.y * scale)
        # flip y to treat rect as normal opengl coordinates
        r = ((left, screen_height - bottom),
             (right, screen_height - top))
        self.use_screen_rect(image_data, r, scale)

    def destroy(self):
        for texture in self.textures:
            gl.glDeleteTextures(np.array([texture], np.uint32))
        self.vbo_vertexes = None
        self.vbo_texture_coordinates = None


class VBOTexture(object):
    def __init__(self, xy, z, tex_id):
        self.xy = xy
        self.z = z
        self.tex_id = tex_id
        self.vbo_vertexes = None


class TileTextures(object):
    """Class to allow sharing of textures between views
    
    """
    static_renderer = False

    def __init__(self, image_data):
        self.blank = np.array([128, 128, 128, 128], 'B')
        self.tiles = []
        self.vbo_texture_coordinates = None  # just one, same one for all images

    def get_vbo_texture_coords(self):
        texcoord_data = np.zeros(
            (1, ),
            dtype=data_types.TEXTURE_COORDINATE_DTYPE,
        ).view(np.recarray)
        texcoord_raw = texcoord_data.view(dtype=np.float32).reshape(-1,8)

        texcoord_data.u_lb = 0
        texcoord_data.v_lb = 1.0
        texcoord_data.u_lt = 0
        texcoord_data.v_lt = 0
        texcoord_data.u_rt = 1.0
        texcoord_data.v_rt = 0
        texcoord_data.u_rb = 1.0
        texcoord_data.v_rb = 1.0

        self.vbo_texture_coordinates = gl_vbo.VBO(texcoord_raw)

    def add_tile(self, image, projection):
        tile = VBOTexture(image.xy, image.z, gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tile.tex_id)
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

        if image.data is not None:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,  # level
                gl.GL_RGBA8,
                image.data.shape[1],  # width
                image.data.shape[0],  # height
                0,  # border
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                image.data
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
        tile.vbo_vertexes = gl_vbo.VBO(vertex_raw)
        self.set_projection(tile, image, projection)

        if self.vbo_texture_coordinates is None:
            self.get_vbo_texture_coords()

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.tiles.append(tile)

    def set_projection(self, tile, image, projection):
        log.debug("  world rect %s: %s" % (tile.xy, str(image.world_rect)))
        lb, lt, rt, rb = image.world_rect
        lb_projected = projection(lb[0], lb[1])
        lt_projected = projection(lt[0], lt[1])
        rt_projected = projection(rt[0], rt[1])
        rb_projected = projection(rb[0], rb[1])

        log.debug("  projected %s: %s" % (tile.xy, str((lb_projected, lt_projected, rt_projected, rb_projected))))
        raw = tile.vbo_vertexes.data
        vertex_data = raw.view(dtype=data_types.QUAD_VERTEX_DTYPE, type=np.recarray)
        vertex_data.x_lb = lb_projected[0]
        vertex_data.y_lb = lb_projected[1]
        vertex_data.x_lt = lt_projected[0]
        vertex_data.y_lt = lt_projected[1]
        vertex_data.x_rt = rt_projected[0]
        vertex_data.y_rt = rt_projected[1]
        vertex_data.x_rb = rb_projected[0]
        vertex_data.y_rb = rb_projected[1]

        tile.vbo_vertexes[: np.alen(vertex_data)] = raw

    def reorder_tiles(self, image_data):
        z_front = image_data.zoom_level
        z_behind = image_data.last_zoom_level
        front = []
        behind = []
        for tile in self.tiles:
            if tile.z == z_front:
                front.append(tile)
            elif tile.z == z_behind:
                behind.append(tile)
            else:
                self.remove_tile(tile)

        # Tiles that appear in front will be drawn last.  Tiles that have
        # been removed won't appear in either the front or behind list
        # will be garbage collected
        self.tiles = behind
        self.tiles.extend(front)

    def remove_tile(self, tile):
        gl.glDeleteTextures(np.array([tile.tex_id], np.uint32))
        tile.vbo_vertexes = None

    def destroy(self):
        for tile in self.tiles:
            self.remove_tile(tile)
        self.vbo_texture_coordinates = None
