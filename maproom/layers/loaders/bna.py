import os
import math
import time
import numpy as np
from osgeo import gdal, gdal_array, osr
import pyproj
from maproom.library.accumulator import accumulator, flatten
import maproom.library.rect as rect
import maproom.library.Bitmap as Bitmap

from maproom.layers import PolygonLayer

class BNALoader(object):
    mime = "application/x-maproom-bna"
    
    def can_load(self, metadata):
        return metadata.mime == self.mime
    
    def load(self, metadata, manager):
        layer = PolygonLayer(manager=manager)
        
        (layer.load_error_string,
         f_polygon_points,
         f_polygon_starts,
         f_polygon_counts,
         f_polygon_types,
         f_polygon_identifiers) = load_bna_file(metadata.uri)
        if (layer.load_error_string == ""):
            layer.set_data(f_polygon_points, f_polygon_starts, f_polygon_counts,
                           f_polygon_types, f_polygon_identifiers)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.type = self.mime
        return layer
    
    def can_save(self, layer):
        return False
    
    def check(self, layer):
        raise RuntimeError("Not abte to check BNA files")
    
    def save_to_file(self, f, layer):
        return "Can't save to BNA yet."


def update_status(message):
    import wx
    tlw = wx.GetApp().GetTopWindow()
    tlw.SetStatusText(message)
    wx.SafeYield()


def load_bna_file(file_path):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:
    
    ( load_error_string, polygon_points, polygon_starts, polygon_counts, polygon_types, polygon_identifiers )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        polygon_points = numpy array (type = 2 x np.float32)
        polygon_starts = numpy array (type = 1 x np.uint32)
        polygon_counts = numpy array (type = 1 x np.uint32)
        polygon_types = numpy array (type = 1 x np.uint32) (these are the BNA feature codes)
        polygon_identifiers = list
    """

    print "******** START"
    t0 = time.clock()
    f = file(file_path)
    s = f.read()
    f.close()
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "read in {0} seconds".format(t)

    # arr = np.fromstring(str, dtype=np.float32, sep=' ')
    t0 = time.clock()
    length = len(s)
    print "length = " + str(length)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "length in {0} seconds".format(t)

    t0 = time.clock()
    nr = s.count("\r")
    print "num \\r = = " + str(nr)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "count in {0} seconds".format(t)

    t0 = time.clock()
    nn = s.count("\n")
    print "num \\n = = " + str(nn)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "count in {0} seconds".format(t)

    if (nr > 0 and nn > 0):
        t0 = time.clock()
        s = s.replace("\r", "")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "replace \\r with empty in {0} seconds".format(t)
        nr = 0

    if (nr > 0):
        t0 = time.clock()
        s = s.replace("\r", "\n")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "replace \\r with \\n in {0} seconds".format(t)
        nr = 0

    t0 = time.clock()
    lines = s.split("\n")
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print lines[0]
    print lines[1]
    print "split in {0} seconds".format(t)

    polygon_points = accumulator(block_shape=(2,), dtype=np.float32)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_types = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []

    t0 = time.clock()
    update_interval = .1
    update_every = 1000
    t_update = t0 + update_interval
    total_points = 0
    i = 0
    num_lines = len(lines)
    while True:
        if (i >= num_lines):
            break
        if (i % update_every) == 0:
            t = time.clock()
            if t > t_update:
                update_status("Loading %d of %d data points." % (i, num_lines))
                t_update = t + update_interval
        line = lines[i].strip()
        i += 1
        if (len(line) == 0):
            continue

        # fixme -- this will break if there are commas in any of the fields!
        pieces = line.split(",")
        if len(pieces) != 3:
            return ("The .bna file {0} is invalid. Error at line {1}.".format(file_path, i), None, None, None, None, None)
        polygon_identifiers.append(pieces[0].strip('"'))
        try:
            feature_code = int(pieces[1].strip('"'))
        except ValueError:
            feature_code = 0
        num_points = int(pieces[2])
        original_num_points = num_points

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        is_polygon = False
        if num_points < 3:
            num_points = abs(num_points)
        else:
            is_polygon = True

        # TODO: for now we just assume it's a polygon (could be a polyline or a point)
        # fixme: should we be adding polylines and points?
        # or put them somewhere separate -- particularly points!
        first_point = ()
        for j in xrange(num_points):
            line = lines[i].strip()
            i += 1
            pieces = line.split(",")
            p = (float(pieces[0]), float(pieces[1]))
            if (j == 0):
                first_point = p
            # if the last point is a duplicate of the first point, remove it
            if (j == (num_points - 1) and p[0] == first_point[0] and p[1] == first_point[1]):
                num_points -= 1
                continue
            polygon_points.append(p)

        polygon_starts.append(total_points)
        polygon_counts.append(num_points)
        polygon_types.append(feature_code)
        total_points += num_points

    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "loop in {0} seconds".format(t)
    print "******** END"
    update_status("Loaded %d data points in %.2fs." % (num_lines, t))

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            np.asarray(polygon_types)[:, 0],
            polygon_identifiers)

#

class ImageData(object):
    NORTH_UP_TOLERANCE = 0.002
    
    def __init__(self, dataset):
        self.nbands = dataset.RasterCount
        self.x = dataset.RasterXSize
        self.y = dataset.RasterYSize
        self.projection = None
        
        self.images = []
        self.image_sizes = []
        self.image_world_rects = []
        self.image_textures = []
        
        self.calc_projection(dataset)
        
        print "Image: %sx%s, %d band, %s" % (self.x, self.y, self.nbands, self.projection.srs)
    
    def release_images(self):
        """Free image data after renderer is done converting to textures.
        
        """
        # release images by allowing garbage collector to collect the now
        # unrefcounted images
        self.images = True
    
    def calc_projection(self, dataset):
        if not (dataset.GetProjection() or dataset.GetGCPProjection()):
            # no projection, assume latlong:
            self.projection = pyproj.Proj("+proj=latlong")
        else:
            native_projection = osr.SpatialReference()
            native_projection.ImportFromWkt(
                dataset.GetProjection() or dataset.GetGCPProjection())
            self.projection = pyproj.Proj(native_projection.ExportToProj4())

        self.pixel_to_projected_transform = calculate_pixel_to_projected_transform(dataset)
    
    def is_north_up(self):
        if (len(self.pixel_to_projected_transform) < 6 or
            math.fabs(self.pixel_to_projected_transform[2]) > self.NORTH_UP_TOLERANCE or
                math.fabs(self.pixel_to_projected_transform[4]) > self.NORTH_UP_TOLERANCE):
            return False
        return True
    
    def get_bounds(self):
        bounds = rect.NONE_RECT

        if (self.image_world_rects):
            world_rect_flat_list = flatten(self.image_world_rects)
            b = world_rect_flat_list[0]
            for r in world_rect_flat_list[1:]:
                b = rect.accumulate_rect(b, r)
            bounds = rect.accumulate_rect(bounds, b)
        
        return bounds

    def load_dataset(self, dataset, texture_size):
        raster_bands = []

        for band_index in range(1, dataset.RasterCount + 1):
            raster_bands.append(dataset.GetRasterBand(band_index))

        palette = get_palette(raster_bands[0])

        num_cols = self.x / texture_size
        if ((self.x % texture_size) != 0):
            num_cols += 1
        num_rows = self.y / texture_size
        if ((self.y % texture_size) != 0):
            num_rows += 1

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
                image = get_image(raster_bands,
                                  self.nbands,
                                  palette,
                                  selection_origin,
                                  (selection_width, selection_height))
                images_row.append(image)
                image_sizes_row.append((selection_width, selection_height))
                # we invert the y in going to projected coordinates
                left_bottom_projected = apply_transform((selection_origin[0],
                                                         selection_origin[1] + selection_height),
                                                        self.pixel_to_projected_transform)
                right_top_projected = apply_transform((selection_origin[0] + selection_width,
                                                       selection_origin[1]),
                                                      self.pixel_to_projected_transform)
                if (self.projection.srs.find("+proj=longlat") != -1):
                    # for longlat projection, apparently someone decided that since the projection
                    # is the identity, it might as well do something and so it returns the coordinates as
                    # radians instead of degrees; so here we avoid using the projection altogether
                    left_bottom_world = left_bottom_projected
                    right_top_world = right_top_projected
                else:
                    left_bottom_world = self.projection(left_bottom_projected[0], left_bottom_projected[1], inverse=True)
                    right_top_world = self.projection(right_top_projected[0], right_top_projected[1], inverse=True)
                image_world_rects_row.append((left_bottom_world, right_top_world))
            self.images.append(images_row)
            self.image_sizes.append(image_sizes_row)
            self.image_world_rects.append(image_world_rects_row)


def load_image_file(file_path):
    """
    Load data from a raster file. Returns:
    
    ( load_error_string, images, image_sizes, image_world_rects )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
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
        projection = the file's projection, as a pyproj-style projection callable object,
                     such that projection( world_x, world_y ) = ( projected_x, projected_y )
    """

    SCANLINE_DRIVER_NAMES = ("BSB")
    TEXTURE_SIZE = 1024

    # disable the default error handler so errors don't end up on stderr
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    dataset = gdal.Open(str(file_path))

    if (dataset is None):
        return ("Unable to load the image file " + file_path, None)

    if (dataset.RasterCount < 0 or dataset.RasterCount > 3):
        return ("The number of raster bands is unsupported for file " + file_path, None)

    has_scaline_data = False
    if (dataset.GetDriver().ShortName in SCANLINE_DRIVER_NAMES):
        has_scaline_data = True

    image_data = ImageData(dataset)

    if (not image_data.is_north_up()):
        return ("The raster is not north-up for file " + file_path, None)

    image_data.load_dataset(dataset, TEXTURE_SIZE)

    return ("", image_data)


def calculate_pixel_to_projected_transform(dataset):
    # Only bother to calculate the raster's geographic transform if GDAL
    # reports it as the default transform. Otherwise, use the transform
    # that GDAL provides.
    transform = dataset.GetGeoTransform()

    DEFAULT_TRANSFORM = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if (transform == DEFAULT_TRANSFORM):
        transform = gdal.GCPsToGeoTransform(dataset.GetGCPs())

    return np.array(transform)


def apply_transform(point, transform):
    return (
        transform[0] +
        point[0] * transform[1] +
        point[1] * transform[2],
        transform[3] +
        point[0] * transform[4] +
        point[1] * transform[5],
    )


def get_palette(raster_band):
    if (raster_band.DataType != gdal.GDT_Byte):
        return None

    color_table = raster_band.GetRasterColorTable()
    if (color_table is None):
        return None

    # Prepare the color table for converting the 8-bit paletted image to
    # a 32-bit RGBA image.
    max_color_count = 256
    component_count = 3  # red, green, and blue
    palette = np.zeros(
        (max_color_count, component_count),
        dtype=np.uint8,
    )

    for index in range(color_table.GetCount()):
        palette[index] = color_table.GetColorEntry(index)[0: component_count]

    return palette


def get_image(raster_bands, raster_count, palette, selection_origin, selection_size):
    DEFAULT_ALPHA = 255
    image = None

    # Fetch the data from GDAL and convert it to RGBA.
    if (palette != None):
        # paletted files have a single band
        image = gdal_array.BandReadAsArray(
            raster_bands[0],
            selection_origin[0],
            selection_origin[1],
            selection_size[0],
            selection_size[1],
            selection_size[0],
            selection_size[1])

        image = Bitmap.paletted_to_rgba(image, palette, DEFAULT_ALPHA)
    else:
        image = np.zeros(
            (selection_size[1], selection_size[0], 4),
            np.uint8,
        )
        image[:, :, 3] = DEFAULT_ALPHA

        for band_index in range(0, raster_count):
            band_data = gdal_array.BandReadAsArray(
                raster_bands[band_index],
                selection_origin[0],
                selection_origin[1],
                selection_size[0],
                selection_size[1],
                selection_size[0],
                selection_size[1])

            image[:, :, band_index] = band_data

    return image
