import os
import time

from fs.opener import opener
import numpy as np
from osgeo import gdal, gdal_array, osr
import pyproj

import maproom.library.Bitmap as Bitmap
from maproom.renderer import ImageData

from common import BaseLayerLoader
from maproom.layers import RasterLayer

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class GDALLoader(BaseLayerLoader):
    mime = "image/*"

    name = "GDAL"

    def can_load(self, metadata):
        return metadata.mime.startswith("image/")

    def load_layers(self, metadata, manager):
        layer = RasterLayer(manager=manager)

        progress_log.info("Loading from %s" % metadata.uri)
        (layer.load_error_string, layer.image_data) = load_image_file(metadata.uri)
        if (layer.load_error_string == ""):
            progress_log.info("Finished loading %s" % metadata.uri)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
            layer.update_bounds()
        return [layer]

    def can_save_layer(self, layer):
        return False

    def save_to_file(self, f, layer):
        return "Can't save to GDAL yet."


class GDALImageData(ImageData):
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

    def __init__(self, dataset):
        ImageData.__init__(self, dataset.RasterXSize, dataset.RasterYSize)
        self.nbands = dataset.RasterCount

        self.calc_projection(dataset)

        log.debug("Image: %sx%s, %d band, %s" % (self.x, self.y, self.nbands, self.projection.srs))

    def calc_projection(self, dataset):
        projection = dataset.GetProjection() or dataset.GetGCPProjection()
        log.debug("DATASET projection: %s" % projection)
        if projection:
            native_projection = osr.SpatialReference()
            native_projection.ImportFromWkt(projection)
            projection = pyproj.Proj(native_projection.ExportToProj4())
        else:
            projection = None
        self.pixel_to_projected_transform = calculate_pixel_to_projected_transform(dataset)
        self.set_projection(projection)


class ImageDataBlocks(GDALImageData):
    """Version of ImageData to load using GDAL blocks.
    
    """

    def load_dataset(self, dataset):
        loader = GDALSubImageLoader(dataset)
        self.load_texture_data(loader)


def get_dataset(uri):
    """Get GDAL Dataset, performing URI to filename conversion since GDAL
    doesn't support URIs, only files on the local filesystem
    """

    # disable the default error handler so errors don't end up on stderr
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    fs, relpath = opener.parse(uri)
    print "GDAL:", relpath
    print "GDAL:", fs
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for GDAL: %s" % uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # GDAL doesn't support extended filenames
        file_path = file_path[4:]
    dataset = gdal.Open(str(file_path))

    if (dataset is None):
        return ("Unable to load the image file " + file_path, None)

    if (dataset.RasterCount < 0 or dataset.RasterCount > 4):
        return ("The number of raster bands is unsupported for file " + file_path, None)

    return "", dataset


def load_image_file(uri):
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

    error, dataset = get_dataset(uri)
    if error:
        return (error, None)

    has_scaline_data = False
    if (dataset.GetDriver().ShortName in SCANLINE_DRIVER_NAMES):
        has_scaline_data = True

    t0 = time.clock()
    image_data = ImageDataBlocks(dataset)
    image_data.load_dataset(dataset)
    log.debug("GDAL load time: %f" % (time.clock() - t0))

    return ("", image_data)


def calculate_pixel_to_projected_transform(dataset):
    # Only bother to calculate the raster's geographic transform if GDAL
    # reports it as the default transform. Otherwise, use the transform
    # that GDAL provides.
    transform = dataset.GetGeoTransform()

    DEFAULT_TRANSFORM = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    if (transform == DEFAULT_TRANSFORM):
        cps = dataset.GetGCPs()
        log.debug("GDAL GCPs to generate transform: %s" % str(cps))
        transform = gdal.GCPsToGeoTransform(cps)
    log.debug("GDAL transform: %s" % str(transform))

    return np.array(transform)


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


class GDALSubImageLoader(object):
    def __init__(self, dataset):
        self.nbands = dataset.RasterCount
        self.raster_bands = []
        for band_index in range(1, self.nbands + 1):
            self.raster_bands.append(dataset.GetRasterBand(band_index))
        self.palette = get_palette(self.raster_bands[0])

    def prepare(self, num_sub_images):
        progress_log.info("TICKS=%d" % (num_sub_images))

    def load(self, selection_origin, selection_size):
        progress_log.info("Loading image data...")
        DEFAULT_ALPHA = 255
        image = None

        # Fetch the data from GDAL and convert it to RGBA.
        if (self.palette is not None):
            # paletted files have a single band
            image = gdal_array.BandReadAsArray(
                self.raster_bands[0],
                selection_origin[0],
                selection_origin[1],
                selection_size[0],
                selection_size[1],
                selection_size[0],
                selection_size[1])

            image = Bitmap.paletted_to_rgba(image, self.palette, DEFAULT_ALPHA)
        else:
            image = np.zeros(
                (selection_size[1], selection_size[0], 4),
                np.uint8,
            )
            image[:, :, 3] = DEFAULT_ALPHA

            for band_index in range(0, self.nbands):
                band_data = gdal_array.BandReadAsArray(
                    self.raster_bands[band_index],
                    selection_origin[0],
                    selection_origin[1],
                    selection_size[0],
                    selection_size[1],
                    selection_size[0],
                    selection_size[1])

                image[:, :, band_index] = band_data

        return image
