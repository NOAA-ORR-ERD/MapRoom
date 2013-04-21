import math
import os.path
import numpy as np
import pyproj
from osgeo import gdal, gdal_array, osr
import maproomlib.utility as utility
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Tile_set_layer import Tile_set_layer
from maproomlib.plugin.Point_set_layer import Point_set_layer


class Load_gdal_error( Exception ):
    """ 
    An error occurred when attempting to load a raster file with GDAL.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Transformation_error( Exception ):
    """ 
    An error occurred when attempting to interpret a raster file's geographic
    transformation.
            
    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Gdal_raster_layer( Composite_layer ):
    """
    A set of raster data loaded by the GDAL library.

    The raster file is opened on initialization, but the data is loaded on
    demand when a ``get_data`` message is received.

    :param filename: full path of raster file to open
    :type filename: str
    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param plugin_loader: used to load the appropriate plugin for a file
    :type plugin_loader: maproomlib.utility.Plugin_loader
    :param parent: parent layer containing this layer (if any)
    :type parent: object or NoneType

    .. attribute:: origin

        lower-left corner of the entire raster in geographic coordinates

    .. attribute:: size

        dimensions of the entire raster in geographic coordinates

    .. attribute:: pixel_size

        dimensions of the entire raster in pixels

    .. attribute:: raster_count

        number of raster bands

    .. attribute:: name

        name of the raster (derived from the filename)

    .. attribute:: projection

        geographic lat-long projection used by this raster
        (pyproj.Proj)

    .. attribute:: children

        list of child layers

    .. function:: get_data( pixel_origin, pixel_size, response_box, raster_index = 0 )

        When a message with ``request = "get_data"`` is received within the
        :attr:`inbox`, a handler retrieves the requested data as a bitmap
        and sends it to the given ``response_box``.

        :param pixel_origin: upper-left corner of desired data in raster
                             pixel coordinates
        :type pixel_origin: ( int, int )
        :param pixel_size: dimensions of desired data in raster pixel
                           coordinates
        :type pixel_size: ( int, int )
        :param response_box: response message sent here
        :type response_box: Inbox
        :param raster_index: zero-based index number of the desired raster band
        :type raster_index: int

        The response message is sent as follows::

            response_box.send( request = "data", data )

        That data that is sent to the ``response_box`` is a list of image data
        containing at least the region requested and possibly other regions as
        well. Each element in the list is a dict with three keys:
        ``pixel_origin`` for the region's upper-left corner in raster pixel
        coordinates, ``pixel_size`` for the region's dimensions, and ``data``
        for the actual image, an HxWx4 RGBA numpy array.

    .. function:: get_properties( response_box, indices ):

        When a message with ``request = "get_properties"`` is received within
        the :attr:`inbox`, a handler sends the property data for this layer
        to the given ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox
        :param indices: ignored
        :type indices: object

        The response message is sent as follows::

            response_box.send( request = "properties", properties )

        ``properties`` is a tuple of the properties for this layer.
    """
    PLUGIN_TYPE = "raster_layer"
    # GDAL drivers which read entire scanlines of data at once
    SCANLINE_DRIVER_NAMES = ( "BSB", )
    DEFAULT_ALPHA = 255

    def __init__( self, filename, command_stack, plugin_loader, parent,
                  name, dataset, unique_id ):
        Composite_layer.__init__(
            self, command_stack, plugin_loader, parent, name,
        )

        self.filename = filename
        self.dataset = dataset
        self.unique_id = unique_id

        if self.raster_count < 0 or self.raster_count > 3:
            raise utility.Load_plugin_errorLoad_gdal_error(
                "The number of raster bands is unsupported."
            )

        if self.dataset.GetDriver().ShortName in self.SCANLINE_DRIVER_NAMES:
            self.scanline_data = True
        else:
            self.scanline_data = False

        if not ( self.dataset.GetProjection() or self.dataset.GetGCPProjection() ):
            # no projection, assume latlong:
            self.projection = pyproj.Proj( "+proj=latlong" )
        else:
            native_projection = osr.SpatialReference()
            native_projection.ImportFromWkt(
                self.dataset.GetProjection() or
                self.dataset.GetGCPProjection()
            )
            self.projection = pyproj.Proj(
                native_projection.ExportToProj4()
            )

        geo_transform = self.calculate_geo_transform()
        TOLERANCE = 0.002
        if len( geo_transform ) < 6 or \
           math.fabs( geo_transform[ 2 ] ) > TOLERANCE or \
           math.fabs( geo_transform[ 4 ] ) > TOLERANCE:
            raise utility.Load_plugin_error(
                'The raster file %s is not "north-up".' % filename
            )

        self.pixel_to_geo_transform = geo_transform
        self.geo_to_pixel_transform = \
            self.invert_geo_transform( self.pixel_to_geo_transform )

        self.origin = self.pixel_to_geo( ( 0, 0 ) )

        size = self.pixel_to_geo( (
            float( self.dataset.RasterXSize ),
            float( self.dataset.RasterYSize ),
        ) )
        self.size = (
            math.fabs( size[ 0 ] - self.origin[ 0 ] ),
            math.fabs( size[ 1 ] - self.origin[ 1 ] ),
        )

        self.raster_bands = []

        for band_index in range( 1, self.raster_count + 1 ):
            self.raster_bands.append(
                self.dataset.GetRasterBand( band_index )
            )

        self.palette = self.get_palette( self.raster_bands[ 0 ] )

#        points = np.zeros(
#            ( self.dataset.GetGCPCount(), 2 ),
#            dtype = np.float32,
#        )

#        for ( index, gcp ) in enumerate( self.dataset.GetGCPs() ):
#            points[ index ] = ( gcp.GCPX, gcp.GCPY )

        self.tiles_layer = Tile_set_layer(
            "map tiles",
            self,
            self.size,
            self.pixel_size,
            self.projection,
        )

        self.children = [
            self.tiles_layer,
#            Point_set_layer(
#                "GCPs",
#                points,
#                self.projection,
#            ),
        ]

        self.tiles_layer.outbox.subscribe(
            self.inbox,
            request = ( "get_data", "start_progress", "end_progress" ),
        )

    pixel_size = property( lambda self: (
        self.dataset.RasterXSize,
        self.dataset.RasterYSize,
    ) )
    raster_count = property( lambda self: self.dataset.RasterCount )

    def run( self, scheduler ):
        Composite_layer.run(
            self,
            scheduler,
            get_data = self.get_data,
            get_properties = self.get_properties,
            find_duplicates = self.find_duplicates,
            triangulate = self.triangulate,
            save = self.save,
            layer_removed = self.layer_removed,
        )

    def get_palette( self, raster_band ):
        if raster_band.DataType != gdal.GDT_Byte:
            return None

        color_table = raster_band.GetRasterColorTable()
        if color_table is None:
            return None
            
        # Prepare the color table for converting the 8-bit paletted image to
        # a 32-bit RGBA image.
        max_color_count = 256
        component_count = 3   # red, green, and blue
        palette = np.zeros(
            ( max_color_count, component_count ),
            dtype = np.uint8,
        )

        for index in range( color_table.GetCount() ):
            palette[ index ] = \
                color_table.GetColorEntry( index )[ 0: component_count ]

        return palette

    def get_data( self, pixel_origin, pixel_size, response_box ):
        import maproomlib.utility.Bitmap as Bitmap

        # If the requested pixel dimensions are entirely beyond the edges of
        # the raster, bail.
        if pixel_origin[ 0 ] >= self.pixel_size[ 0 ] or \
           pixel_origin[ 1 ] >= self.pixel_size[ 1 ] or \
           pixel_origin[ 0 ] + pixel_size[ 0 ] < 0 or \
           pixel_origin[ 1 ] + pixel_size[ 1 ] < 0:
            response_box.send(
                request = "data",
                data = None,
                pixel_origin = None,
                pixel_size = None,
            )
            return

        pixel_origin = (
            max( pixel_origin[ 0 ], 0 ),
            max( pixel_origin[ 1 ], 0 ),
        )

        requested_pixel_size = pixel_size
        pixel_size = (
            min( pixel_size[ 0 ], self.pixel_size[ 0 ] - pixel_origin[ 0 ] ),
            min( pixel_size[ 1 ], self.pixel_size[ 1 ] - pixel_origin[ 1 ] ),
        )

        # Fetch the data from GDAL and convert it to RGBA. However, if the
        # data for this file is organized as a series of scanlines (and so
        # requesting any one pixel requires reading an entire line of pixels),
        # then we might as well request the entire width of the raster all at
        # once and break that up into a row of tiles ourselves.
        if self.scanline_data is True:
            raw_data = gdal_array.BandReadAsArray(
                self.raster_bands[ 0 ],
                0,                    # left side of raster
                pixel_origin[ 1 ],
                self.pixel_size[ 0 ], # width of entire raster
                pixel_size[ 1 ],
                self.pixel_size[ 0 ],
                pixel_size[ 1 ],
            )
            images = []
            x = 0

            while x < self.pixel_size[ 0 ]:
                right = min(
                    x + requested_pixel_size[ 0 ],
                    self.pixel_size[ 0 ],
                )
                bottom = pixel_size[ 1 ]

                image = raw_data[ 0: bottom, x: right ].copy()

                # FIXME: Add support for non-paletted (RGBA) scanline images.
                if self.palette is not None:
                    image = Bitmap.paletted_to_rgba(
                        image, self.palette, self.DEFAULT_ALPHA,
                    )

                image = Bitmap.expand(
                    image,
                    requested_pixel_size[ 0 ],
                    requested_pixel_size[ 1 ],
                )

                images.append( dict(
                    pixel_origin = ( x, pixel_origin[ 1 ] ),
                    pixel_size = requested_pixel_size,
                    data = image,
                ) )

                x += requested_pixel_size[ 0 ]
        else:
            if self.palette is not None:
                raw_data = gdal_array.BandReadAsArray(
                    self.raster_bands[ 0 ],
                    pixel_origin[ 0 ],
                    pixel_origin[ 1 ],
                    pixel_size[ 0 ],
                    pixel_size[ 1 ],
                    pixel_size[ 0 ],
                    pixel_size[ 1 ],
                )

                image = Bitmap.paletted_to_rgba(
                    raw_data, self.palette, self.DEFAULT_ALPHA,
                )
            else:
                image = np.zeros(
                    ( pixel_size[ 1 ], pixel_size[ 0 ], 4 ),
                    np.uint8,
                )
                image[ :, :, 3 ] = self.DEFAULT_ALPHA

                for band_index in range( 0, self.raster_count ):
                    raw_data = gdal_array.BandReadAsArray(
                        self.raster_bands[ band_index ],
                        pixel_origin[ 0 ],
                        pixel_origin[ 1 ],
                        pixel_size[ 0 ],
                        pixel_size[ 1 ],
                        pixel_size[ 0 ],
                        pixel_size[ 1 ],
                    )

                    image[ :, :, band_index ] = raw_data

            image = Bitmap.expand(
                image,
                requested_pixel_size[ 0 ],
                requested_pixel_size[ 1 ],
            )

            images = [ dict(
                pixel_origin = pixel_origin,
                pixel_size = requested_pixel_size,
                data = image,
            ) ]

        response_box.send(
            request = "data",
            data = images,
        )

        pixbuf = None
        scaled_pixbuf = None

    def get_properties( self, response_box, indices = None ):
        response_box.send(
            request = "properties",
            properties = (
                self.name,
            )
        )

    def calculate_geo_transform( self ):
        # Only bother to calculate the raster's geographic transform if GDAL
        # reports it as the default transform. Otherwise, use the transform
        # that GDAL provides.
        geo_transform = self.dataset.GetGeoTransform()

        DEFAULT_TRANSFORM = ( 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 )
        if geo_transform == DEFAULT_TRANSFORM:
            geo_transform = gdal.GCPsToGeoTransform( self.dataset.GetGCPs() )

        return np.array( geo_transform )

    def pixel_to_geo( self, pixel_point ):
        # Raster pixel coordinates are flipped vertically with respect to
        # geographic coordinates.
        pixel_point = (
            pixel_point[ 0 ],
            self.pixel_size[ 1 ] - pixel_point[ 1 ],
        )

        geo_point = \
            self.apply_transform( pixel_point, self.pixel_to_geo_transform )

        return geo_point

    def geo_to_pixel( self, geo_point ):
        pixel_point = \
            self.apply_transform( geo_point, self.geo_to_pixel_transform )

        return (
            min(
                max( int( pixel_point[ 0 ] ), 0 ),
                int( self.pixel_size[ 0 ] ),
            ),
            min(
                max( int( pixel_point[ 1 ] ), 0 ),
                int( self.pixel_size[ 1 ] ),
            ),
        )

    def box_pixel_to_geo( self, pixel_origin, pixel_size ):
        geo_upper_left = self.apply_transform(
            pixel_origin,
            self.pixel_to_geo_transform,
        )
        geo_lower_right = self.apply_transform(
            (
                pixel_origin[ 0 ] + pixel_size[ 0 ],
                pixel_origin[ 1 ] + pixel_size[ 1 ],
            ),
            self.pixel_to_geo_transform,
        )

        geo_origin = (
            geo_upper_left[ 0 ],
            geo_lower_right[ 1 ],
        )
        geo_size = (
            math.fabs( geo_lower_right[ 0 ] - geo_upper_left[ 0 ] ),
            math.fabs( geo_lower_right[ 1 ] - geo_upper_left[ 1 ] ),
        )

        return ( geo_origin, geo_size )

    @staticmethod
    def apply_transform( point, transform ):
        return (
            transform[ 0 ] +
                point[ 0 ] * transform[ 1 ] +
                point[ 1 ] * transform[ 2 ],
            transform[ 3 ] +
                point[ 0 ] * transform[ 4 ] +
                point[ 1 ] * transform[ 5 ],
        )

    @staticmethod
    def invert_geo_transform( geo_transform ):
        # TODO: Convert this to use numpy.linalg.inv() instead.
        det = geo_transform[ 1 ] * geo_transform[ 5 ] - \
              geo_transform[ 2 ] * geo_transform[ 4 ]

        if det == 0:
            raise Transformation_error()

        inv_det = 1.0 / det
        inverted = np.array( [ 0.0 ] * 6 )

        inverted[ 1 ] =  geo_transform[ 5 ] * inv_det;
        inverted[ 4 ] = -geo_transform[ 4 ] * inv_det;

        inverted[ 2 ] = -geo_transform[ 2 ] * inv_det;
        inverted[ 5 ] =  geo_transform[ 1 ] * inv_det;

        inverted[ 0 ] = (
            geo_transform[ 2 ] * geo_transform[ 3 ] -
            geo_transform[ 0 ] * geo_transform[ 5 ]
        ) * inv_det;

        inverted[ 3 ] = (
            -geo_transform[ 1 ] * geo_transform[ 3 ] +
            geo_transform[ 0 ] * geo_transform[ 4 ]
        ) * inv_det;

        return inverted

    def find_duplicates( self, distance_tolerance, depth_tolerance,
                         response_box, layer = None ):
        response_box.send(
            exception = NotImplementedError(
                "Raster layers do not support finding duplicate points.",
            ),
        )

    def triangulate( self, transformer, response_box ):
        response_box.send(
            exception = NotImplementedError(
                "Raster layers do not support triangulation.",
            ),
        )

    def save( self, filename, saver, response_box, layer = None ):
        response_box.send(
            exception = NotImplementedError(
                "Saving raster files is not yet implemented.",
            ),
        )

    def layer_removed( self, layer, parent = None ):
        # We're only interested in our own removal.
        if layer != self:
            return

        self.tiles_layer.inbox.send(
            request = "clear_cache",
        )
