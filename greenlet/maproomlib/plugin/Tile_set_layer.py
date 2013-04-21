import time
import math
import heapq
import numpy as np
import pyproj
import maproomlib.utility as utility


class Stop_error( Exception ):
    pass


class Tile_set_layer:
    """
    A set of raster tiles. Given an underlying raster layer to draw data from,
    a Tile_set_layer breaks up a request for a particular area into a series
    of individual tiles.

    :param name: textual name of the layer
    :type name: str
    :param layer: source layer to fetch raster data from, usually the parent
                  layer
    :type layer: raster layer
    :param projection: geographic projection of the given raster layer
    :type projection: pyproj.Proj
    :param size: dimensions of the entire raster layer in geographic
                 coordinates
    :type size: (float, float)
    :param pixel_size: dimensions of the entire raster layer in raster pixel
                       coordinates
    :type pixel_size: (int, int)

    .. attribute:: name

        name of the layer

    .. attribute:: projection

        geographic projection of the data (pyproj.Proj)

    .. attribute:: size

        dimensions of the entire raster layer in geographic coordinates

    .. attribute:: pixel_size

        dimensions of the entire raster layer in raster pixel coordinates

    .. attribute:: children

        list of child layers

    .. function:: get_tiles( origin, size, scale_factor, response_box )

        When a message with ``request = "get_tiles"`` is received within the
        :attr:`inbox`, a handler retrieves the requested data as a series of
        bitmap tiles from the raster layer, and sends each in a separate
        message to the given ``response_box``.

        :param origin: lower-left corner of viewport in geographic coordinates
        :type origin: ( float, float )
        :param size: dimensions of viewport in geographic coordinates
        :type size: ( float, float )
        :param scale_factor: linear factor to transform from raster pixel
                             coordinates to viewport pixel coordinates
        :type scale_factor: float
        :param response_box: response messages sent here
        :type response_box: Inbox

        Before data is requested from the underlying raster layer as part of
        servicing a ``get_tiles`` request, a ``start_progress`` message is
        sent to the :attr:`outbox` as follows::

            outbox.send( request = "start_progress", id, message )

        The ``id`` parameter is a unique identifier for the layer's file, and
        ``message`` is a textual message describing the loading.

        And when the tile cache is done requesting tiles as part of servicing
        a particular ``get_tiles`` request, an ``end_progress`` message is
        sent to the :attr:`outbox` as follows::

            outbox.send( request = "end_progress", id )

        The ``id`` parameter is a unique identifier for the layer's file.

        Note that these ``start_progress`` and ``end_progress`` messages are
        only sent if tiles are loaded from the underlying raster layer. If the
        cache can handle all tile requests on its own, then no progress
        messages are issued.

        Between those start and end messages, the tile cache sends a series
        of ``get_data`` requests to the raster layer, one per tile::

            layer.send( request = "get_data", origin, size, scale_factor, response_box )

        Note that a request for a particular tile is skipped if that tile
        is present in the cache.

        Each retrieved tile, either from the layer or the cache, is sent in a
        separate message as follows::

            response_box.send( request = "tile", data, projection, origin, size, scale_factor )

        That data that is sent to the ``response_box`` is an HxWx4 RGBA numpy
        array. ``origin`` is the lower-left corner of the data in geographic
        coordinates, while ``size`` is the dimensions of the data in
        geographic coordinates.

        ``projection`` is the geographic projection of the provided ``origin``
        and ``scale``.

        If a projection was provided to ``get_tiles``, then the returned
        ``origin`` and ``size`` are transformed to the requested projection.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "tile_set"
    MAX_CACHE_TILES = 500
    MAX_DISK_CACHE_TILES = 4000

    def __init__( self, name, layer, size, pixel_size, projection ):
        self.name = name
        self.layer = layer
        self.projection = projection
        self.size = size
        self.pixel_size = pixel_size
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()

        # Start out with the disk cache recording all set() calls but not
        # writing them to disk. Until the tiles are fully loaded and
        # pyramided, we want to prevent contention for disk I/O.
        self.disk_cache = utility.Disk_tile_cache(
            self.MAX_DISK_CACHE_TILES,
            pause_writes = True,
        )

        self.cache = utility.Cache(
            self.MAX_CACHE_TILES,
            backing_cache = self.disk_cache,
        )

    TILE_SIZE = 256

    def run( self, scheduler ):
        scheduler.add( self.disk_cache.bulk_setter.run )

        while True:
            message = self.inbox.receive( request = (
                "get_tiles", "stop", "clear_cache",
            ) )

            request = message.pop( "request" )
            if request == "stop":
                continue
            if request == "clear_cache":
                self.cache.clear()
                continue

            response_box = message.pop( "response_box" )

            self.disk_cache.pause_writes()

            # Break up the request into individual tiles.
            tiles = self.tile( **message )
            cache_miss = False
            stopped = False

            for tile in tiles:
                try:
                    ( tile, cache_miss ) = self.fetch_tile_data(
                        cache_miss = cache_miss,
                        **tile
                    )
                except Stop_error:
                    stopped = True
                    break

                if tile is None: continue

                if tile.get( "origin" ) is None:
                    self.tile_pixel_to_geo( tile )

                response_box.send( request = "tile", tile = tile )
                scheduler.switch()

            if cache_miss is True:
                self.outbox.send(
                    request = "end_progress",
                    id = self.name,
                )

            if not stopped:
                self.disk_cache.unpause_writes()

    def tile( self, origin, size, scale_factor ):
        tiles = []
        tile_pixel_size = int( self.TILE_SIZE / scale_factor )

        pixel_lower_left = self.layer.geo_to_pixel( origin )
        pixel_upper_right = self.layer.geo_to_pixel( (
            origin[ 0 ] + size[ 0 ],
            origin[ 1 ] + size[ 1 ],
        ) )

        # Note: The geographic coordinate origin is in the lower-left, and
        # the geographic origin + size is in the upper-right. But the raster
        # pixel coordinates are flipped vertically with respect to geographic
        # coordinates. So the upper-left origin becomes the lower-left. And
        # the lower-right origin + size becomes the upper-right.
        pixel_origin = ( # upper left corner
            pixel_lower_left[ 0 ],
            pixel_upper_right[ 1 ],
        )
        pixel_size = (
            abs( pixel_upper_right[ 0 ] - pixel_lower_left[ 0 ] ),
            abs( pixel_upper_right[ 1 ] - pixel_lower_left[ 1 ] ),
        )

        viewport_pixel_center = (
            pixel_origin[ 0 ] + pixel_size[ 0 ] // 2,
            pixel_origin[ 1 ] + pixel_size[ 1 ] // 2,
        )

        # Round the pixel origin down to the nearest tile boundary.
        pixel_origin = (
            pixel_origin[ 0 ] // tile_pixel_size * tile_pixel_size,
            pixel_origin[ 1 ] // tile_pixel_size * tile_pixel_size,
        )

        # Round the pixel size up to the nearest tile boundary.
        pixel_size = (
            ( pixel_size[ 0 ] + tile_pixel_size - 1 )
                // tile_pixel_size * tile_pixel_size,
            ( pixel_size[ 1 ] + tile_pixel_size - 1 )
                // tile_pixel_size * tile_pixel_size,
        )

        # Make a list of tiles to fetch, sorted by distance from the center of
        # the viewport.
        for tile_y in range(
            pixel_origin[ 1 ],
            pixel_origin[ 1 ] + pixel_size[ 1 ] + tile_pixel_size,
            tile_pixel_size,
        ):
            for tile_x in range(
                pixel_origin[ 0 ],
                pixel_origin[ 0 ] + pixel_size[ 0 ] + tile_pixel_size,
                tile_pixel_size,
            ):
                tile_center = (
                    tile_x + tile_pixel_size // 2,
                    tile_y + tile_pixel_size // 2,
                )

                distance_from_center = math.sqrt(
                    math.pow( viewport_pixel_center[ 0 ] - tile_center[ 0 ], 2 ) +
                    math.pow( viewport_pixel_center[ 1 ] - tile_center[ 1 ], 2 )
                )

                tile = dict(
                    pixel_origin = ( tile_x, tile_y ),
                    scale_factor = scale_factor,
                )
                heapq.heappush( tiles, ( distance_from_center, tile ) )

        return [ tile for ( distance_from_center, tile ) in tiles ]

    def fetch_tile_data( self, pixel_origin, scale_factor, cache_miss = False ):
        import maproomlib.utility.Bitmap as Bitmap

        pixel_size = int( self.TILE_SIZE / scale_factor )

        # If the entire tile is outside the borders of the raster, bail.
        if pixel_origin[ 0 ] >= self.pixel_size[ 0 ] or \
           pixel_origin[ 1 ] >= self.pixel_size[ 1 ] or \
           pixel_origin[ 0 ] + pixel_size < 0 or \
           pixel_origin[ 1 ] + pixel_size < 0:
            return ( None, cache_miss )

        # Look for the tile's data in the cache.
        key = (
            self.layer.unique_id,
            pixel_origin,
            scale_factor,
        )
        tile = self.cache.get( key, None )

        # If there's a cache hit, we're done.
        if tile is not None:
            #print "cache hit:", scale_factor, ( pixel_origin[ 0 ] / 256.0, pixel_origin[ 1 ] / 256.0 )
            return ( tile, cache_miss )

        #print "cache miss:", scale_factor, ( pixel_origin[ 0 ] / 256.0, pixel_origin[ 1 ] / 256.0 )
        if cache_miss is False:
            cache_miss = True
            self.outbox.send(
                request = "start_progress",
                id = self.name,
                message = "Loading %s" % self.name,
            )

        # Otherwise, if the scale factor is not 1.0, then grab tiles from the
        # next level and scale them down to make a new tile. Recurse as
        # necessary. This general technique is known as pyramiding.
        if scale_factor < 1.0:
            image = None # Composited image of the inner tiles.

            def paste_inner_tile( inner_pixel_offset, image, cache_miss ):
                ( offset_x, offset_y ) = inner_pixel_offset
                inner_scale_factor = scale_factor * 2.0

                inner_pixel_origin = (
                    pixel_origin[ 0 ] +
                        int( offset_x / inner_scale_factor ),
                    pixel_origin[ 1 ] +
                        int( offset_y / inner_scale_factor ),
                )

                ( inner_tile, cache_miss ) = self.fetch_tile_data(
                    inner_pixel_origin,
                    inner_scale_factor,
                    cache_miss,
                )
                if inner_tile is None or inner_tile.get( "data" ) is None:
                    return image

                if image is None:
                    doubled_size = self.TILE_SIZE * 2
                    image = np.zeros(
                        ( doubled_size, doubled_size, 4 ),
                        dtype = np.uint8,
                    )

                inner_data = inner_tile.get( "data" )
                image[
                    offset_y: offset_y + inner_data.shape[ 0 ],
                    offset_x: offset_x + inner_data.shape[ 1 ],
                    :,
                ] = inner_data

                return image

            image = paste_inner_tile(
                ( 0, 0 ),
                image, cache_miss,
            )
            image = paste_inner_tile(
                ( 0, self.TILE_SIZE ),
                image, cache_miss,
            )
            image = paste_inner_tile(
                ( self.TILE_SIZE, 0 ),
                image, cache_miss,
            )
            image = paste_inner_tile(
                ( self.TILE_SIZE, self.TILE_SIZE ),
                image, cache_miss,
            )

            if image is None:
                return ( None, cache_miss )

            # Finally, scale down the composite image.
            image = Bitmap.scale_half( image )

            tile = dict(
                data = image,
                pixel_origin = pixel_origin,
                pixel_size = ( pixel_size, pixel_size ),
                scale_factor = scale_factor,
            )
            #image.save( "test_%s_%s.png" % ( scale_factor, id( image ) ) )

            self.cache.set( key, tile )
            return ( tile, cache_miss )

        # Scale factor is 1.0, so we can request data directly from the raster
        # layer.
        self.layer.inbox.send(
            request = "get_data",
            pixel_origin = pixel_origin,
            pixel_size = ( self.TILE_SIZE, self.TILE_SIZE ),
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = ( "data", "stop" ) )
        if message.pop( "request" ) == "stop":
            # We've received a stop request, so throw away any pending data.
            self.layer.inbox.discard( request = "get_data" )
            self.inbox.discard( request = "data" )
            raise Stop_error()

        if not message.get( "data" ):
            return ( None, cache_miss )

        # We might get back more tiles than we requested (e.g. if it was more
        # efficient to read them in all at once). Store all of them in the
        # cache and return the one requested.
        requested_tile = None
        for tile in message.get( "data" ):
            key = (
                self.layer.unique_id,
                tile.get( "pixel_origin" ),
                scale_factor,
            )
            tile[ "scale_factor" ] = scale_factor

            self.cache.set( key, tile )
            if tile.get( "pixel_origin" ) == pixel_origin:
                requested_tile = tile

        if requested_tile is None:
            raise RuntimeError( "Did not receive the tile requested." )

        return ( requested_tile, cache_miss )

    def tile_pixel_to_geo( self, tile ):
        ( geo_origin, geo_size ) = self.layer.box_pixel_to_geo(
            tile[ "pixel_origin" ],
            tile[ "pixel_size" ],
        )

        tile[ "projection" ] = self.projection
        tile[ "origin" ] = geo_origin
        tile[ "size" ] = geo_size

    def box_pixel_to_geo( self, pixel_origin, pixel_size ):
        return self.layer.box_pixel_to_geo( pixel_origin, pixel_size )
