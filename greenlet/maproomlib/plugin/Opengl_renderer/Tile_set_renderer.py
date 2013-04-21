import weakref
import OpenGL.GL as gl
import maproomlib.ui as ui
import maproomlib.utility as utility
from maproomlib.plugin.Opengl_renderer.Raster_tile import Raster_tile


class Tile_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering tile_set layers.
    """
    MAX_CACHE_TILES = 500

    def __init__( self, root_layer, layer, viewport, opengl_renderer,
                  transformer, picker ):
        self.root_layer = root_layer
        self.layer = layer
        self.viewport = viewport
        self.inbox = ui.Wx_inbox()
        self.outbox = utility.Outbox()
        self.opengl_renderer = opengl_renderer
        self.transformer = transformer
        self.picker = picker
        self.current_scale_factor = None
        self.previous_scale_factor = None
        self.selected = True

        self.cache = utility.Cache( self.MAX_CACHE_TILES )
        self.tiles = {} # zoom level -> set of Raster_tile weakrefs

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = "selection_updated",
        )

        self.current_scale_factor = self.scale_factor
        self.fetch_tiles()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "tile",
                    "projection_changed",
                    "selection_updated",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "update" and len( self.tiles ) > 0:
                force = message.get( "force" )
                self.inbox.discard( request = "update", force = force )

                self.reap_tiles()

                if force:
                    current_tiles = \
                        self.tiles.get( self.current_scale_factor, () )
                    if len( current_tiles ) > 0:
                        self.previous_scale_factor = self.current_scale_factor
                    self.current_scale_factor = self.scale_factor

                self.fetch_tiles()
            elif request == "tile":
                self.add_tile( message.get( "tile" ) )
            elif request == "projection_changed":
                # TODO: Handle a projection change correctly.
                self.inbox.discard( request = "projection_changed" )
            elif request == "selection_updated":
                self.selection_updated( **message )
            elif request == "close":
                self.delete()
                return

    def fetch_tiles( self ):
        self.layer.inbox.discard( request = "get_tiles" )
        self.layer.inbox.discard( request = "stop" )
        self.layer.inbox.send( request = "stop" )
        self.inbox.discard( request = "tile" )

        self.layer.inbox.send(
            request = "get_tiles",
            origin = self.viewport.geo_origin( self.layer.projection ),
            size = self.viewport.geo_size( self.layer.projection ),
            scale_factor = self.current_scale_factor,
            response_box = self.inbox,
        )

    def add_tile( self, tile ):
        key = (
            tile.get( "origin" ),
            tile.get( "scale_factor" ),
        )

        raster_tile = self.cache.get( key, None )
        if raster_tile is None:
            raster_tile = Raster_tile(
                transformer = self.transformer, **tile
            )
            self.cache.set( key, raster_tile )

        bucket = self.tiles.get( raster_tile.scale_factor )

        if bucket is None:
            self.tiles[ raster_tile.scale_factor ] = set( [
                weakref.ref( raster_tile ),
            ] )
        else:
            if raster_tile in bucket:
                self.opengl_renderer.Refresh( False )
                return

            bucket.add( weakref.ref( raster_tile ) )

        self.opengl_renderer.Refresh( False )

    def reap_tiles( self ):
        """
        Remove all tiles whose weak references are no longer good.
        """
        for ( scale_factor, bucket ) in self.tiles.items():
            for tile_ref in list( bucket ):
                tile = tile_ref()

                if tile is None:
                    bucket.remove( tile_ref )

    def render( self, pick_mode = False ):
        # If there's a previous scale factor and it's not the same as the
        # current scale factor, then render its tiles too. This makes zooming
        # nicer, as tiles from the previous zoom level are shown for a split
        # second before the tiles for the current level replace them.
        if self.previous_scale_factor and \
           self.previous_scale_factor != self.current_scale_factor:
            factors = [ self.previous_scale_factor, self.current_scale_factor ]
        else:
            factors = [ self.current_scale_factor ]

        if self.current_scale_factor:
            fade_factor = 2 ** self.current_scale_factor * 0.4
        else:
            fade_factor = 0.0

        gl.glEnable( gl.GL_TEXTURE_2D )
        for scale_factor in factors:
            bucket = self.tiles.get( scale_factor )

            if bucket is None:
                continue

            for tile_ref in bucket:
                tile = tile_ref()

                if tile is not None:
                    tile.render(
                        pick_mode,
                        faded = not self.selected
                                and self.current_scale_factor >= 0.125,
                        fade_factor = fade_factor,
                    )

                    # Access the cache in order to update the tile's access
                    # time.
                    key = (
                        tile.origin,
                        tile.scale_factor,
                    )
                    self.cache.get( key, None )

        gl.glDisable( gl.GL_TEXTURE_2D )

    def selection_updated( self, selections, **other ):
        for ( layer, indices ) in selections:
            if hasattr( layer, "wrapped_layer" ) and \
               hasattr( layer.wrapped_layer, "tiles_layer" ) and \
               layer.wrapped_layer.tiles_layer == self.layer:
                self.selected = True
                self.opengl_renderer.Refresh( False )
                return
        
        self.selected = False
        self.opengl_renderer.Refresh( False )

    def delete( self ):
        """
        Remove all tiles.
        """
        self.layer = None

        for ( scale_factor, bucket ) in self.tiles.items():
            for tile_ref in bucket:
                tile = tile_ref()

                if tile is not None:
                    tile.delete()
            bucket.clear()

        self.cache.clear()

    def get_scale_factor( self ):
        pixel_size = self.viewport.pixel_size
        if pixel_size == ( 0, 0 ):
            pixel_size = self.opengl_renderer.GetClientSize()

        viewport_size = self.viewport.geo_size( self.layer.projection )

        zoom_level = (
            viewport_size[ 0 ] / self.layer.size[ 0 ],
            viewport_size[ 1 ] / self.layer.size[ 1 ],
        )

        scale_factor = min(
            ( pixel_size[ 0 ] / float( self.layer.pixel_size[ 0 ] ) ) /
                zoom_level[ 0 ],
            ( pixel_size[ 1 ] / float( self.layer.pixel_size[ 1 ] ) ) /
                zoom_level[ 1 ],
        )

        # Round the scale factor down to the nearest inverse power of two.
        new_scale_factor = 1.0
        while scale_factor < new_scale_factor * 0.5:
            new_scale_factor *= 0.5

        return new_scale_factor

    scale_factor = property( get_scale_factor )
