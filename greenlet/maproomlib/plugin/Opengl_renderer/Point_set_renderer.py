import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Point_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering point_set layers.
    """
    PICKER_POINT_SIZE = 12.0      # in pixels
    POINT_RESIZE_THRESHOLD = 10.0
    RENDER_THRESHOLD_FACTOR = 10000
    RENDER_COUNT_THRESHOLD = 20
    POINTS_XY_DTYPE = np.dtype( [
        ( "xy", "2f4" ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )             

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
        self.vertex_buffer = None
        self.vertex_count = 0
        self.vertex_capacity = 0
        self.color_buffer = None
        self.pick_color_buffer = None
        self.point_size = 1.0
        self.shown_indices = None
        self.shown_count = 0
        self.render_threshold = None
        self.logger = logging.getLogger( __name__ )

        gl.glEnable( gl.GL_POINT_SMOOTH )
        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

    def run( self, scheduler ):
        self.layer.outbox.subscribe(
            self.transformer( self.inbox ),
            request = (
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "points_added",
                "line_points_added",
                "points_nuked",
                "size_changed",
                "shown_updated",
            ),
        )
        self.fetch_points()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "points",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "points_added",
                    "line_points_added",
                    "points_nuked",
                    "projection_changed",
                    "size_changed",
                    "shown_updated",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "update":
                self.inbox.discard( request = "update" )
            elif request == "points":
                unique_id = "render points %s" % self.layer.name
                count = message.get( "count" )

                if count > 1:
                    self.outbox.send(
                        request = "start_progress",
                        id = unique_id,
                        message = "Initializing points"
                    )
                try:
                    self.init_points(
                        message.get( "points" ),
                        count,
                        message.get( "point_size" ),
                        message.get( "origin" ),
                        message.get( "size" ),
                        message.get( "shown_indices" ),
                        message.get( "shown_count" ),
                    )
                finally:
                    if count > 1:
                        self.outbox.send(
                            request = "end_progress",
                            id = unique_id,
                        )
            elif request in ( "points_updated", "points_undeleted" ):
                self.update_points( **message )
            elif request == "points_deleted":
                self.delete_points( **message )
            elif request in ( "points_added", "line_points_added" ) and \
                 message.get( "count" ) > 0:
                self.add_points( **message )

                import maproomlib.plugin as plugin
                if isinstance( self.layer, plugin.Point_set_layer ):
                    object_index = message.get( "selected_end_index" )
                    if object_index is None:
                        object_index = message.get( "start_index" )

                    self.root_layer.inbox.send(
                        request = "replace_selection",
                        layer = self.layer,
                        object_indices = ( object_index, ),
                    )
            elif request == "projection_changed":
                self.fetch_points()

                message = self.inbox.receive( request = "points" )
                points = message.get( "points" )

                self.update_points(
                    layer = self.layer,
                    points = points,
                    projection = message.get( "projection" ),
                    indices = None,
                    count = message.get( "count" ),
                )
            elif request == "size_changed":
                size = message.get( "size" )

                if size is not None and \
                   self.vertex_count >= self.RENDER_COUNT_THRESHOLD:
                    self.render_threshold = max( size ) * \
                        self.RENDER_THRESHOLD_FACTOR
            elif request == "shown_updated":
                self.shown_indices = message.get( "shown_indices" )
                self.shown_count = message.get( "shown_count" )
                self.opengl_renderer.Refresh( False )
            elif request == "close":
                self.layer.outbox.unsubscribe( self.inbox )
                self.layer = None
                return

    def fetch_points( self ):
        self.layer.inbox.send(
            request = "get_points",
            origin = self.viewport.geo_origin( self.layer.projection ),
            size = self.viewport.geo_size( self.layer.projection ),
            response_box = self.transformer( self.inbox ),
        )

    def init_points( self, points, count, point_size, origin, size,
                     shown_indices, shown_count ):
        self.point_size = point_size
        self.shown_indices = shown_indices
        self.shown_count = shown_count
        self.vertex_count = count

        # Make the threshold at which this layer is rendered based on its
        # geographic size, but record that threshold in arbitrary reference
        # render units.
        if size is not None and \
           self.vertex_count >= self.RENDER_COUNT_THRESHOLD:
            self.render_threshold = max( size ) * \
                self.RENDER_THRESHOLD_FACTOR

        self.vertex_buffer = gl_vbo.VBO(
            points.view( self.POINTS_XY_DTYPE ).xy.copy(),
        )
        self.vertex_count = count
        self.vertex_capacity = len( points )
        self.color_buffer = gl_vbo.VBO(
            points.color.copy().view( dtype = np.uint8 )
        )
        self.pick_color_buffer = gl_vbo.VBO(
            np.zeros( ( len( points ) * 4, ), dtype = np.uint8 ),
        )

        self.opengl_renderer.Refresh( False )

    def update_points( self, layer, points, projection, indices, count = None,
                       undo_recorded = None ):
        if self.vertex_buffer is None:
            return

        if indices is None and count is not None:
            self.vertex_buffer[ 0 : count ] = \
                points.view( self.POINTS_XY_DTYPE ).xy[
                    0 : count
                ].copy()

            self.color_buffer[ 0 : count * 4 ] = \
                points.color[
                    0 : count
                ].copy().view( dtype = np.uint8 )

            self.vertex_count = max( self.vertex_count, count )
        else:
            for index in indices:
                self.vertex_buffer[ index : index + 1 ] = \
                    points.view( self.POINTS_XY_DTYPE ).xy[
                        index : index + 1
                    ].copy()

                self.color_buffer[ index * 4 : index * 4 + 4 ] = \
                    points.color[
                        index : index + 1
                    ].copy().view( dtype = np.uint8 )

                self.vertex_count = max( self.vertex_count, index )

        self.opengl_renderer.Refresh( False )

    def add_points( self, layer, points, projection, start_index, count,
                    *args, **kwargs ):
        if len( points ) > self.vertex_capacity:
            new_capacity = len( points )
            self.logger.debug(
                "Growing points VBOs from %d points capacity to %d." % (
                    self.vertex_capacity, new_capacity,
                ),
            )

            shape = list( self.vertex_buffer.data.shape )
            shape[ 0 ] = new_capacity
            self.vertex_buffer = gl_vbo.VBO(
                np.resize( self.vertex_buffer.data, tuple( shape ) )
            )

            shape = list( self.color_buffer.data.shape )
            shape[ 0 ] = new_capacity * 4
            self.color_buffer = gl_vbo.VBO(
                np.resize( self.color_buffer.data, tuple( shape ) )
            )

            self.pick_color_buffer = gl_vbo.VBO(
                np.resize( self.pick_color_buffer.data, tuple( shape ) )
            )

            self.vertex_capacity = new_capacity

        self.vertex_buffer[ start_index : start_index + count ] = \
            points.view( self.POINTS_XY_DTYPE ).xy[
                start_index : start_index + count
            ].copy()

        self.color_buffer[ start_index * 4 : ( start_index + count ) * 4 ] = \
            points.color[
                start_index : start_index + count
            ].copy().view( dtype = np.uint8 )

        self.vertex_count = max( self.vertex_count, start_index + count )

        self.opengl_renderer.Refresh( False )

    def delete_points( self, layer, points, projection, indices,
                       undo_recorded = None ):
        nans = np.empty(
            ( 1, 2 ),
            dtype = np.float32,
        )
        nans.fill( np.nan )

        for index in indices:
            self.vertex_buffer[ index : index + 1 ] = nans
            self.vertex_count = max( self.vertex_count, index )

        self.opengl_renderer.Refresh( False )

    def render( self, pick_mode = False ):
        if self.vertex_buffer is None or self.vertex_count == 0:
            return

        reference_render_length = self.viewport.reference_render_length()
        threshold_ratio = 1.0

        if self.render_threshold is not None and \
           self.shown_indices is None and \
           reference_render_length > self.render_threshold and \
           self.point_size < self.POINT_RESIZE_THRESHOLD:
            # If the user is zoomed out enough, then reduce the point size.
            # This makes the map easier to read when there are a lot of
            # points.
            threshold_ratio = \
                    self.render_threshold / float( reference_render_length )
            threshold_ratio = max( min( threshold_ratio, 0.75 ), 0.25 )

        def draw():
            if self.shown_indices is not None:
                if self.shown_count > 0:
                    gl.glDrawElements(
                        gl.GL_POINTS,
                        self.shown_count,
                        gl.GL_UNSIGNED_INT,
                        self.shown_indices,
                    )
            else:
                gl.glDrawArrays( gl.GL_POINTS, 0, self.vertex_count )

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        # To make the points stand out better, especially when rendered on top
        # of line segments, draw translucent white halos under them.
        if not pick_mode and self.point_size < 15.0:
            gl.glPointSize( self.point_size * threshold_ratio * 2.5 )
            gl.glColor( 1, 1, 1, 0.5 )
            draw()
            gl.glColor( 1, 1, 1, 1 )

        if pick_mode:
            self.picker.bind_pick_colors(
                self.pick_color_buffer, self, self.vertex_count,
            )
            gl.glDisable( gl.GL_BLEND )
            gl.glDisable( gl.GL_POINT_SMOOTH )

            gl.glPointSize( self.PICKER_POINT_SIZE )
        else:
            self.picker.bind_colors(
                self.color_buffer, self, self.vertex_count,
            )
            gl.glPointSize( self.point_size * threshold_ratio )

        draw()

        if pick_mode:
            self.picker.unbind_colors( self.pick_color_buffer )
            gl.glEnable( gl.GL_BLEND )
            gl.glEnable( gl.GL_POINT_SMOOTH )
        else:
            self.picker.unbind_colors( self.color_buffer )

        self.vertex_buffer.unbind()

    def delete( self ):
        if self.vertex_buffer:
            self.vertex_buffer = None
