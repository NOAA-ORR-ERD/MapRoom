import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Line_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering line_set layers. These represent individual line segments that
    may or may not be connected.
    """
    PICKER_LINE_WIDTH = 6.0   # in pixels
    LINE_RESIZE_THRESHOLD = 12.0
    RENDER_THRESHOLD_FACTOR = 17000
    POINTS_XY_DTYPE = np.dtype( [
        ( "xy", "2f4" ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )             
    SEGMENT_POINTS_DTYPE = np.dtype(
        [ ( "x", np.float32 ), ( "y", np.float32 ) ],
    )
    SEGMENT_POINT_PAIRS_DTYPE = np.dtype(
        [ ( "x1", np.float32 ), ( "y1", np.float32 ),
          ( "x2", np.float32 ), ( "y2", np.float32 ) ],
    )
    LINES_POINTS_DTYPE = np.dtype( [
        ( "points", "2u4" ),
        ( "type", np.uint32 ),
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
        self.deleted_points = {} # buffer point index -> deleted point data
        self.line_width = 1.0
        self.color_buffer = None
        self.pick_color_buffer = None
        self.render_threshold = None
        self.logger = logging.getLogger( __name__ )

        self.nans = np.empty(
            1, dtype = self.SEGMENT_POINTS_DTYPE
        ).view( np.recarray )
        self.nans.x[ 0 ] = np.nan
        self.nans.y[ 0 ] = np.nan

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE );

    def run( self, scheduler ):
        self.layer.outbox.subscribe(
            self.transformer( self.inbox ),
            request = (
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "points_added",
                "lines_added",
                "lines_deleted",
                "lines_undeleted",
                "size_changed",
            ),
        )
        self.fetch_lines()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "lines",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "points_added",
                    "lines_added",
                    "lines_deleted",
                    "lines_undeleted",
                    "projection_changed",
                    "size_changed",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "update":
                self.inbox.discard( request = "update" )
            elif request == "lines":
                unique_id = "render lines %s" % self.layer.name
                self.outbox.send(
                    request = "start_progress",
                    id = unique_id,
                    message = "Initializing boundaries"
                )
                try:
                    self.init_lines(
                        message.get( "points" ),
                        message.get( "lines" ),
                        message.get( "line_count" ),
                        message.get( "line_width" ),
                        message.get( "origin" ),
                        message.get( "size" ),
                    )
                finally:
                    self.outbox.send(
                        request = "end_progress",
                        id = unique_id,
                    )
            elif request in ( "points_updated", "points_undeleted" ):
                self.update_points( **message )
            elif request == "points_added":
                self.add_points( **message )
            elif request == "points_deleted":
                self.delete_points( **message )
            elif request in ( "lines_added", "lines_undeleted" ):
                self.add_lines( **message )
            elif request == "lines_deleted":
                self.delete_lines( **message )
            elif request == "projection_changed":
                self.fetch_lines()

                message = self.inbox.receive( request = "lines" )
                points = message.get( "points" )

                self.update_points(
                    layer = self.layer,
                    points = points,
                    projection = message.get( "projection" ),
                    indices = None,
                    point_map = message.get( "point_map" ),
                    count = message.get( "point_count" ),
                )
            elif request == "size_changed":
                size = message.get( "size" )

                if size is not None:
                    self.render_threshold = max( size ) * \
                        self.RENDER_THRESHOLD_FACTOR
            elif request == "close":
                self.layer = None
                return

        self.layer.outbox.unsubscribe( self.inbox )

    def fetch_lines( self ):
        self.layer.inbox.send(
            request = "get_lines",
            origin = self.viewport.geo_origin( self.layer.projection ),
            size = self.viewport.geo_size( self.layer.projection ),
            response_box = self.transformer( self.inbox ),
        )

    def init_lines( self, points, lines, line_count, line_width, origin,
                    size ):
        self.line_width = line_width
        vertex_count = line_count * 2

        # Make the threshold at which this layer is rendered based on its
        # geographic size, but record that threshold in arbitrary reference
        # render units.
        if size is not None:
            self.render_threshold = max( size ) * \
                self.RENDER_THRESHOLD_FACTOR

        # OpenGL doesn't allow a given vertex to have multiple colors
        # simultaneously. So this code is needed to make each line segment in
        # a line a different color. It prepares an array with duplicated
        # vertices to hold the individual line segments.
        points_xy = points.view( self.POINTS_XY_DTYPE )[ "xy" ]
        segment_points = points_xy[
            lines.view( self.LINES_POINTS_DTYPE )[ "points" ].reshape( -1 )
        ].reshape( -1 ).view( self.SEGMENT_POINTS_DTYPE ).view( np.recarray )

        segment_colors = np.c_[ lines.color, lines.color ].reshape( -1 )

        segment_pairs = segment_points.view(
            self.SEGMENT_POINT_PAIRS_DTYPE,
        )

        self.deleted_points = dict( [
            ( buffer_index + 1, segment_points[ buffer_index + 1 : buffer_index + 2 ] )
                if ( buffer_index % 2 == 0 ) else
            ( buffer_index - 1, segment_points[ buffer_index - 1 : buffer_index ] )
            for ( buffer_index, is_nan )
            in enumerate( np.isnan( segment_points.x[ : vertex_count ] ) )
            if is_nan
        ] )

        # If one of a segment's point coordinates are NaN, set the coordinates
        # for both points to NaN. This prevents a rendering bug with some
        # drivers.
        segment_pairs[
            np.isnan( segment_pairs.x1 ) | np.isnan( segment_pairs.x2 )
        ] = ( np.nan, np.nan, np.nan, np.nan )

        self.vertex_buffer = gl_vbo.VBO( segment_points )
        self.vertex_count = vertex_count
        self.vertex_capacity = len( lines ) * 2

        self.color_buffer = gl_vbo.VBO(
            segment_colors.view( dtype = np.uint8 ),
        )
        self.pick_color_buffer = gl_vbo.VBO(
            np.zeros( ( len( segment_points ) * 4, ), dtype = np.uint8 ),
        )

        self.opengl_renderer.Refresh( False )

    def update_points( self, layer, points, projection, indices, point_map,
                       count = None, undo_recorded = None ):
        if self.vertex_buffer is None:
            return

        updated_points = points.view(
            self.POINTS_XY_DTYPE,
        ).xy.copy().view(
            self.SEGMENT_POINTS_DTYPE,
        )

        if indices is None:
            indices = xrange( 0, count or len( updated_points ) )

        for index in indices:
            point = updated_points[ index : index + 1 ]

            for buffer_index in point_map.get( index, list() ):
                # If the point is deleted, don't update it.
                if self.deleted_points.get( buffer_index ):
                    continue

                self.vertex_buffer[ buffer_index: buffer_index + 1 ] = point

                buffer_index += 1 if ( buffer_index % 2 == 0 ) else -1

                # If the other point in this line segment was deleted, then
                # restore it.
                deleted_point = self.deleted_points.get( buffer_index )
                if deleted_point and not np.isnan( deleted_point.x ):
                    self.vertex_buffer[ buffer_index: buffer_index + 1 ] = \
                        deleted_point
                    del( self.deleted_points[ buffer_index ] )

        self.opengl_renderer.Refresh( False )

    def add_points( self, layer, points, projection, start_index, count,
                    point_map, from_index = None, to_layer = None,
                    to_index = None, undo_recorded = None ):
        updated_points = points.view(
            self.POINTS_XY_DTYPE,
        ).xy[ start_index : start_index + count ].copy().view(
            self.SEGMENT_POINTS_DTYPE,
        )

        for point_index in range( start_index, start_index + count ):
            point = updated_points[ point_index - start_index :
                                    point_index - start_index + 1 ]

            for buffer_index in point_map.get( point_index, list() ):
                self.vertex_buffer[ buffer_index: buffer_index + 1 ] = point

                buffer_index += 1 if ( buffer_index % 2 == 0 ) else -1

                # If the other point in this line segment was deleted, then
                # restore it.
                deleted_point = self.deleted_points.get( buffer_index )
                if deleted_point:
                    self.vertex_buffer[ buffer_index: buffer_index + 1 ] = \
                        deleted_point
                    del( self.deleted_points[ buffer_index ] )

        self.opengl_renderer.Refresh( False )

    def delete_points( self, layer, points, projection, indices, point_map,
                       undo_recorded = None ):
        # Delete the requested points by setting them to NaN in the VBO. Also
        # NaN out the other point in the line segment, since some GL
        # implementations will still try to draw the line if only one endpoint
        # is NaNed. But first save a copy of it to make undeletion easier.
        for point_index in indices:
            for buffer_index in point_map.get( point_index, list() ):
                # The deleted_points array is only for recording point data
                # that has been NaNed out due to sharing a line segment with a
                # deleted point.
                self.deleted_points.pop( buffer_index, None )

                buffer_index += 1 if ( buffer_index % 2 == 0 ) else -1

                self.deleted_points[ buffer_index ] = \
                     self.vertex_buffer.data[ buffer_index: buffer_index + 1 ].copy()

        for point_index in indices:
            for buffer_index in point_map.get( point_index, list() ):
                self.vertex_buffer[ buffer_index: buffer_index + 1 ] = self.nans
                buffer_index += 1 if ( buffer_index % 2 == 0 ) else -1
                self.vertex_buffer[ buffer_index: buffer_index + 1 ] = self.nans

        self.opengl_renderer.Refresh( False )

    def add_lines( self, layer, points, lines, projection, indices,
                   undo_recorded = None ):
        if self.vertex_buffer is None:
            return

        new_capacity = len( lines ) * 2

        if new_capacity > self.vertex_capacity:
            self.logger.debug(
                "Growing lines VBOs from %d points capacity to %d." % (
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

        for segment_index in indices:
            doubled_index = segment_index * 2

            point1_index = lines.point1[ segment_index ]
            point2_index = lines.point2[ segment_index ]

            self.vertex_buffer[ doubled_index : doubled_index + 1 ] = \
                points[ point1_index ].view(
                    self.POINTS_XY_DTYPE
                )[ "xy" ].copy().view( self.SEGMENT_POINTS_DTYPE )
            self.color_buffer[ doubled_index * 4 : ( doubled_index + 1 ) * 4 ] = \
                lines.color[ segment_index : segment_index + 1 ].view( np.uint8 )

            self.vertex_buffer[ doubled_index + 1 : doubled_index + 2 ] = \
                points[ point2_index : point2_index + 1 ].view(
                    self.POINTS_XY_DTYPE
                )[ "xy" ].copy().view( self.SEGMENT_POINTS_DTYPE )
            self.color_buffer[ ( doubled_index + 1 ) * 4 : ( doubled_index + 2 ) * 4 ] = \
                lines.color[ segment_index : segment_index + 1 ].view( np.uint8 )

            self.vertex_count = max( self.vertex_count, doubled_index + 2 )

        self.opengl_renderer.Refresh( False )

    def delete_lines( self, layer, indices, undo_recorded = None ):
        if self.vertex_buffer is None:
            return

        for segment_index in indices:
            doubled_index = segment_index * 2

            self.vertex_buffer[ doubled_index : doubled_index + 1 ] = \
                self.nans
            self.vertex_buffer[ doubled_index + 1 : doubled_index + 2 ] = \
                self.nans

            self.vertex_count = max( self.vertex_count, doubled_index + 2 )

        self.opengl_renderer.Refresh( False )

    def render( self, pick_mode = False ):
        if self.vertex_buffer is None or self.vertex_count == 0:
            return

        reference_render_length = self.viewport.reference_render_length()
        threshold_ratio = 1.0

        if self.render_threshold is not None and \
           reference_render_length > self.render_threshold and \
           self.line_width < self.LINE_RESIZE_THRESHOLD:
            # If the user is zoomed out enough, then reduce the line size.
            # This makes the map easier to read when there are a lot of
            # lines.
            threshold_ratio = \
                    self.render_threshold / float( reference_render_length )
            threshold_ratio = max( min( threshold_ratio, 0.75 ), 0.25 )

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        if pick_mode:
            self.picker.bind_pick_colors(
                self.pick_color_buffer, self, self.vertex_count,
                doubled = True,
            )
            gl.glDisable( gl.GL_BLEND )
            gl.glDisable( gl.GL_LINE_SMOOTH )
            gl.glLineWidth( self.PICKER_LINE_WIDTH )
        else:
            self.picker.bind_colors(
                self.color_buffer, self, self.vertex_count,
                doubled = True,
            )
            gl.glLineWidth( self.line_width * threshold_ratio )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )

        gl.glDrawArrays( gl.GL_LINES, 0, self.vertex_count )

        if pick_mode:
            self.picker.unbind_colors( self.pick_color_buffer )
            gl.glEnable( gl.GL_BLEND )
            gl.glEnable( gl.GL_LINE_SMOOTH )
        else:
            self.picker.unbind_colors( self.color_buffer )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.vertex_buffer.unbind()

    def delete( self ):
        if self.vertex_buffer:
            self.vertex_buffer.delete()
            self.vertex_buffer = None
