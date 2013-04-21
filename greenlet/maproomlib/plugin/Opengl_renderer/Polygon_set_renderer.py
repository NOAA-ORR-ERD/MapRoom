import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Polygon_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering polygon_set layers.
    """
    LINE_RESIZE_THRESHOLD = 12.0
    LINE_COLOR = utility.color_to_int( 0, 0, 0, 1.0 )
    HOVER_COLOR = utility.color_to_int( 0, 1.0, 0, 0.5 )
    PICKER_LINE_WIDTH = 6.0   # in pixels
    RENDER_THRESHOLD_FACTOR = 17000
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
        self.polygon_points = None
        self.polygon_count = 0
        self.polygon_colors = None
        self.polygon_pick_colors = None
        self.triangle_vertex_buffers = None
        self.triangle_vertex_counts = None
        self.line_vertex_buffers = None
        self.line_vertex_counts = None
        self.line_nan_counts = None
        self.line_width = 1.0
        self.broken_polygon_index = None
        self.render_threshold = None
        self.logger = logging.getLogger( __name__ )

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE );

        # Don't cull polygons that are wound the wrong way.
        gl.glDisable( gl.GL_CULL_FACE )

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = ( "selection_updated", "clear_selection" ),
        )

        self.layer.outbox.subscribe(
            self.transformer( self.inbox ),
            request = (
                "polygons_updated",
                "size_changed",
            )
        )
        self.fetch_polygons()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "polygons",
                    "polygons_updated",
                    "projection_changed",
                    "selection_updated",
                    "clear_selection",
                    "size_changed",
                    "close",
                )
            )
            request = message.pop( "request" )

            if request == "update":
                self.inbox.discard( request = "update" )
            elif request == "polygons":
                self.init_polygons( **message )
            elif request == "polygons_updated":
                self.inbox.discard( request = "polygons_updated" )
                self.update_polygons( **message )
            elif request == "projection_changed":
                self.fetch_polygons()
            elif request == "selection_updated":
                self.selection_updated( **message )
            elif request == "clear_selection":
                self.selection_updated( selections = () )
            elif request == "size_changed":
                size = message.get( "size" )

                if size is not None:
                    self.render_threshold = max( size ) * \
                        self.RENDER_THRESHOLD_FACTOR
            elif request == "close":
                self.layer = None
                return

        self.layer.outbox.unsubscribe( self.inbox )

    def fetch_polygons( self ):
        self.layer.inbox.send(
            request = "get_polygons",
            origin = self.viewport.geo_origin( self.layer.projection ),
            size = self.viewport.geo_size( self.layer.projection ),
            response_box = self.transformer( self.inbox ),
        )

    def init_polygons( self, points, point_count, polygon_points, polygons,
                       polygon_count, projection, origin, size ):
        from Tessellator import init_vertex_buffers, tessellate

        if size is not None:
            self.render_threshold = max( size ) * \
                self.RENDER_THRESHOLD_FACTOR

        polygon_capacity = polygons.shape[ 0 ]
        self.polygon_points = polygon_points
        self.polygon_count = polygon_count
        self.polygon_colors = polygons.color
        self.polygon_pick_colors = polygons.color.copy()

        if self.triangle_vertex_buffers is None:
            self.triangle_vertex_buffers = np.ndarray(
                ( polygon_capacity, ),
                dtype = np.uint32,
            )
        if self.triangle_vertex_counts is None:
            self.triangle_vertex_counts = np.ndarray(
                ( polygon_capacity, ),
                dtype = np.uint32,
            )

        if self.line_vertex_buffers is None:
            self.line_vertex_buffers = np.ndarray(
                ( polygon_capacity, ),
                dtype = np.uint32,
            )
        self.line_vertex_counts = polygons.count.copy()

        if self.line_nan_counts is None:
            self.line_nan_counts = np.zeros(
                ( polygon_capacity, ),
                dtype = np.uint32,
            )

        if polygon_count == 0:
            return

        init_vertex_buffers(
            self.triangle_vertex_buffers,
            self.line_vertex_buffers,
            start_index = 0,
            count = polygon_count, # total number of polygons
            pygl = gl,
        )

        tessellate(
            points.view( self.POINTS_XY_DTYPE ).xy[ : point_count ].copy(),
            polygon_points.next,
            polygon_points.polygon,
            polygons.start[ : polygon_count ],
            polygons.count[ : polygon_count ], # per-polygon point count
            self.line_nan_counts,
            polygons.group[ : polygon_count ],
            self.triangle_vertex_buffers,
            self.triangle_vertex_counts,
            self.line_vertex_buffers,
            gl,
        )

        self.opengl_renderer.Refresh( False )

        self.set_invalid_polygons( polygons, polygon_count )

    def set_invalid_polygons( self, polygons, polygon_count ):
        # Invalid polygons are those that couldn't be tessellated and thus
        # have zero fill triangles. But don't consider hole polygons as
        # invalid polygons.
        invalid_indices_including_holes = np.where(
            self.triangle_vertex_counts[ : polygon_count ] == 0
        )[ 0 ]
        invalid_indices = []

        for index in invalid_indices_including_holes:
            if index > 0 and \
               polygons.group[ index ] != polygons.group[ index - 1 ]:
                invalid_indices.append( index )

        self.layer.inbox.send(
            request = "set_invalid_polygons",
            polygon_indices = np.array( invalid_indices, np.uint32 ),
        )

    def update_polygons( self, points, polygon_points, polygons,
                         polygon_count, updated_points, projection ):
        from Tessellator import init_vertex_buffers, tessellate

        if len( polygons ) > len( self.triangle_vertex_buffers ):
            new_capacity = len( polygons )
            self.logger.debug(
                "Growing polygons VBOs from %d polygons capacity to %d." % (
                    len( self.triangle_vertex_buffers ), new_capacity,
                ),
            )

            self.triangle_vertex_buffers = np.resize(
                self.triangle_vertex_buffers, ( new_capacity, )
            )
            self.triangle_vertex_counts = np.resize(
                self.triangle_vertex_counts, ( new_capacity, )
            )
            self.line_vertex_buffers = np.resize(
                self.line_vertex_buffers, ( new_capacity, )
            )
            self.line_nan_counts = np.resize(
                self.line_nan_counts, ( new_capacity, )
            )

        if polygon_count > self.polygon_count:
            new_polygons = polygon_count - self.polygon_count
            init_vertex_buffers(
                self.triangle_vertex_buffers,
                self.line_vertex_buffers,
                start_index = polygon_count - new_polygons,
                count = new_polygons,
                pygl = gl,
            )

        self.polygon_points = polygon_points
        self.polygon_count = polygon_count
        self.polygon_colors = polygons.color
        self.polygon_pick_colors = polygons.color.copy()
        self.line_vertex_counts = polygons.count.copy()

        if polygon_count == 0:
            return

        start_points = []
        polygon_counts = []
        polygon_groups = []

        for ( updated_point_index, point_index ) in \
            enumerate( list( updated_points ) ):
            # Determine the polygon group that the point is in.
            polygon_index = polygon_points.polygon[ point_index ]
            group = polygons.group[ polygon_index ]

            # Find the first polygon in the group, searching backward from
            # the current point's polygon.
            while polygon_index > 0:
                if polygons.group[ polygon_index - 1 ] != group:
                    break
                polygon_index -= 1

            # Replace the current point with a start point for each polygon in
            # the group. This ensures that tessellating a hole polygon due to
            # an updated point triggers the tessellation of all other polygons
            # in the hole's group as well.
            while polygon_index <= polygon_count - 1:
                start_points.append( polygons.start[ polygon_index ] )
                polygon_counts.append( polygons.count[ polygon_index ] )
                polygon_groups.append( polygons.group[ polygon_index ] )

                if polygon_index == polygon_count - 1 or \
                   polygons.group[ polygon_index + 1 ] != group:
                    break
                polygon_index += 1

        tessellate(
            points.view( self.POINTS_XY_DTYPE ).xy.copy(),
            polygon_points.next,
            polygon_points.polygon,
            np.array( start_points, np.uint32 ),
            np.array( polygon_counts, np.uint32 ),
            self.line_nan_counts,
            np.array( polygon_groups, np.uint32 ),
            self.triangle_vertex_buffers,
            self.triangle_vertex_counts,
            self.line_vertex_buffers,
            gl,
        )

        self.opengl_renderer.Refresh( False )

        self.set_invalid_polygons( polygons, polygon_count )

    def selection_updated( self, selections, **other ):
        selected_points = False
        indices = ()

        # We're just interested in selection changes to the points layer.
        for ( layer, indices ) in selections:
            if hasattr( layer, "points_layer" ) and \
               layer.points_layer == self.layer.points_layer:
                selected_points = True
                break

        if selected_points is False or len( indices ) == 0:
            # If polygon points are no longer selected, then close the polygon
            # back up.
            self.broken_polygon_index = None
            self.opengl_renderer.Refresh( False )
            return

        # Arbitrarily pick the largest point index of the selected points as
        # the break point.
        point_index = max( indices )
        new_start_point = self.polygon_points.next[ point_index ]
        self.broken_polygon_index = self.polygon_points.polygon[ point_index ]

        # Setting the start index makes sure that the broken boundary line is
        # after that point.
        self.layer.inbox.send(
            request = "set_polygon_starts",
            start_indices = [ new_start_point ],
        )

    def render( self, pick_mode = False ):
        from Render import render_buffers_with_colors, \
                           render_buffers_with_one_color

        if self.triangle_vertex_buffers is None or self.polygon_count == 0:
            return

        reference_render_length = self.viewport.reference_render_length()
        threshold_ratio = 1.0

        if self.render_threshold is not None and \
           reference_render_length > self.render_threshold and \
           self.line_width < self.LINE_RESIZE_THRESHOLD:
            # If the user is zoomed out enough, then reduce the line size.
            # This makes the map easier to read when there are a lot of
            # lines.
            threshold_ratio = max(
                self.render_threshold / float( reference_render_length ),
                0.25,
            )

        if pick_mode is True:
            self.picker.set_pick_colors( self.polygon_pick_colors, self )
            gl.glDisable( gl.GL_BLEND )
        else:
            self.picker.set_colors(
                self.polygon_colors, self,
            )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        # Render polygon fill triangles.
        render_buffers_with_colors(
            self.triangle_vertex_buffers[ : self.polygon_count ],
            self.polygon_pick_colors if pick_mode else self.polygon_colors,
            self.triangle_vertex_counts[ : self.polygon_count ],
            gl.GL_TRIANGLES,
            gl,
        )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )

        # Reduce each polygon's rendered vertex count by the number of
        # NaN points in that polygon's boundary.
        vertex_counts = \
            self.line_vertex_counts[ : self.polygon_count ] - \
            self.line_nan_counts[ : self.polygon_count ]

        # In pick mode, render each line loop with the same color as its
        # fill triangles. This is necessary in case a particular polygon
        # has no fill triangles.
        if pick_mode is True:
            gl.glLineWidth( self.PICKER_LINE_WIDTH )
            render_buffers_with_colors(
                self.line_vertex_buffers[ : self.polygon_count ],
                self.polygon_pick_colors,
                vertex_counts,
                gl.GL_LINE_LOOP,
                gl,
            )
        else:
        # In non-pick mode, render all line loops with the same color.
            gl.glLineWidth( self.line_width * threshold_ratio )
            print "line color is: {0:X}".format( int( self.LINE_COLOR ) )
            render_buffers_with_one_color(
                self.line_vertex_buffers[ : self.polygon_count ],
                self.LINE_COLOR,
                vertex_counts,
                gl.GL_LINE_LOOP,
                gl,
                self.broken_polygon_index or 0,
                # If needed, render with one polygon border popped open.
                gl.GL_LINE_LOOP if self.broken_polygon_index is None else \
                    gl.GL_LINE_STRIP,
            )

        if pick_mode is True:
            gl.glEnable( gl.GL_BLEND )

    def delete( self ):
        gl.glDeleteBuffers( self.triangle_vertex_buffers )
