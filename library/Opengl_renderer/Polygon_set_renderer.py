import numpy as np
import OpenGL.GL as gl
from Tessellator import init_vertex_buffers, tessellate
from Render import render_buffers_with_colors, render_buffers_with_one_color

FILL_SUB_LAYER_PICKER_OFFSET = 0
POINTS_SUB_LAYER_PICKER_OFFSET = 1
LINES_SUB_LAYER_PICKER_OFFSET = 2

class Polygon_set_renderer:
    
    ogrl = None
    
    points = None
    point_adjacency_array = None
    polygons = None
    
    polygon_count = 0
    triangle_vertex_buffers = None # projected
    triangle_vertex_counts = None
    line_vertex_buffers = None # projected
    line_vertex_counts = None
    line_nan_counts = None
    
    def __init__( self, opengl_renderer, points, point_adjacency_array, polygons, projection, projection_is_identity ):
        """
        points = 2 x np.float32, i.e., "2f4"
        point_adjacency = np array of
                            POLYGON_ADJACENCY_DTYPE = np.dtype( [ # parallels the points array
                                ( "next", np.uint32 ),       # Index of next adjacent point in polygon.
                                ( "polygon", np.uint32 )     # Index of polygon this point is in.
                            ] )
        polygons = np array of
                            POLYGON_DTYPE = np.dtype( [
                                ( "start", np.uint32 ),  # Index of arbitrary point in this polygon.
                                ( "count", np.uint32 ),  # Number of points in this polygon.
                                ( "group", np.uint32 ),  # An outer polygon and all of its holes have
                                                         # the same opaque group id.
                                ( "color", np.uint32 ),  # Color of this polygon.
                                ( "state", np.uint32 )   # standard maproom object states, plus polygon type
                            ] )
        projection = a pyproj-style projection callable object, such that
                        projection( world_x, world_y ) = ( projected_x, projected_y )
        """
        
        self.ogrl = opengl_renderer
        
        # gl.glEnable( gl.GL_BLEND )
        # gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
        # gl.glEnable( gl.GL_LINE_SMOOTH )
        # gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE )
        
        self.points = points.copy()
        self.point_adjacency_array = point_adjacency_array.copy()
        self.polygons = polygons.copy()
        self.polygon_count = np.alen( polygons )
        self.line_vertex_counts = polygons.count.copy()
        
        self.reproject( projection, projection_is_identity )
    
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
        
        # this is a mechanism to inform the calling program of invalid polygons
        # TODO: make this a pull (call to a get_invalid_polygons() method) instead
        # of a push (message)
        """
        self.layer.inbox.send(
            request = "set_invalid_polygons",
            polygon_indices = np.array( invalid_indices, np.uint32 ),
        )
        """
    
    def reproject( self, projection, projection_is_identity ):
        if ( self.polygon_count == 0 ):
            return
        
        self.destroy()
        
        self.triangle_vertex_buffers = np.ndarray(
            self.polygon_count,
            dtype = np.uint32
        )
        self.triangle_vertex_counts = np.ndarray(
            self.polygon_count,
            dtype = np.uint32
        )
        self.line_vertex_buffers = np.ndarray(
            self.polygon_count,
            dtype = np.uint32
        )
        self.line_nan_counts = np.zeros(
            self.polygon_count,
            dtype = np.uint32
        )
        
        init_vertex_buffers(
            self.triangle_vertex_buffers, # out parameter -- init_vertex_buffers() builds a vbo buffer for each polygon and stores them in this handle
            self.line_vertex_buffers, # out parameter -- init_vertex_buffers() builds a vbo buffer for each polygon and stores them in this handle
            start_index = 0,
            count = self.polygon_count,
            pygl = gl
        )
        
        projected_points = np.ndarray(
            ( np.alen( self.points ), 2 ),
            dtype = np.float32
        )
        if ( projection_is_identity ):
            projected_points[ : , 0 ] = self.points[ : , 0 ]
            projected_points[ : , 1 ] = self.points[ : , 1 ]
        else:
            projected_points[ : , 0 ], projected_points[ : , 1 ] = projection( self.points[ : , 0 ], self.points[ : , 1 ] )
        
        tessellate(
            projected_points, # used to be: self.points
            self.point_adjacency_array.next,
            self.point_adjacency_array.polygon,
            self.polygons.start,
            self.polygons.count, # per-polygon point count
            self.line_nan_counts, # out parameter -- how many nan/deleted points in each polygon
            self.polygons.group,
            self.triangle_vertex_buffers, # out parameter -- fills in the triangle vertex points
            self.triangle_vertex_counts, # out parameter -- how many triangle points for each polygon?
            self.line_vertex_buffers, # out parameter -- fills in the line vertex points
            gl
        )
        
        # print "total line_nan_counts = " + str( self.line_nan_counts.sum() )
        self.set_invalid_polygons( self.polygons, self.polygon_count )
    
    def render( self, layer_index_base, pick_mode, polygon_colors, line_color, line_width, broken_polygon_index = None ):
        """
        layer_index_base = the base number of this layer renderer for pick buffer purposes
        pick_mode = True if we are drawing to the off-screen pick buffer
        polygon_colors = np array of np.uint32, one per polygon
        line_color = uint
        broken_polygon_index = <for editing polygons; not currently used>
        """
        if self.triangle_vertex_buffers is None or self.polygon_count == 0:
            return
        
        # the fill triangles
        
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
        
        active_colors = polygon_colors
        if ( pick_mode ):
            start_color = ( layer_index_base + FILL_SUB_LAYER_PICKER_OFFSET ) << 24
            active_colors = np.arange( start_color, start_color + self.polygon_count, dtype = np.uint32 )
        
        render_buffers_with_colors(
            self.triangle_vertex_buffers[ : self.polygon_count ],
            active_colors,
            self.triangle_vertex_counts[ : self.polygon_count ],
            gl.GL_TRIANGLES,
            gl
        )
        
        # the lines
        
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        
        if ( pick_mode ):
            gl.glLineWidth( 6 )
            # note that all of the lines of each polygon get the color of the polygon as a whole
            render_buffers_with_colors(
                self.line_vertex_buffers[ : self.polygon_count ],
                active_colors,
                self.line_vertex_counts[ : self.polygon_count ],
                gl.GL_LINE_LOOP,
                gl
            )
        else:
            gl.glLineWidth( line_width )
            render_buffers_with_one_color(
                self.line_vertex_buffers[ : self.polygon_count ],
                line_color,
                self.line_vertex_counts[ : self.polygon_count ],
                gl.GL_LINE_LOOP,
                gl,
                0 if broken_polygon_index is None else broken_polygon_index,
                # If needed, render with one polygon border popped open.
                gl.GL_LINE_LOOP if broken_polygon_index is None else gl.GL_LINE_STRIP
            )
        
        # TODO: drawt the points if the polygon is selected for editing
        
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
    
    """
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
        
        for ( updated_point_index, point_index ) in enumerate( list( updated_points ) ):
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
                
                if polygon_index == polygon_count - 1 or  polygons.group[ polygon_index + 1 ] != group:
                    break
                polygon_index += 1
        
        tessellate(
            points.view( self.POINT_XY_DTYPE ).xy.copy(),
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
            if hasattr( layer, "points_layer" ) and layer.points_layer == self.layer.points_layer:
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
    """
    
    def destroy( self ):
        if ( self.triangle_vertex_buffers != None ):
            gl.glDeleteBuffers( self.triangle_vertex_buffers )
        if ( self.line_vertex_buffers != None ):
            gl.glDeleteBuffers( self.line_vertex_buffers )
