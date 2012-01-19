import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo

class Triangle_set_renderer:
    
    CHANNELS = 4 # i.e., RBGA
    
    oglr = None
    
    vbo_point_xys = None
    vbo_point_colors = None
    vbo_triangle_point_indexes = None
    
    def __init__( self, opengl_renderer, points, point_colors, triangle_point_indexes, projection, projection_is_identity ):
        """
            points = np array of 2 x np.float32, i.e., "2f4"
            point_colors = np array of np.uint32, one per point
            triangle_point_indexes = np array of 3 x np.uint32, i.e., "3u4"
            projection = a pyproj-style projection callable object, such that
                            projection( world_x, world_y ) = ( projected_x, projected_y )
        """
        
        self.oglr = opengl_renderer
        # self.world_points = points.copy() # .view( self.SIMPLE_POINT_DTYPE ) # .view( np.recarray )
        
        if ( points == None or len( points ) == 0 ):
            return
        
        projected_point_data = np.zeros(
            ( len( points ), 2 ),
            dtype = np.float32
        )
        
        self.vbo_point_xys = gl_vbo.VBO(
            projected_point_data,
            usage = "GL_STATIC_DRAW"
        )
        self.vbo_point_colors = gl_vbo.VBO(
            point_colors
        )
        
        self.build_triangle_buffers( points, triangle_point_indexes )
        
        self.reproject( points, projection, projection_is_identity )
    
    def build_triangle_buffers( self, points, triangle_point_indexes ):
        """
        projected_triangle_point_data = np.zeros(
            ( len( points ), 2 ),
            dtype = np.float32
        )
        self.vbo_triangle_point_xys = gl_vbo.VBO(
            projected_triangle_point_data
        )
        """
        self.vbo_triangle_point_indexes = gl_vbo.VBO(
            triangle_point_indexes.view( np.uint32 ),
            usage = "GL_STATIC_DRAW",
            target = "GL_ELEMENT_ARRAY_BUFFER"
        )
    
    def reproject( self, points, projection, projection_is_identity ):
        if ( points == None or len( points ) == 0 ):
            self.vbo_point_xys = None
            self.vbo_point_colors = None
            self.vbo_triangle_point_indexes = None
            #
            return
        
        projected_point_data = self.vbo_point_xys.data
        if ( projection_is_identity ):
            projected_point_data[ : , 0 ] = points[ : , 0 ]
            projected_point_data[ : , 1 ] = points[ : , 1 ]
        else:
            projected_point_data[ : , 0 ], projected_point_data[ : , 1 ] = projection( points[ : , 0 ], points[ : , 1 ] )
        self.vbo_point_xys[ : np.alen( projected_point_data ) ] = projected_point_data
        """
        if ( self.vbo_triangle_point_xys != None and len( self.vbo_triangle_point_xys.data ) > 0 ):
            projected_triangle_point_data = self.vbo_triangle_point_xys.data
            if ( projection.srs.find( "+proj=longlat" ) != -1 ):
                projected_triangle_point_data[ : , 0 ] = self.world_triangle_points[ : , 0 ]
                projected_triangle_point_data[ : , 1 ] = self.world_triangle_points[ : , 1 ]
            else:
                projected_triangle_point_data[ : , 0 ], projected_triangle_point_data[ : , 1 ] = projection( self.world_triangle_points[ : , 0 ], self.world_triangle_points[ : , 1 ] )
            self.vbo_triangle_point_xys[ : np.alen( projected_triangle_point_data ) ] = projected_triangle_point_data
        """
    
    def render( self, pick_mode, point_size, line_width ):
        # TODO: for now triangles can't be selected with mouse click
        if ( pick_mode ):
            return
        
        # the line segments
        if ( self.vbo_triangle_point_indexes != None and len( self.vbo_triangle_point_indexes.data ) > 0 ):
            """
            gl.glLineWidth( line_width )
            gl.glColor( 0.75, 0.75, 0.75, 0.75 )
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
            gl.glBegin( gl.GL_TRIANGLES )
            for i in xrange( len( self.vbo_triangle_point_indexes ) ):
                for j in xrange( 3 ):
                    p = self.vbo_point_xys.data[ self.vbo_triangle_point_indexes.data[ i ][ j ] ]
                    gl.glVertex( p[ 0 ], p[ 1 ], 0 )
            gl.glEnd()
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
            gl.glColor( 1, 1, 1, 1 )
            """
            
            gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
            
            gl.glEnableClientState( gl.GL_INDEX_ARRAY ) # FIXME: deprecated
            self.vbo_triangle_point_indexes.bind()
            
            gl.glColor( 0.5, 0.5, 0.5, 0.75 )
            # gl.glEnableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
            # self.vbo_triangle_colors.bind()
            # gl.glColorPointer( self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None ) # FIXME: deprecated
            
            gl.glLineWidth( line_width )
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
            
            # import code; code.interact( local = locals() )
            gl.glDrawElements( gl.GL_TRIANGLES, np.alen( self.vbo_triangle_point_indexes.data ) * 3, gl.GL_UNSIGNED_INT, None )
            # gl.glDrawArrays( gl.GL_TRIANGLES, 0, np.alen( self.vbo_triangle_point_indexes ) )
            
            gl.glColor( 1, 1, 1, 1 )
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
            
            # gl.glDisableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
            # self.vbo_triangle_colors.unbind()
            gl.glDisableClientState( gl.GL_INDEX_ARRAY ) # FIXME: deprecated
            gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
            self.vbo_triangle_point_indexes.unbind()
            self.vbo_point_xys.unbind()
        
        # the points
        if ( self.vbo_point_xys != None and len( self.vbo_point_xys ) > 0 ):
            gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
            self.vbo_point_xys.bind()
            gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
            
            # To make the points stand out better, especially when rendered on top
            # of line segments, draw translucent white rings under them.
            gl.glPointSize( point_size + 2 )
            gl.glColor( 1, 1, 1, 0.75 )
            gl.glDrawArrays( gl.GL_POINTS, 0, np.alen( self.vbo_point_xys.data ) )
            gl.glColor( 1, 1, 1, 1 )
            
            gl.glEnableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
            self.vbo_point_colors.bind()
            gl.glColorPointer( self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None ) # FIXME: deprecated
            gl.glPointSize( point_size )
            
            gl.glDrawArrays( gl.GL_POINTS, 0, np.alen( self.vbo_point_xys.data ) )
            
            self.vbo_point_colors.unbind()
            gl.glDisableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
            
            gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
            self.vbo_point_xys.unbind()
    
    def destroy( self ):
        self.vbo_point_xys = None
        self.vbo_point_colors = None
        self.vbo_triangle_point_indexes = None
