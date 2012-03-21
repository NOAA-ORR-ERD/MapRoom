import bisect
import numpy as np
import library.rect as rect
import library.coordinates
import app_globals

class Lon_lat_grid:
    LINE_WIDTH = 1.0
    LINE_COLOR = ( 0, 0, 0, 0.75 )
    # a nice amount of spacing between lines
    REFERENCE_PIXEL_SIZE = ( 100, 100 )
    
    DEGREE = np.float64( 1.0 )
    MINUTE = DEGREE / 60.0
    SECOND = MINUTE / 60.0
    
    STEPS = (
        MINUTE,
        MINUTE * 2,
        MINUTE * 3,
        MINUTE * 4,
        MINUTE * 5,
        MINUTE * 10,
        MINUTE * 15,
        MINUTE * 20,
        MINUTE * 30,
        DEGREE,
        DEGREE * 2,
        DEGREE * 3,
        DEGREE * 4,
        DEGREE * 5,
        DEGREE * 10,
        DEGREE * 15,
        DEGREE * 20,
        DEGREE * 30,
        DEGREE * 40,
    )
    STEP_COUNT = len( STEPS )
    
    def __init__( self ):
        pass
    
    ## fixme == this should be able to get the various rects from the render_window object...
    def draw( self, render_window, world_rect, projected_rect, screen_rect ):
        #app = app_globals.application
        opengl_renderer = render_window.opengl_renderer
        degrees_lon_per_pixel = float( rect.width( world_rect ) ) / float( rect.width( screen_rect ) )
        degrees_lat_per_pixel = float( rect.height( world_rect ) ) / float( rect.height( screen_rect ) )
        
        reference_size = ( self.REFERENCE_PIXEL_SIZE[ 0 ] * degrees_lon_per_pixel,
                           self.REFERENCE_PIXEL_SIZE[ 1 ] * degrees_lat_per_pixel )
        
        # Determine which grid lines should be shown (degrees, minutes).
        lon_step = self.STEPS[ min(
            bisect.bisect( self.STEPS, abs( reference_size[ 0 ] ) ),
            self.STEP_COUNT - 1,
        ) ]
        lat_step = self.STEPS[ min(
            bisect.bisect( self.STEPS, abs( reference_size[ 1 ] ) ),
            self.STEP_COUNT - 1,
        ) ]
        
        # print "lon_step = " + str( lon_step )
        # print "lat_step = " + str( lat_step )
        # print "world_rect = " + str( world_rect )
        
        for longitude in np.arange(
            world_rect[ 0 ][ 0 ] + lon_step - world_rect[ 0 ][ 0 ] % lon_step,
            world_rect[ 1 ][ 0 ],
            lon_step,
            dtype = np.float64 ):
            
            # print "  longitude = " + str( longitude )
            if ( longitude < -180 or longitude > 180 ):
                continue
            w_p = ( longitude, world_rect[ 0 ][ 1 ] )
            s_p = render_window.get_screen_point_from_world_point( w_p )
            s = library.coordinates.format_lon_line_label( longitude )
            size = opengl_renderer.get_drawn_string_dimensions( s )
            opengl_renderer.draw_screen_line( ( s_p[ 0 ], screen_rect[ 0 ][ 1 ] + size[ 1 ] + 5 ),
                                              ( s_p[ 0 ], screen_rect[ 1 ][ 1 ] ) )
            """
            for offset in xrange( 200 ):
                opengl_renderer.draw_screen_string( ( s_p[ 0 ] - size[ 0 ] / 2, screen_rect[ 0 ][ 1 ] + offset * 2 ), s )
            """
            opengl_renderer.draw_screen_string( ( s_p[ 0 ] - size[ 0 ] / 2, screen_rect[ 0 ][ 1 ] ), s )
        
        for latitude in np.arange(
            world_rect[ 0 ][ 1 ] + lat_step - world_rect[ 0 ][ 1 ] % lat_step,
            world_rect[ 1 ][ 1 ],
            lat_step,
            dtype = np.float64 ):
            
            # print "  latitude = " + str( latitude )
            if ( latitude < -89 or latitude > 89 ):
                continue
            w_p = ( world_rect[ 0 ][ 0 ], latitude )
            s_p = render_window.get_screen_point_from_world_point( w_p )
            s = library.coordinates.format_lat_line_label( latitude )
            size = opengl_renderer.get_drawn_string_dimensions( s )
            opengl_renderer.draw_screen_line( ( screen_rect[ 0 ][ 0 ], s_p[ 1 ] ),
                                              ( screen_rect[ 1 ][ 0 ] - size[ 0 ] - 5, s_p[ 1 ] ) )
            opengl_renderer.draw_screen_string(
                ( screen_rect[ 1 ][ 0 ] - size[ 0 ] - 3, s_p[ 1 ] - size[ 1 ] / 2 - 1 ), s )
