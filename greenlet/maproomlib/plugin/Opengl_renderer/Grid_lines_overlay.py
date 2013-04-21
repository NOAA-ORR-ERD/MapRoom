import bisect
import pyproj
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.utility as utility
from Label_set_renderer import Label_set_renderer


class Grid_lines_overlay:
    """
    Labeled grid lines displayed on top of the rendered layers.
    """
    POINTS_DTYPE = np.dtype(
        [ ( "x", np.float32 ), ( "y", np.float32 ) ],
    )
    LINE_WIDTH = 1.0
    LINE_COLOR = ( 0, 0, 0, 0.75 )
    RESET_COLOR = ( 1.0, 1.0, 1.0, 1.0 )
    LAT_LABEL_PIXEL_OFFSET = ( -25, 1 )
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
    )
    STEP_COUNT = len( STEPS )

    def __init__( self, opengl_renderer, viewport ):
        self.opengl_renderer = opengl_renderer
        self.viewport = viewport
        self.latlong = pyproj.Proj( "+proj=latlong" )
        self.initialized = False
        self.shown = True

    def toggle_shown( self ):
        self.shown = not self.shown

    def render( self ):
        if not self.shown or self.viewport.uninitialized():
            return

        if not self.initialized:
            self.labels_renderer = Label_set_renderer(
                None, None, self.viewport, self.opengl_renderer,
                None, None,
                max_char_count = 13,
            )
            gl.glEnable( gl.GL_BLEND )
            gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
            gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE );
            self.initialized = True

        latlong_origin = self.viewport.geo_origin( self.latlong )
        latlong_size = self.viewport.geo_size( self.latlong )
        reference_size = self.viewport.pixel_size_to_geo_size(
            self.REFERENCE_PIXEL_SIZE,
            self.latlong,
        )

        line_points = []
        labels = []

        # Determine which grid lines should be shown (degrees, minutes).
        long_step = self.STEPS[ min(
            bisect.bisect( self.STEPS, abs( reference_size[ 0 ] ) ),
            self.STEP_COUNT - 1,
        ) ]
        lat_step = self.STEPS[ min(
            bisect.bisect( self.STEPS, abs( reference_size[ 1 ] ) ),
            self.STEP_COUNT - 1,
        ) ]

        pixel_origin = ( 0, 0 )

        latlong_corner = (
            latlong_origin[ 0 ] + latlong_size[ 0 ],
            latlong_origin[ 1 ] + latlong_size[ 1 ],
        )
        pixel_size = self.viewport.pixel_size

        for longitude in np.arange(
            latlong_origin[ 0 ] + long_step - latlong_origin[ 0 ] % long_step,
            latlong_corner[ 0 ],
            long_step,
            dtype = np.float64,
        ):
            pixel_x = int(
                ( ( longitude - latlong_origin[ 0 ] ) / latlong_size[ 0 ] )
                * pixel_size[ 0 ]
            )

            line_points.append( ( pixel_x, pixel_size[ 1 ] ) )
            line_points.append( ( pixel_x, pixel_origin[ 1 ] ) )

            labels.append( (
                utility.format_long_line_label( longitude ),
                line_points[ -1 ],
            ) )

        for latitude in np.arange(
            latlong_origin[ 1 ] + lat_step - latlong_origin[ 1 ] % lat_step,
            latlong_corner[ 1 ],
            lat_step,
            dtype = np.float64,
        ):
            pixel_y = int(
                pixel_size[ 1 ] # Flip vertically.
                - ( ( latitude - latlong_origin[ 1 ] )
                    / latlong_size[ 1 ] ) * pixel_size[ 1 ]
            )

            line_points.append( ( pixel_origin[ 0 ], pixel_y ) )
            line_points.append( ( pixel_size[ 0 ], pixel_y ) )
            labels.append( (
                utility.format_lat_line_label( latitude ),
                # The offset shifts the latitude labels down and to the left
                # so that they aren't partially obscured by the edge of the
                # window.
                (
                    line_points[ -1 ][ 0 ] + self.LAT_LABEL_PIXEL_OFFSET[ 0 ],
                    line_points[ -1 ][ 1 ] + self.LAT_LABEL_PIXEL_OFFSET[ 1 ],
                )
            ) )

        point_count = len( line_points )

        if point_count == 0:
            return

        line_points = np.array(
            line_points, self.POINTS_DTYPE,
        ).view( np.recarray )

        line_vertex_buffer = gl_vbo.VBO(
            line_points,
            usage = gl.GL_STREAM_DRAW,
        )

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glLineWidth( self.LINE_WIDTH )
        gl.glDisable( gl.GL_LINE_SMOOTH )

        line_vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        gl.glColor( *self.LINE_COLOR )
        gl.glDrawArrays( gl.GL_LINES, 0, point_count )
        line_vertex_buffer.unbind()

        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glColor( *self.RESET_COLOR )

        gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.labels_renderer.init_labels(
            None, labels, len( labels ), None, None, None,
        )
        self.labels_renderer.render()
