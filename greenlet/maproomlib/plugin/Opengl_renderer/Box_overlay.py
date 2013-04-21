import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Box_overlay:
    """
    A simple box outline displayed on top of the rendered layers.
    """
    LINE_WIDTH = 1.0
    QUAD_CORNER_COUNT = 16
    LINE_CORNER_COUNT = 4
    CORNERS_DTYPE = np.dtype(
        [ ( "x", np.float32 ), ( "y", np.float32 ) ],
    )
    FILL_COLOR = ( 0, 0, 0, 0.3 )
    LINE_COLOR = ( 0, 0, 0, 1.0 )
    RESET_COLOR = ( 1.0, 1.0, 1.0, 1.0 )

    def __init__( self, box_zoomer, opengl_renderer, viewport ):
        self.box_zoomer = box_zoomer
        self.inbox = ui.Wx_inbox()
        self.opengl_renderer = opengl_renderer
        self.viewport = viewport
        self.start_position = None
        self.end_position = None

        self.fill_points = np.zeros(
            ( self.QUAD_CORNER_COUNT, ), self.CORNERS_DTYPE,
        ).view( np.recarray )
        self.line_points = np.zeros(
            ( self.LINE_CORNER_COUNT, ), self.CORNERS_DTYPE,
        ).view( np.recarray )

        self.fill_vertex_buffer = gl_vbo.VBO( self.fill_points )
        self.line_vertex_buffer = gl_vbo.VBO( self.line_points )

    def run( self, scheduler ):
        self.box_zoomer.outbox.subscribe(
            self.inbox,
            request = ( "start_box", "end_box", "move_box" ),
        )

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE );

        while True:
            message = self.inbox.receive( request = "start_box" )
            self.start_position = message.get( "position" )
            self.line_points.x[ 0 ] = self.start_position[ 0 ]
            self.line_points.y[ 0 ] = self.start_position[ 1 ]

            while True:
                message = self.inbox.receive(
                    request = ( "move_box", "end_box" ),
                )
                self.end_position = message.get( "position" )

                if message.get( "request" ) == "end_box":
                    self.start_position = None
                    self.end_position = None
                    self.opengl_renderer.Refresh( False )
                    break

                viewport_origin = ( 0, 0 )
                viewport_size = self.viewport.pixel_size

                # Line border around box area.
                self.line_points.x[ 1 ] = self.start_position[ 0 ]
                self.line_points.y[ 1 ] = self.end_position[ 1 ]

                self.line_points.x[ 2 ] = self.end_position[ 0 ]
                self.line_points.y[ 2 ] = self.end_position[ 1 ]

                self.line_points.x[ 3 ] = self.end_position[ 0 ]
                self.line_points.y[ 3 ] = self.start_position[ 1 ]

                self.line_vertex_buffer[ : ] = self.line_points

                # Fill quad below box area.
                self.fill_points.x[ 0 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 0 ] = viewport_origin[ 1 ]

                self.fill_points.x[ 1 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 1 ] = min(
                    self.start_position[ 1 ], self.end_position[ 1 ],
                )

                self.fill_points.x[ 2 ] = viewport_size[ 0 ]
                self.fill_points.y[ 2 ] = self.fill_points.y[ 1 ]

                self.fill_points.x[ 3 ] = viewport_size[ 0 ]
                self.fill_points.y[ 3 ] = viewport_origin[ 1 ]

                # Fill quad above box area.
                self.fill_points.x[ 4 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 4 ] = max(
                    self.start_position[ 1 ], self.end_position[ 1 ],
                )

                self.fill_points.x[ 5 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 5 ] = viewport_size[ 1 ]

                self.fill_points.x[ 6 ] = viewport_size[ 0 ]
                self.fill_points.y[ 6 ] = viewport_size[ 1 ]

                self.fill_points.x[ 7 ] = viewport_size[ 0 ]
                self.fill_points.y[ 7 ] = self.fill_points.y[ 4 ]

                # Fill quad to the left of the box area.
                self.fill_points.x[ 8 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 8 ] = self.fill_points.y[ 1 ]

                self.fill_points.x[ 9 ] = viewport_origin[ 0 ]
                self.fill_points.y[ 9 ] = self.fill_points.y[ 4 ]

                self.fill_points.x[ 10 ] = min(
                    self.start_position[ 0 ], self.end_position[ 0 ],
                )
                self.fill_points.y[ 10 ] = self.fill_points.y[ 9 ]

                self.fill_points.x[ 11 ] = self.fill_points.x[ 10 ]
                self.fill_points.y[ 11 ] = self.fill_points.y[ 1 ]

                # Fill quad to the right of the box area.
                self.fill_points.x[ 12 ] = max(
                    self.start_position[ 0 ], self.end_position[ 0 ],
                )
                self.fill_points.y[ 12 ] = self.fill_points.y[ 1 ]

                self.fill_points.x[ 13 ] = self.fill_points.x[ 12 ]
                self.fill_points.y[ 13 ] = self.fill_points.y[ 4 ]

                self.fill_points.x[ 14 ] = viewport_size[ 0 ]
                self.fill_points.y[ 14 ] = self.fill_points.y[ 9 ]

                self.fill_points.x[ 15 ] = viewport_size[ 0 ]
                self.fill_points.y[ 15 ] = self.fill_points.y[ 1 ]

                self.fill_vertex_buffer[ : ] = self.fill_points

                self.opengl_renderer.Refresh( False )

    def render( self ):
        if self.start_position is None or self.end_position is None:
            return

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.fill_vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        gl.glColor( *self.FILL_COLOR )
        gl.glDrawArrays( gl.GL_QUADS, 0, self.QUAD_CORNER_COUNT )
        self.fill_vertex_buffer.unbind()

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glLineWidth( self.LINE_WIDTH )

        self.line_vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        gl.glColor( *self.LINE_COLOR )
        gl.glDrawArrays( gl.GL_LINE_LOOP, 0, self.LINE_CORNER_COUNT )
        self.line_vertex_buffer.unbind()

        gl.glColor( *self.RESET_COLOR )

        gl.glDisableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
