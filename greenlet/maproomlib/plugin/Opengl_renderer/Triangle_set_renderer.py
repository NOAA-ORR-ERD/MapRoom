import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.ui as ui
import maproomlib.utility as utility


class Triangle_set_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering triangle_set layers. These represent individual triangles that
    may or may not be adjacent.
    """
    POINT_SIZE = 8.0      # in pixels
    POINTS_XY_DTYPE = np.dtype( [
        ( "xy", "2f4" ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )             
    TRIANGLES_POINTS_DTYPE = np.dtype( [
        ( "points", "3u4" ),
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
        self.triangle_buffer = None
        self.triangle_count = 0
        self.line_width = 1.0
        self.logger = logging.getLogger( __name__ )

        gl.glEnable( gl.GL_BLEND )
        gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )
        gl.glEnable( gl.GL_LINE_SMOOTH )
        gl.glHint( gl.GL_LINE_SMOOTH_HINT, gl.GL_DONT_CARE );

    def run( self, scheduler ):
        self.fetch_triangles()

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "triangles",
                    "projection_changed",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "update":
                self.inbox.discard( request = "update" )
            elif request == "triangles":
                unique_id = "render triangles %s" % self.layer.name
                self.outbox.send(
                    request = "start_progress",
                    id = unique_id,
                    message = "Initializing boundaries"
                )
                try:
                    self.init_triangles(
                        message.get( "points" ),
                        message.get( "point_count" ),
                        message.get( "triangles" ),
                        message.get( "triangle_count" ),
                    )
                finally:
                    self.outbox.send(
                        request = "end_progress",
                        id = unique_id,
                    )
            elif request == "projection_changed":
                self.fetch_triangles()
            elif request == "close":
                self.layer = None
                return

        self.layer.outbox.unsubscribe( self.inbox )

    def fetch_triangles( self ):
        self.layer.inbox.send(
            request = "get_triangles",
            response_box = self.transformer( self.inbox ),
        )

    def init_triangles( self, points, point_count, triangles, triangle_count ):
        points_xy = points.view( self.POINTS_XY_DTYPE )[ "xy" ].copy()

        self.vertex_buffer = gl_vbo.VBO(
            points_xy,
            usage = "GL_STATIC_DRAW",
        )
        self.vertex_count = point_count

        self.triangle_buffer = gl_vbo.VBO(
            triangles.view( np.uint32 ),
            usage = "GL_STATIC_DRAW",
            target = "GL_ELEMENT_ARRAY_BUFFER",
        )
        self.triangle_count = triangle_count

        self.opengl_renderer.Refresh( False )

    def render( self, pick_mode = False ):
        if self.vertex_buffer is None or pick_mode is True:
            return

        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        gl.glEnableClientState( gl.GL_INDEX_ARRAY ) # FIXME: deprecated
        self.triangle_buffer.bind()

        gl.glLineWidth( self.line_width )
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        gl.glColor( 0.75, 0.75, 0.75, 0.75 )

        gl.glDrawElements(
            gl.GL_TRIANGLES, self.triangle_count * 3, gl.GL_UNSIGNED_INT, None,
        )

        gl.glColor( 1, 1, 1, 1 )
        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )

        self.vertex_buffer.unbind()
        self.triangle_buffer.unbind()

    def delete( self ):
        if self.vertex_buffer:
            self.vertex_buffer.delete()
            self.vertex_buffer = None
        if self.triangle_buffer:
            self.triangle_buffer.delete()
            self.triangle_buffer = None
