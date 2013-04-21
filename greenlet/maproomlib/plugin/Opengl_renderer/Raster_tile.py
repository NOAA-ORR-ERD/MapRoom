import exceptions
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo


class Raster_tile:
    """
    A single raster tile stored as a texture.
    """
    def __init__( self, data, origin, size, pixel_origin, pixel_size,
                  scale_factor, transformer, projection = None ):
        self.origin = origin
        self.size = size
        self.scale_factor = scale_factor
        self.transformer = transformer
        self.source_projection = projection

        self.texture = gl.glGenTextures( 1 )
        gl.glBindTexture( gl.GL_TEXTURE_2D, self.texture )

        # A max level of zero prevents GL from trying to use mipmap levels
        # beyond zero (e.g. half-sized, quarter-sized, etc). This is done
        # because our own tiling and zooming mechanic essentially takes care
        # of that, and we don't want GL making its own separate set of smaller
        # tiles.
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0 )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE )
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            data.shape[ 1 ],
            data.shape[ 0 ],
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data.tostring(),
        )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                           gl.GL_LINEAR )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                           gl.GL_LINEAR_MIPMAP_NEAREST )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S,
                            gl.GL_CLAMP_TO_EDGE )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T,
                            gl.GL_CLAMP_TO_EDGE )

        # Note that geographic coordinates have their y origin at the bottom
        # rather than the top.
        vertex_data = np.array( [
            self.transformer.transform(
                ( origin[ 0 ], origin[ 1 ] + size[ 1 ] ),
                self.source_projection,
            ), # upper left
            self.transformer.transform(
                ( origin[ 0 ] + size[ 0 ], origin[ 1 ] + size[ 1 ] ),
                self.source_projection,
            ), # upper right
            self.transformer.transform(
                ( origin[ 0 ] + size[ 0 ], origin[ 1 ] ),
                self.source_projection,
            ), # bottom right
            self.transformer.transform(
                ( origin[ 0 ], origin[ 1 ] ),
                self.source_projection,
            ), # bottom left
        ], dtype = np.float32 )
        texcoord_data = np.array( [
            ( 0.0, 0.0 ),
            ( 1.0, 0.0 ),
            ( 1.0, 1.0 ),
            ( 0.0, 1.0 ),
        ], dtype = np.float32 )

        self.vertex_buffer = gl_vbo.VBO( vertex_data )
        self.texcoord_buffer = gl_vbo.VBO( texcoord_data )
        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )

    def render( self, pick_mode = False, faded = False, fade_factor = 0.0 ): 
        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vertex_buffer.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated

        if pick_mode is False:
            gl.glEnableClientState( gl.GL_TEXTURE_COORD_ARRAY )
            gl.glBindTexture( gl.GL_TEXTURE_2D, self.texture )

            self.texcoord_buffer.bind()
            gl.glTexCoordPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        else:
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                               gl.GL_NEAREST )
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                               gl.GL_NEAREST_MIPMAP_LINEAR )

        gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
        gl.glDrawArrays( gl.GL_QUADS, 0, 4 )

        if pick_mode is False:
            self.texcoord_buffer.unbind()
            gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
            gl.glDisableClientState( gl.GL_TEXTURE_COORD_ARRAY )

            # If requested, render the texture as faded white. Make it whiter
            # at higher scale factors (zoom levels).
            if faded:
                gl.glColor( 1, 1, 1, fade_factor )
                gl.glDrawArrays( gl.GL_QUADS, 0, 4 )
                gl.glColor( 1, 1, 1, 1 )
        else:
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                               gl.GL_NEAREST )
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                               gl.GL_NEAREST )

        # For debugging: Draw tile boundaries.
        #gl.glColor( 255, 0, 0, 255 )
        #gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_LINE )
        #gl.glDrawArrays( gl.GL_QUADS, 0, 4 )
        #gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
        #gl.glColor( 1, 1, 1, 1 )

        self.vertex_buffer.unbind()

        gl.glDisableClientState( gl.GL_VERTEX_ARRAY )

    # TODO: Don't rely on __del__ and instead make the tile cache explictly
    # call delete() when reaping a tile.
    def __del__( self ):
        self.delete()

    def delete( self ):
        if self.texture:
            gl.glDeleteTextures( np.array( [ self.texture ], np.uint32 ) )
            self.texture = None

        self.vertex_buffer = None
        self.texcoord_buffer = None

    def __hash__( self ):
        return hash( ( self.origin, self.scale_factor ) )

    def __eq__( self, other ):
        return hash( self ) == hash( other )
