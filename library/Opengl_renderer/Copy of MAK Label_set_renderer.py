import os
import wx
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import library.rect as rect
import Font_extents
import globals

class Label_set_renderer:
    VERTEX_DTYPE = np.dtype(
        [ ( "x", np.float32 ), ( "y", np.float32 ) ]
    )
    
    MAX_CHARACTERS = 10000
    font_texture = None
    font_texture_size = None
    font_extents = Font_extents.FONT_EXTENTS
    vbo_character_vertexes = None
    vbo_character_texture_coordinates = None
    
    def __init__( self ):
        self.read_and_set_up_texture()
        
        # reserve as much space as we'll ever use
        vertex_data = np.zeros(
            4 * self.MAX_CHARACTERS,
            dtype = self.VERTEX_DTYPE
        ).view( np.recarray )
        self.vbo_character_vertexes = gl_vbo.VBO( vertex_data )
        texcoord_data = np.zeros(
            4 * self.MAX_CHARACTERS,
            dtype = self.VERTEX_DTYPE
        ).view( np.recarray )
        self.vbo_character_texture_coordinates = gl_vbo.VBO( texcoord_data )
    
    def read_and_set_up_texture( self ):
        path = os.path.join( os.path.dirname( __file__ ), "font.png" )
        image = wx.Image( path, wx.BITMAP_TYPE_PNG )
        width = image.GetWidth()
        height = image.GetHeight()
        self.font_texture_size = ( width, height )
        # create an RBG buffer
        buffer = np.frombuffer( image.GetDataBuffer(), np.uint8 ).reshape(
            ( width, height, 3 ),
        )
        
        # Make an alpha channel that is opaque where the pixels are black
        # and semi-transparent where the pixels are white.
        buffer_with_alpha = np.empty( ( width, height, 4 ), np.uint8 )
        buffer_with_alpha[ :, :, 0:3 ] = buffer
        buffer_with_alpha[ :, :, 3 ] = (
            255 - buffer[ :, :, 0:3 ].sum( axis = 2 ) / 3
        ).clip( 230, 255 )
        
        gl.glEnable( gl.GL_TEXTURE_2D )
        self.font_texture = gl.glGenTextures( 1 )
        gl.glBindTexture( gl.GL_TEXTURE_2D, self.font_texture )
        
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            buffer_with_alpha.tostring()
        )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST )
        gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE )
        gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE )
        
        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
        gl.glDisable( gl.GL_TEXTURE_2D )
    
    def draw_labels( self, points, strings ):
        """"
        vertex_data = np.zeros(
            4 * self.MAX_CHARACTERS,
            dtype = self.VERTEX_DTYPE
        ).view( np.recarray )
        self.vbo_character_vertexes = gl_vbo.VBO( vertex_data )
        texcoord_data = np.zeros(
            4 * self.MAX_CHARACTERS,
            dtype = self.VERTEX_DTYPE
        ).view( np.recarray )
        self.vbo_character_texture_coordinates = gl_vbo.VBO( texcoord_data )
        """
        
        total_char_count = sum( map( len, strings ) )
        print "total_char_count = {0}".format( total_char_count )
        vertex_data = self.vbo_character_vertexes.data
        texcoord_data = self.vbo_character_texture_coordinates.data
        
        window_size = globals.application.renderer.GetClientSize()
        degrees_per_pixel_h = rect.width( globals.application.world_rect ) / float( window_size[ 0 ] )
        degrees_per_pixel_v = rect.height( globals.application.world_rect ) / float( window_size[ 1 ] )
        
        texture_width = float( self.font_texture_size[ 0 ] )
        texture_height = float( self.font_texture_size[ 1 ] )
        
        # dx10 = 10 * degrees_per_pixel_h
        # dy10 = 10 * degrees_per_pixel_v
        # print "dx10 = {0}, dy10 = {1}".format( str( dx10 ), str( dy10 ) )
        
        a = np.array( [ 0, 0 ], dtype = np.float32 )
        num_characters = 0
        for i in xrange( len( points ) ):
            if ( num_characters >= self.MAX_CHARACTERS ):
                break
            p = points[ i ]
            s = strings[ i ]
            world_x_offset = 0
            for c in s:
                if ( num_characters < self.MAX_CHARACTERS ):
                    ( x, y, w, h ) = self.font_extents[ c ]
                    base = num_characters * 4
                    
                    dx = w * degrees_per_pixel_h
                    dy = h * degrees_per_pixel_v
                    # left-bottom
                    vertex_data.x[ base ] = world_x_offset + p[ 0 ]
                    vertex_data.y[ base ] = p[ 1 ]
                    # left-top
                    vertex_data.x[ base + 1 ] = world_x_offset + p[ 0 ]
                    vertex_data.y[ base + 1 ] = p[ 1 ] + dy
                    # a[ 0 ] = world_x_offset + p[ 0 ]
                    # a[ 1 ] = p[ 1 ] + dy
                    # import code; code.interact( local = locals() )
                    # self.vbo_character_vertexes[ base + 1 : base + 1 ] = a.view( self.VERTEX_DTYPE )
                    # right-top
                    vertex_data.x[ base + 2 ] = world_x_offset + p[ 0 ] + dx
                    vertex_data.y[ base + 2 ] = p[ 1 ] + dy
                    # right-bottom
                    vertex_data.x[ base + 3 ] = world_x_offset + p[ 0 ] + dx
                    vertex_data.y[ base + 3 ] = p[ 1 ]
                    world_x_offset += ( w + 1 ) * degrees_per_pixel_h
                    
                    x1 = x / texture_width
                    x2 = ( x + w ) / texture_width
                    y1 = y / texture_height
                    y2 = ( y + h ) / texture_height
                    # left-bottom
                    texcoord_data.x[ base ] = x1
                    texcoord_data.y[ base ] = y2
                    # left-top
                    texcoord_data.x[ base + 1 ] = x1
                    texcoord_data.y[ base + 1 ] = y1
                    # right-top
                    texcoord_data.x[ base + 2 ] = x2
                    texcoord_data.y[ base + 2 ] = y1
                    # right-bottom
                    texcoord_data.x[ base + 3 ] = x2
                    texcoord_data.y[ base + 3 ] = y2
                    
                    num_characters += 1
        
        """
        vertex_data.x[ 0 ] = -80
        vertex_data.y[ 0 ] = 0
        vertex_data.x[ 1 ] = -80
        vertex_data.y[ 1 ] = 10
        vertex_data.x[ 2 ] = 1080
        vertex_data.y[ 2 ] = 10
        vertex_data.x[ 3 ] = 1080
        vertex_data.y[ 3 ] = 0
        
        texcoord_data.x[ 0 ] = 0
        texcoord_data.y[ 0 ] = 1.0
        texcoord_data.x[ 1 ] = 0
        texcoord_data.y[ 1 ] = 0
        texcoord_data.x[ 2 ] = 1.0
        texcoord_data.y[ 2 ] = 0
        texcoord_data.x[ 3 ] = 1.0
        texcoord_data.y[ 3 ] = 1.0
        """
        
        gl.glEnable( gl.GL_TEXTURE_2D )
        gl.glBindTexture( gl.GL_TEXTURE_2D, self.font_texture )
        
        # self.read_and_set_up_texture()
        
        gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
        self.vbo_character_vertexes.bind()
        gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        gl.glEnableClientState( gl.GL_TEXTURE_COORD_ARRAY )
        self.vbo_character_texture_coordinates.bind()
        gl.glTexCoordPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
        
        gl.glDrawArrays( gl.GL_QUADS, 0, num_characters * 4 )
        # gl.glDrawArrays( gl.GL_QUADS, 0, 4 )
        
        self.vbo_character_vertexes.unbind()
        self.vbo_character_texture_coordinates.unbind()
        gl.glDisableClientState( gl.GL_TEXTURE_COORD_ARRAY )
        gl.glDisableClientState( gl.GL_VERTEX_ARRAY )
        
        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
        gl.glDisable( gl.GL_TEXTURE_2D )
    
    def destroy( self ):
        if ( self.font_texture != None ):
            gl.glDeleteTextures(
                np.array( [ self.font_texture ], np.uint32 ),
            )
            self.font_texture = None
        self.vbo_character_vertexes = None
        self.texcoord_buffer = None
