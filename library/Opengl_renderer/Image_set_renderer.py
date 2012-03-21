import os
import wx
import numpy as np
import time
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import library.rect as rect
from library.accumulator import flatten

class Image_set_renderer:
    oglr = None
    
    textures = None # flat list, one per image
    image_sizes = None # flat list, one per image
    image_world_rects = None # flat list, one per image
    image_projected_rects = None # flat list, one per image
    
    vbo_vertexes = None # flat list, one per texture
    vbo_texture_coordinates = None # just one, same one for all images
    
    def __init__( self, opengl_renderer, images, image_sizes, image_world_rects, projection, projection_is_identity ):
        """
            images = list of lists, where each sublist is a row of images
                        and each image is a numpy array [ 0 : max_y, 0 : max_x, 0 : num_bands ]
                        where:
                            num_bands = 4
                            max_x and max_y = 1024,
                                except for the last image in each row (may be narrower) and
                                the images in the last row (may be shorter)
            image_sizes = list of lists, the same shape as images,
                          but where each item gives the ( width, height ) pixel size
                          of the corresponding image
            image_world_rects = list of lists, the same shape as images,
                                but where each item gives the world rect
                                of the corresponding image
            projection = a pyproj-style projection callable object, such that
                         projection( world_x, world_y ) = ( projected_x, projected_y )
        """
        
        self.oglr = opengl_renderer
        image_list = flatten( images )
        self.image_sizes = flatten( image_sizes )
        self.image_world_rects = flatten ( image_world_rects )
        
        total_image_count = len( image_list )
        
        texcoord_data = np.zeros(
            ( 1, ),
            dtype = self.oglr.TEXTURE_COORDINATE_DTYPE,
        ).view( np.recarray )
        
        self.textures = []
        self.vbo_vertexes = []
        
        n = 0
        for i in xrange( len( image_list ) ):
            image_data = image_list[ i ]
            self.textures.append( gl.glGenTextures( 1 ) )
            gl.glBindTexture( gl.GL_TEXTURE_2D, self.textures[ i ] )
            # Mipmap levels: half-sized, quarter-sized, etc.
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 4 )
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_TRUE )
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR )
            gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR )
            # gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR )
            # gl.glTexParameter( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST )
            gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE )
            gl.glTexParameteri( gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE )
            gl.glTexEnvf( gl.GL_TEXTURE_FILTER_CONTROL, gl.GL_TEXTURE_LOD_BIAS, -0.5 )
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0, # level
                gl.GL_RGBA8,
                image_data.shape[ 1 ], # width
                image_data.shape[ 0 ], # height
                0, # border
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                image_data.tostring()
            )
            
            vertex_data = np.zeros(
                ( 1, ),
                dtype = self.oglr.QUAD_VERTEX_DTYPE,
            ).view( np.recarray )
            # we fill the vbo_vertexes data in reproject() below
            self.vbo_vertexes.append( gl_vbo.VBO( vertex_data ) )
        
        texcoord_data.u_lb = 0
        texcoord_data.v_lb = 1.0
        texcoord_data.u_lt = 0
        texcoord_data.v_lt = 0
        texcoord_data.u_rt = 1.0
        texcoord_data.v_rt = 0
        texcoord_data.u_rb = 1.0
        texcoord_data.v_rb = 1.0
        
        self.vbo_texture_coordinates = gl_vbo.VBO( texcoord_data )
        
        gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
        
        self.reproject( projection, projection_is_identity )
    
    def reproject( self, projection, projection_is_identity ):
        self.image_projected_rects = []
        for r in self.image_world_rects:
            if ( projection_is_identity ):
                left_bottom_projected = r[ 0 ]
                right_top_projected = r[ 1 ]
            else:
                left_bottom_projected = projection( r[ 0 ][ 0 ], r[ 0 ][ 1 ] )
                right_top_projected = projection( r[ 1 ][ 0 ], r[ 1 ][ 1 ] )
            self.image_projected_rects.append( ( left_bottom_projected, right_top_projected ) )
        
        for i, projected_rect in enumerate( self.image_projected_rects ):
            vertex_data = self.vbo_vertexes[ i ].data
            vertex_data.x_lb = projected_rect[ 0 ][ 0 ]
            vertex_data.y_lb = projected_rect[ 0 ][ 1 ]
            vertex_data.x_lt = projected_rect[ 0 ][ 0 ]
            vertex_data.y_lt = projected_rect[ 1 ][ 1 ]
            vertex_data.x_rt = projected_rect[ 1 ][ 0 ]
            vertex_data.y_rt = projected_rect[ 1 ][ 1 ]
            vertex_data.x_rb = projected_rect[ 1 ][ 0 ]
            vertex_data.y_rb = projected_rect[ 0 ][ 1 ]
            
            self.vbo_vertexes[ i ][ : np.alen( vertex_data ) ] = vertex_data
    
    def render( self, layer_index_base, pick_mode ):
        if ( self.vbo_vertexes == None ):
            return
        
        # images can't be selected with mouse click
        if ( pick_mode ):
            return
        
        for i, vbo in enumerate( self.vbo_vertexes ):
            gl.glEnable( gl.GL_TEXTURE_2D )
            gl.glBindTexture( gl.GL_TEXTURE_2D, self.textures[ i ] )
            
            gl.glEnableClientState( gl.GL_VERTEX_ARRAY ) # FIXME: deprecated
            vbo.bind()
            gl.glVertexPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
            
            gl.glEnableClientState( gl.GL_TEXTURE_COORD_ARRAY )
            self.vbo_texture_coordinates.bind()
            gl.glTexCoordPointer( 2, gl.GL_FLOAT, 0, None ) # FIXME: deprecated
            
            gl.glColor( 1.0, 1.0, 1.0, 1.0 )
            gl.glPolygonMode( gl.GL_FRONT_AND_BACK, gl.GL_FILL )
            gl.glDrawArrays( gl.GL_QUADS, 0, 4 )
            
            self.vbo_texture_coordinates.unbind()
            gl.glDisableClientState( gl.GL_TEXTURE_COORD_ARRAY )
            
            vbo.unbind()
            gl.glDisableClientState( gl.GL_VERTEX_ARRAY )
            
            gl.glBindTexture( gl.GL_TEXTURE_2D, 0 )
    
    def destroy( self ):
        if ( self.textures != None ):
            for texture in self.textures:
                gl.glDeleteTextures( np.array( [ texture ], np.uint32 ) )
        self.vbo_vertexes = None
        self.vbo_texture_coordinates = None
