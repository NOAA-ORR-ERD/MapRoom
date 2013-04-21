import wx
import bisect
import logging
import numpy as np
import OpenGL.GL as gl
import OpenGL.arrays.vbo as gl_vbo
import maproomlib.utility as utility
import maproomlib.ui as ui


class Object_picker:
    """
    Supports selection of objects by clicking on them with the mouse.
    """
    CHANNELS = 4   # number of channels in an RGBA color

    def __init__( self, window, viewport, root_layer, command_stack ):
        self.window = window
        self.viewport = viewport
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.frame_buffer = None
        self.render_buffer = None
        self.render_buffer_size = ( 0, 0 )
        self.start_drag_position = None
        self.drag_position = None
        self.hovered_renderer = None
        self.hovered_index = None
        self.colored_objects = {} # renderer -> ( index, original color )
        self.NORMAL_COLOR = utility.color_to_int( 0, 0, 0, 1 )
        self.HOVER_COLOR = utility.color_to_int( 1, 0, 0, 1 )
        self.BACKGROUND_COLOR = utility.color_to_int( 1, 1, 1, 1 )
        self.BACKGROUND_COLOR_STRING = chr( 255 ) * 3
        self.logger = logging.getLogger( __name__ )
        self.initialized = False
        self.enabled = False
        self.gl_fbo = None
        self.reset()

        self.window.Bind( wx.EVT_MOTION, self.mouse_moved )
        self.window.Bind( wx.EVT_LEFT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_LEFT_UP, self.mouse_released )
        self.window.Bind( wx.EVT_RIGHT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_RIGHT_UP, self.mouse_right_released )
        self.window.Bind( wx.EVT_LEFT_DCLICK, self.mouse_double_clicked )
        self.window.Bind( wx.EVT_LEAVE_WINDOW, self.mouse_left )

    def reset( self ):
        self.renderers = []
        self.renderer_start_colors = []
        self.start_color = self.NORMAL_COLOR

    def initialize( self ):
        if self.initialized: return

        try:
            import OpenGL.GL.EXT.framebuffer_object as gl_fbo

            if gl_fbo.glInitFramebufferObjectEXT():
                self.gl_fbo = gl_fbo
                self.logger.debug(
                    "GL_EXT_frameboffer_object support enabled."
                )
        except ImportError:
            pass

        if self.gl_fbo is None:
            self.logger.warning(
                "This system does not appear to support the " +
                "GL_EXT_frameboffer_object extension. " +
                "There may be some rendering artifacts."
            )

        self.initialized = True

    def enable( self ):
        self.enabled = True
        self.window.Refresh( False )

    def disable( self ):
        self.enabled = False
        self.hovered_renderer = None
        self.hovered_index = None

    def render( self, root_renderer ):
        """
        Render each object in a different color to an off-screen pick buffer.
        """
        if self.enabled is False:
            return

        self.initialize()
        window_size = self.window.GetSize()

        if self.frame_buffer is None:
            gl.glClear( gl.GL_COLOR_BUFFER_BIT )

        # If the window size has changed, our buffer needs to be recreated
        # at the new size.
        if self.frame_buffer and \
           self.render_buffer_size != window_size:
            self.gl_fbo.glDeleteFramebuffersEXT(
                1, np.array( [ self.frame_buffer ], dtype = np.uint32 ),
            )
            self.gl_fbo.glDeleteRenderbuffersEXT(
                1, np.array( [ self.render_buffer ], dtype = np.uint32 ),
            )
            self.frame_buffer = None
            self.render_buffer = None

        # Bind to the pick buffer, creating one if it doesn't yet exist.
        if self.gl_fbo and self.frame_buffer is None:
            self.frame_buffer = int( self.gl_fbo.glGenFramebuffersEXT( 1 ) )
            self.bind_frame_buffer()

            self.render_buffer = int( self.gl_fbo.glGenRenderbuffersEXT( 1 ) )
            self.gl_fbo.glBindRenderbufferEXT(
                self.gl_fbo.GL_RENDERBUFFER_EXT,
                self.render_buffer,
            )

            self.gl_fbo.glRenderbufferStorageEXT(
                self.gl_fbo.GL_RENDERBUFFER_EXT,
                gl.GL_RGBA,
                window_size[ 0 ],
                window_size[ 1 ],
            )

            self.gl_fbo.glFramebufferRenderbufferEXT(
                self.gl_fbo.GL_FRAMEBUFFER_EXT,
                self.gl_fbo.GL_COLOR_ATTACHMENT0_EXT,
                self.gl_fbo.GL_RENDERBUFFER_EXT,
                self.render_buffer,
            )
            self.render_buffer_size = window_size
        else:
            self.bind_frame_buffer()

        self.reset()

        # Render all objects with a different color for each object.
        gl.glClear( gl.GL_COLOR_BUFFER_BIT )
        root_renderer.render( pick_mode = True )
        self.unbind_frame_buffer()

        # If no renderers called bind(), then there are no objects to pick.
        if len( self.renderers ) == 0:
            return False

        return True

    def bind_frame_buffer( self ):
        if self.gl_fbo is None or self.enabled is False:
            return

        gl.glDrawBuffer( gl.GL_NONE );

        self.gl_fbo.glBindFramebufferEXT(
            gl.GL_FRAMEBUFFER_EXT,
            self.frame_buffer,
        )

    def unbind_frame_buffer( self ):
        if self.gl_fbo is None or self.enabled is False:
            return

        self.gl_fbo.glBindFramebufferEXT( self.gl_fbo.GL_FRAMEBUFFER_EXT, 0 )

        gl.glDrawBuffer( gl.GL_BACK );
        
    def mouse_moved( self, event ):
        self.pick( event )

    def pick( self, event ):
        if self.enabled is False or \
           ( self.gl_fbo and self.render_buffer is None ):
            event.Skip()
            return

        if event.LeftIsDown():
            if self.hovered_renderer and self.hovered_index is not None and \
               self.drag_position and self.hovered_renderer.layer is not None:
                position = self.viewport.pixel_to_geo(
                    event.GetPosition(),
                    self.hovered_renderer.layer.projection,
                )
                movement_vector = (
                    position[ 0 ] - self.drag_position[ 0 ],
                    position[ 1 ] - self.drag_position[ 1 ],
                )
                self.drag_position = position

                self.root_layer.inbox.send(
                    request = "move_selection",
                    movement_vector = movement_vector,
                )

                event.Skip( False )
            else:
                event.Skip()
            return

        # Get the color of the pixel at the mouse position. Work around an odd
        # bug in which the color is not read correctly from the first row of
        # pixels in the window (y coordinate 0).
        self.bind_frame_buffer()
        position = event.GetPosition()
        if position[ 1 ] == 0:
            color_string = None
        else:
            color_string = gl.glReadPixels(
                x = position[ 0 ],
                # glReadPixels() expects coordinates with a lower-left origin.
                y = self.window.GetSize()[ 1 ] - position[ 1 ],
                width = 1,
                height = 1,
                format = gl.GL_RGBA,
                type = gl.GL_UNSIGNED_BYTE,
                outputType = str,
            )

        # If the color string corresponds to the background, then no object
        # was moused over.
        if color_string is None or \
           color_string[ 0: 3 ] == self.BACKGROUND_COLOR_STRING:
            self.unbind_frame_buffer()
            if self.hovered_renderer == None and \
               self.hovered_index == None:
                event.Skip()
                return

            self.end_drag()
            self.hovered_renderer = None
            self.hovered_index = None
            self.window.Refresh( False )
            event.Skip()
            return

        # Find the object that corresponds to the moused-over color. Do this
        # by first finding the renderer containing objects with the range of
        # colors that the moused-over color is in. Then subtract that color
        # from the renderer's start color to get the object index.
        color = np.frombuffer( color_string, dtype = np.uint32 )[ 0 ]
        renderer_index = bisect.bisect( self.renderer_start_colors, color ) - 1
        if renderer_index == -1:
            self.unbind_frame_buffer()
            event.Skip()
            return

        renderer_start_color = self.renderer_start_colors[ renderer_index ]
        hovered_renderer = self.renderers[ renderer_index ]
        hovered_index = color - renderer_start_color

        self.window.SetCursor( wx.StockCursor( wx.CURSOR_HAND ) )
        self.unbind_frame_buffer()
        event.Skip( False )

        if hovered_renderer == self.hovered_renderer and \
           hovered_index == self.hovered_index:
            return

        self.hovered_renderer = hovered_renderer
        self.hovered_index = hovered_index
        self.window.Refresh( False )

    def end_drag( self ):
        if self.start_drag_position is None or self.drag_position is None:
            return False

        cumulative_movement_vector = (
            self.drag_position[ 0 ] - self.start_drag_position[ 0 ],
            self.drag_position[ 1 ] - self.start_drag_position[ 1 ],
        )

        self.drag_position = None
        self.start_drag_position = None

        if cumulative_movement_vector == ( 0.0, 0.0 ):
            return False

        self.root_layer.inbox.send(
            request = "move_selection",
            movement_vector = cumulative_movement_vector,
            cumulative = True,
        )

        return True

    def mouse_pressed( self, event ):
        if self.enabled is False:
            event.Skip()
            return

        self.pick( event )

        if self.hovered_renderer is None or self.hovered_index is None:
            event.Skip()
            return

        # Prevent other click handlers from handling this click.
        event.Skip( False )
        self.drag_position = self.viewport.pixel_to_geo(
            event.GetPosition(),
            self.hovered_renderer.layer.projection,
        )
        self.start_drag_position = self.drag_position

        # If the control/command or shift key is down, then add to the
        # selection rather than replacing it (see mouse_released() below).
        # This allows multi-selection.
        if event.CmdDown() or event.ShiftDown():
            return

        self.command_stack.inbox.send(
            request = "start_command"
        )
        self.root_layer.inbox.send(
            request = "replace_selection",
            layer = self.hovered_renderer.layer,
            object_indices = ( self.hovered_index, ),
        )

    def mouse_released( self, event ):
        if self.enabled is False:
            event.Skip()
            return

        dragged = self.end_drag()
        self.pick( event )

        if self.hovered_renderer is None or self.hovered_index is None:
            event.Skip()
            return

        event.Skip( False )
        if dragged:
            return

        if not event.CmdDown() and not event.ShiftDown():
            return

        self.command_stack.inbox.send(
            request = "start_command"
        )
        self.root_layer.inbox.send(
            request = "add_selection",
            layer = self.hovered_renderer.layer,
            object_indices = ( self.hovered_index, ),
            include_range = event.ShiftDown(),
        )

    def mouse_right_released( self, event ):
        if self.enabled is False:
            event.Skip()
            return

        self.pick( event )

        if self.hovered_renderer is None or self.hovered_index is None:
            event.Skip()
            return

        self.window.PopupMenu(
            ui.Object_context_menu(),
            event.GetPosition(),
        )

    def mouse_double_clicked( self, event ):
        if self.enabled is False:
            event.Skip()
            return

        if self.hovered_renderer is None or self.hovered_index is None:
            event.Skip()
            return

        # Prevent other click handlers from handling this double click.
        event.Skip( False )

    def mouse_left( self, event ):
        if self.enabled is False:
            event.Skip()
            return

        self.end_drag()
        event.Skip()

        if self.hovered_renderer == None and \
           self.hovered_index == None:
            return

        self.hovered_renderer = None
        self.hovered_index = None
        self.window.Refresh( False )

    def bind_pick_colors( self, color_buffer, renderer, object_count,
                          doubled = False ):
        # Set the color buffer with a different color for each object in the
        # renderer.
        if doubled:
            color_count = object_count // 2
        else:
            color_count = object_count

        color_data = np.arange(
            self.start_color,
            self.start_color + color_count,
            dtype = np.uint32,
        )

        if doubled:
            color_data = np.c_[ color_data, color_data ].reshape( -1 )

        color_buffer[ 0: object_count * self.CHANNELS ] = \
            color_data.view( dtype = np.uint8 )

        # Record the correspondence of colors to objects in the given renderer.
        self.renderers.append( renderer )
        self.renderer_start_colors.append( self.start_color )
        self.start_color += color_count

        gl.glEnableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
        color_buffer.bind()

        # FIXME: deprecated
        gl.glColorPointer( self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None )

    def set_pick_colors( self, colors, renderer ):
        # Alternative to bind_pick_colors() for renderers that need a little
        # more control over binding but still want to use the object picker.
        color_count = colors.shape[ 0 ]

        colors[ : ] = np.arange(
            self.start_color,
            self.start_color + color_count,
            dtype = np.uint32,
        )

        # Record the correspondence of colors to objects in the given renderer.
        self.renderers.append( renderer )
        self.renderer_start_colors.append( self.start_color )
        self.start_color += color_count

    def bind_colors( self, color_buffer, renderer, object_count,
                     doubled = False ):
        doubled = 2 if doubled else 1
        doubled_channels = self.CHANNELS * doubled

        # Restore the original color of the previously hovered object (if
        # any).
        if self.colored_objects.get( renderer ):
            ( object_index, color ) = self.colored_objects.pop( renderer )
            buffer_index = object_index * doubled_channels
            color_buffer[ buffer_index : buffer_index + doubled_channels ] = \
                np.array(
                    [ color ] * doubled,
                    dtype = np.uint32,
                ).view( np.uint8 )

        # Then, change the current hovered object to a specific hover color,
        # saving off its original color for later restoration.
        if renderer == self.hovered_renderer and \
           self.hovered_index is not None:
            buffer_index = self.hovered_index * doubled_channels
            color = np.array( color_buffer.data[
                buffer_index : buffer_index + self.CHANNELS
            ], dtype = np.uint8 ).view( dtype = np.uint32 )[ 0 ]
            self.colored_objects[ renderer ] = ( self.hovered_index, color )

            color_buffer[ buffer_index : buffer_index + doubled_channels ] = \
                np.array(
                    [ self.HOVER_COLOR ] * doubled,
                    dtype = np.uint32,
                ).view( np.uint8 )

        gl.glEnableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
        color_buffer.bind()
        gl.glColorPointer( self.CHANNELS, gl.GL_UNSIGNED_BYTE, 0, None ) # FIXME: deprecated

    def set_colors( self, colors, renderer ):
        # Alternative to bind_colors() for renderers that need a little more
        # control over binding but still want to use the object picker.

        # Restore the original color of the previously hovered object (if
        # any).
        if self.colored_objects.get( renderer ):
            ( object_index, color ) = self.colored_objects.pop( renderer )
            colors[ object_index ] = color

        # Then, change the current hovered object by simply lightening its
        # color, saving off its original color for later restoration.
        if renderer == self.hovered_renderer and \
           self.hovered_index is not None:
            color = colors[ self.hovered_index ]
            self.colored_objects[ renderer ] = ( self.hovered_index, color )
            color_bytes = colors[
                self.hovered_index : self.hovered_index + 1
            ].view( np.uint8 )
            color_bytes[ 0 ] = min( color_bytes[ 0 ] + 48, 255 )
            color_bytes[ 1 ] = min( color_bytes[ 1 ] + 48, 255 )
            color_bytes[ 2 ] = min( color_bytes[ 2 ] + 48, 255 )

    def unbind_colors( self, color_buffer ):
        gl.glDisableClientState( gl.GL_COLOR_ARRAY ) # FIXME: deprecated
        color_buffer.unbind()

    hovered = property( lambda self: ( self.hovered_renderer, self.hovered_index ) )
