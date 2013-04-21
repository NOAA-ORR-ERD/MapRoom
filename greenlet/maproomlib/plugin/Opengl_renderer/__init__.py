import wx
import os
import sys
import numpy as np
import wx.glcanvas as glcanvas
import OpenGL
if not __debug__ or hasattr( sys, "frozen" ):
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pyproj
import maproomlib.ui as ui
import maproomlib.utility as utility
from maproomlib.plugin.Opengl_renderer.Object_picker import Object_picker
from maproomlib.plugin.Opengl_renderer.Composite_renderer import Composite_renderer
from maproomlib.plugin.Opengl_renderer.Box_overlay import Box_overlay
from maproomlib.plugin.Opengl_renderer.Grid_lines_overlay import Grid_lines_overlay

# Make the job of dependency detectors easier by doing these imports
# explicitly.
import sys
if sys.platform.startswith( "win" ):
    import OpenGL.platform.win32
elif sys.platform.startswith( "darwin" ):
    import OpenGL.platform.darwin
else:
    import OpenGL.platform.glx


class Requirements_error( Exception ):
    pass


class Opengl_renderer( glcanvas.GLCanvas ):
    """
    A layer renderer that uses hardware-accelerated OpenGL.
    """
    PLUGIN_TYPE = "renderer"
    CURSOR_PATH = "ui/images/cursors"

    def __init__( self, parent, root_layer, command_stack ):
        glcanvas.GLCanvas.__init__(
            self,
            parent,
            attribList = (
                glcanvas.WX_GL_RGBA,
                glcanvas.WX_GL_DOUBLEBUFFER,
                glcanvas.WX_GL_MIN_ALPHA, 8,
            ),
        )

        # The canvas is initialized in response to the first paint event.
        self.initialized = False
        self.root_layer = root_layer
        self.inbox = ui.Wx_inbox()
        self.outbox = utility.Outbox()

        # The order of creation here is important, because among those
        # objects that bind to the same type of wx events, the last-created
        # object gets first crack at handling those issued events.
        self.viewport = ui.Viewport( self, self.change_viewport )
        self.background = ui.Background( self, root_layer, command_stack )
        self.picker = Object_picker(
            self, self.viewport, root_layer, command_stack
        )
        self.point_adder = ui.Point_adder(
            self, self.viewport, root_layer, command_stack
        )
        self.box_zoomer = ui.Box_zoomer(
            self, self.viewport, root_layer, command_stack
        )
        self.root_renderer = Composite_renderer(
            root_layer, root_layer, self.viewport, self, None, self.picker,
        )
        self.root_renderer.outbox.subscribe(
            self.inbox,
            request = ( "start_progress", "end_progress" ),
        )
        self.grid_lines_overlay = Grid_lines_overlay(
            self, self.viewport,
        )
        self.box_overlay = Box_overlay(
            self.box_zoomer, self, self.viewport,
        )
        self.picker.root_renderer = self.root_renderer
        self.transformer = None

        if sys.platform == "darwin":
            cursor_path = self.CURSOR_PATH
            if os.path.basename( os.getcwd() ) != "maproom":
                cursor_path = os.path.join( "maproom", cursor_path )

            self.grab_cursor = wx.Cursor(
                os.path.join( cursor_path, "grab.ico" ),
                type = wx.BITMAP_TYPE_ICO,
                hotSpotX = 7,
                hotSpotY = 7,
            )
        else:
            self.grab_cursor = wx.StockCursor( wx.CURSOR_SIZING )

        self.Bind( wx.EVT_PAINT, self.draw )
        self.Bind( wx.EVT_SIZE, self.resize )

        # Prevent flashing on Windows by doing nothing on an erase background
        # event.
        self.Bind( wx.EVT_ERASE_BACKGROUND, lambda event: None )

    def run( self, scheduler ):
        scheduler.add( self.root_renderer.run )
        scheduler.add( self.point_adder.run )
        scheduler.add( self.box_zoomer.run )
        scheduler.add( self.box_overlay.run )
        scheduler.add( self.background.run )

        self.root_layer.outbox.subscribe(
            self.inbox,
            request = (
                "layer_added",
                "pan_mode",
                "add_points_mode",
                "add_lines_mode",
            )
        )

        # Start by polling the root layer, in case it already has child
        # layers.
        self.root_layer.inbox.send(
            request = "get_layers",
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "layers" )
        for layer in message.get( "layers" ):
            self.init_viewport( layer )

        cursor = self.grab_cursor
        self.viewport.enable_drag( cursor )

        while True:
            message = self.inbox.receive(
                request = (
                    "layer_added",
                    "pan_mode",
                    "add_points_mode",
                    "add_lines_mode",
                    "start_progress",
                    "end_progress",
                    "take_viewport_screenshot",
                )
            )

            request = message.get( "request" )

            if request == "layer_added":
                self.init_viewport( message.get( "layer" ) )
            elif request == "pan_mode":
                cursor = self.grab_cursor
                self.viewport.enable_drag( cursor )
                self.picker.disable()
            elif request == "add_points_mode":
                cursor = wx.StockCursor( wx.CURSOR_CROSS )
                self.viewport.disable_drag( cursor )
                self.picker.enable()
            elif request == "add_lines_mode":
                cursor = wx.StockCursor( wx.CURSOR_PENCIL )
                self.viewport.disable_drag( cursor )
                self.picker.enable()
            elif request in ( "start_progress", "end_progress" ):
                self.outbox.send( **message )
            elif request == "take_viewport_screenshot":
                message.pop( "request" )
                self.take_viewport_screenshot( **message )

    def init_viewport( self, layer ):
        viewport_pixel_size = self.viewport.window.GetClientSize()
        viewport_aspect = \
            viewport_pixel_size[ 0 ] / float( viewport_pixel_size[ 1 ] )

        # If the layer has a specific pixel size, then use its projection
        # as the render projection.
        if hasattr( layer, "pixel_size" ):
            geo_origin = layer.origin
            geo_size = layer.size

            if self.transformer is None:
                self.transformer = utility.Transformer( layer.projection )

                self.root_renderer.inbox.send(
                    request = "transformer",
                    transformer = self.transformer,
                )
            elif self.transformer.projection != layer.projection:
                self.transformer.change_projection( layer.projection )
                self.root_renderer.inbox.send(
                    request = "projection_changed",
                )
            projection = layer.projection

        # Otherwise, the layer is vector data, so it can be reprojected.
        else:
            if hasattr( layer, "origin" ) and \
               hasattr( layer, "size" ):
                geo_origin = layer.origin
                geo_size = layer.size
                projection = layer.projection
            elif hasattr( layer, "get_dimensions" ):
                ( geo_origin, geo_size, projection ) = layer.get_dimensions()
            else:
                return

            if self.transformer is not None and \
               ( geo_origin is None or geo_size is None ):
                return

            if self.transformer and \
               projection.srs == self.transformer.projection.srs:
                return

            geo_origin = geo_origin or ( 0.0, 0.0 )
            geo_size = geo_size or ( 0.01, 0.01 )
            layer_size = geo_size or ( 0.01, 0.01 )

            latlong = pyproj.Proj( "+proj=latlong" )
            if projection.srs != latlong.srs:
                latlong_transformer = utility.Transformer( latlong )

                geo_corner = latlong_transformer.transform(
                    ( geo_origin[ 0 ] + geo_size[ 0 ],
                      geo_origin[ 1 ] + geo_size[ 1 ] ),
                    projection,
                )
                geo_origin = latlong_transformer.transform(
                    geo_origin, projection,
                )
                geo_size = (
                    geo_corner[ 0 ] - geo_origin[ 0 ],
                    geo_corner[ 1 ] - geo_origin[ 1 ],
                )

            render_projection = pyproj.Proj(
                "+proj=merc +lon_0=%s +lat_ts=%s +x_0=0 +y_0=0 +units=m" % \
                    ( geo_origin[ 0 ],
                      geo_origin[ 1 ] + geo_size[ 1 ] * 0.5 )
            )

            if self.transformer is None:
                self.transformer = utility.Transformer( render_projection )
                self.root_renderer.inbox.send(
                    request = "transformer",
                    transformer = self.transformer,
                )

        self.viewport.jump_geo_boundary(
            geo_origin,
            geo_size,
            projection,
            self.transformer,
        )

    def init( self ):
        if gl.glGetInteger( gl.GL_RED_BITS ) != 8 or \
           gl.glGetInteger( gl.GL_GREEN_BITS ) != 8 or \
           gl.glGetInteger( gl.GL_BLUE_BITS ) != 8 or \
           gl.glGetInteger( gl.GL_ALPHA_BITS ) != 8:
            raise Requirements_error(
                "Your display must support 32-bit color.",
            )

        gl.glClearColor( 1, 1, 1, 0 )

    def draw( self, event ):
        self.point_adder.set_hovered( *self.picker.hovered )

        dc = wx.PaintDC( self )
        self.SetCurrent()

        if not self.initialized:
            self.init()
            self.initialized = True

        gl.glClear( gl.GL_COLOR_BUFFER_BIT )

        self.root_renderer.render()

        self.set_screen_projection_matrix()
        self.grid_lines_overlay.render()
        self.box_overlay.render()
        self.set_render_projection_matrix()

        self.SwapBuffers()

        self.picker.render( self.root_renderer )

        event.Skip()

    def resize( self, event ):
        if not self.GetContext():
            return

        event.Skip()

        # Make sure the frame is shown before calling SetCurrent().
        self.Show()
        self.SetCurrent()

    def change_viewport( self, render_origin, render_size,
                         new_scale = False ):
        window_size = self.GetClientSize()

        if window_size[ 0 ] <= 0 or window_size[ 1 ] <= 0 or \
           render_size[ 0 ] <= 0 or render_size[ 1 ] <= 0:
            return

        gl.glViewport( 0, 0, window_size[ 0 ], window_size[ 1 ] )
        self.set_projection_matrix( render_origin, render_size )

        self.Refresh( False )

        self.root_renderer.inbox.discard( request = "update", force = new_scale )
        self.root_renderer.inbox.send( request = "update", force = new_scale )

    def set_projection_matrix( self, origin, size, flip = False ):
        gl.glMatrixMode( gl.GL_PROJECTION )
        gl.glLoadIdentity()

        if flip:
            glu.gluOrtho2D(
                origin[ 0 ],
                origin[ 0 ] + size[ 0 ],
                origin[ 1 ] + size[ 1 ],
                origin[ 1 ],
            )
        else:
            glu.gluOrtho2D(
                origin[ 0 ],
                origin[ 0 ] + size[ 0 ],
                origin[ 1 ],
                origin[ 1 ] + size[ 1 ],
            )

        gl.glMatrixMode( gl.GL_MODELVIEW )
        gl.glLoadIdentity()

    def set_render_projection_matrix( self ):
        if self.viewport.render_size[ 0 ] > 0 and \
           self.viewport.render_size[ 1 ] > 0:
            self.set_projection_matrix(
                self.viewport.render_origin,
                self.viewport.render_size,
            )

    def set_screen_projection_matrix( self ):
        window_size = self.viewport.pixel_size

        if window_size[ 0 ] > 0 and window_size[ 1 ] > 0:
            self.set_projection_matrix(
                ( 0, 0 ),
                window_size,
                flip = True, # GL coordinate system is flipped vertically
            )

    def take_viewport_screenshot( self, response_box ):
        window_size = self.GetClientSize()

        gl.glReadBuffer( gl.GL_FRONT )

        raw_data = gl.glReadPixels(
            x = 0,
            y = 0,
            width = window_size[ 0 ],
            height = window_size[ 1 ],
            format = gl.GL_RGB,
            type = gl.GL_UNSIGNED_BYTE,
            outputType = str,
        )

        bitmap = wx.BitmapFromBuffer(
            width = window_size[ 0 ],
            height = window_size[ 1 ],
            dataBuffer = raw_data,
        )

        image = wx.ImageFromBitmap( bitmap )

        # Flip the image vertically, because glReadPixel()'s y origin is at
        # the bottom and wxPython's y origin is at the top.
        screenshot = image.Mirror( horizontally = False )

        response_box.send(
            request = "viewport_screenshot",
            screenshot = screenshot,
        )

    def toggle_grid_lines( self ):
        self.grid_lines_overlay.toggle_shown()
        self.Refresh( False )

    def shutdown( self ):
        self.root_renderer.delete()
