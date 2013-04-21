import wx
import os
import sys
import pyproj
import maproomlib.utility


class Viewport:
    """
    A means of tracking the portion of the source layers that the viewport
    currently views.

    :param viewport_window: the window that represents the viewport
    :type viewport_window: wx.Window
    :param callback: invoked whenever the viewport origin or size changes
    :type callback: callable that takes two arguments: origin and size

    Example usage::

        def callback( render_origin, render_size, new_scale = False ):
            ( render_origin_x, render_origin_y ) = origin
            ( render_width, render_height ) = size
            do_something()

        Viewport( viewport_window, callback )

    The ``render_origin`` parameter is the coordinate of the upper left of
    the viewport in render coordinates. The ``size`` parameter is the
    dimensions of the viewport in render coordinates. The boolean value of
    the ``new_scale`` parameter indicates whether the viewport scale has
    changed due to a zoom or bounds change. This is useful to know because
    the callback may want to delay a screen refresh if ``new_scale`` is
    False.

    .. attribute:: origin

        lower-left corner of the viewport in geographic coordinates

    .. attribute:: size

        dimensions of the viewport in geographic coordinates

    When the mouse leaves the viewport entirely, the following message is sent
    to the viewport's :attr:`outbox`::

        outbox.send( request = "mouse_left" )

    If the viewport is successfully zoomed, the following message is sent::

        outbox.send( request = "zoomed" )

    And if the maximum or minimum zoom levels are reached, the following
    message is sent::

        outbox.send( request = "zoom_limit_reached", message )

    The ``message`` value is a user-facing string about the zoom limit being
    reached.

    """
    ZOOM_OUT_LIMIT = 1500000 # in reference units
    ZOOM_IN_LIMIT = 1        # in reference units
    CURSOR_PATH = "ui/images/cursors"

    def __init__( self, viewport_window, callback ):
        self.window = viewport_window
        # define an area so we can handle the case of adding a layer
        # before we've loaded a file (mak Jan 5, 2011)
        self.render_origin = ( -5000, -5000 )
        self.render_size = ( 10000.0, 10000.0 )
        self.pixel_size = ( 0.0, 0.0 )
        # self.transformer = None
        
        self.outbox = maproomlib.utility.Outbox()

        self.callback = callback
        self.last_drag_position = None
        self.background_cursor = wx.StockCursor( wx.CURSOR_ARROW )
        self.drag_enabled = True
        self.reference_length = 0
        self.reference_projection = pyproj.Proj(
            "+proj=merc +x_0=0 +y_0=0 +units=m"
            # "+proj=latlong"
        )
        # define a transformer so we can handle the case of adding a layer
        # before we've loaded a file (mak Jan 5, 2011)
        self.transformer = maproomlib.utility.Transformer( self.reference_projection )
 
        frame = wx.GetApp().GetTopWindow()

        if sys.platform == "darwin":
            cursor_path = self.CURSOR_PATH
            if os.path.basename( os.getcwd() ) != "maproom":
                cursor_path = os.path.join( "maproom", cursor_path )

            self.drag_cursor = wx.Cursor(
                os.path.join( cursor_path, "drag.ico" ),
                type = wx.BITMAP_TYPE_ICO,
                hotSpotX = 7,
                hotSpotY = 7,
            )
        else:
            self.drag_cursor = wx.StockCursor( wx.CURSOR_SIZING )

        self.window.Bind( wx.EVT_SIZE, self.resized )
        self.window.Bind( wx.EVT_LEFT_DCLICK, self.mouse_double_clicked )
        self.window.Bind( wx.EVT_LEFT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_LEFT_UP, self.mouse_released )
        self.window.Bind( wx.EVT_MOTION, self.mouse_moved )
        self.window.Bind( wx.EVT_LEAVE_WINDOW, self.mouse_left )
        self.window.Bind( wx.EVT_MOUSEWHEEL, self.mouse_wheel_scrolled )
        frame.Bind( wx.EVT_MOUSEWHEEL, self.mouse_wheel_scrolled )
        self.window.SetFocus()

    def enable_drag( self, background_cursor ):
        self.drag_enabled = True
        self.background_cursor = background_cursor
        self.window.SetCursor( background_cursor )

    def disable_drag( self, background_cursor ):
        self.drag_enabled = False
        self.background_cursor = background_cursor
        self.window.SetCursor( background_cursor )

    def uninitialized( self ):
        return self.render_size == ( 0.0, 0.0 )

    def jump_geo_center( self, geo_center_point, geo_projection ):
        try:
            render_center_point = self.transformer.transform(
                geo_center_point,
                geo_projection,
            )
        # If pyproj fails to perform the transformation (e.g. due to lat or
        # long exceeding limits), then bail without jumping.
        except RuntimeError:
            return

        self.render_origin = (
            render_center_point[ 0 ] - self.render_size[ 0 ] * 0.5,
            render_center_point[ 1 ] - self.render_size[ 1 ] * 0.5,
        )

        self.callback( self.render_origin, self.render_size, new_scale = False )

    def jump_geo_boundary( self, geo_origin, geo_size, geo_projection,
                           transformer = None ):
        if transformer is not None:
            self.pixel_size = self.window.GetClientSize()
            self.transformer = transformer

        if geo_size == ( 0.0, 0.0 ):
            return

        try:
            render_origin = self.transformer.transform(
                geo_origin,
                geo_projection,
            )

            render_lower_right = self.transformer.transform(
                (
                    geo_origin[ 0 ] + geo_size[ 0 ],
                    geo_origin[ 1 ] + geo_size[ 1 ],
                ),
                geo_projection,
            )
        except RuntimeError:
            return

        render_size = (
            render_lower_right[ 0 ] - render_origin[ 0 ],
            render_lower_right[ 1 ] - render_origin[ 1 ],
        )

        self.jump_render_boundary( render_origin, render_size )

    def jump_render_boundary( self, render_origin, render_size ):
        if self.reference_length != 0 and \
           self.reference_length <= self.ZOOM_IN_LIMIT:
            self.outbox.send(
                request = "zoom_limit_reached",
                message = "The maximum zoom level has been reached.",
            )
            return

        self.render_origin = render_origin

        current_aspect = self.pixel_size[ 0 ] / float( self.pixel_size[ 1 ] )
        new_aspect = render_size[ 0 ] / render_size[ 1 ]

        if current_aspect > new_aspect:
            self.render_size = (
                    render_size[ 1 ] * current_aspect,
                    render_size[ 1 ],
                )
        else:
            self.render_size = (
                render_size[ 0 ],
                render_size[ 0 ] / current_aspect,
            )

        # Move the origin so that the loaded layer is centered vertically and
        # horizontally within the viewport.
        self.render_origin = (
            self.render_origin[ 0 ] + \
                ( render_size[ 0 ] - self.render_size[ 0 ] ) * 0.5,
            self.render_origin[ 1 ] + \
                ( render_size[ 1 ] - self.render_size[ 1 ] ) * 0.5,
        )

        self.update_reference_render_length()
        self.callback( self.render_origin, self.render_size, new_scale = True )

    def resized( self, event = None ):
        if event is None:
            pixel_size = self.window.GetClientSize()
        else:
            pixel_size = event.Size
            event.Skip()

        if self.render_size[ 0 ] <= 0 or self.render_size[ 1 ] <= 0:
            return

        if self.pixel_size[ 0 ] > 0 and self.pixel_size[ 1 ] > 0:
            # Determine how much the window size scaled since the last
            # update.
            resize_factor = (
                pixel_size[ 0 ] / float( self.pixel_size[ 0 ] ),
                pixel_size[ 1 ] / float( self.pixel_size[ 1 ] ),
            )

            # Account for the fact that viewport pixel coordinates have a
            # top-left origin, while geographic coordinates have a bottom-left
            # origin.
            vertical_change = self.render_size[ 1 ] * resize_factor[ 1 ] - \
                              self.render_size[ 1 ]
            self.render_origin = (
                self.render_origin[ 0 ],
                self.render_origin[ 1 ] - vertical_change,
            )

            # Scale the source size accordingly.
            self.render_size = (
                self.render_size[ 0 ] * resize_factor[ 0 ],
                self.render_size[ 1 ] * resize_factor[ 1 ],
            )

        self.pixel_size = pixel_size
        self.update_reference_render_length()

        self.callback( self.render_origin, self.render_size, new_scale = False )

    def mouse_double_clicked( self, event ):
        # Zoom in one step, centered on the click position.
        self.zoom(
            1,
            focus_point = event.GetPosition(),
            center_viewport = True,
        )

    def mouse_pressed( self, event ):
        event.Skip()
        if not self.drag_enabled:
            return

        self.window.SetCursor( self.drag_cursor )
        self.last_drag_position = event.GetPosition()
        self.window.SetFocus()

    def mouse_released( self, event ):
        event.Skip()
        if not self.drag_enabled:
            return

        if self.last_drag_position:
            self.window.SetCursor( self.background_cursor )
            self.last_drag_position = None

    def mouse_moved( self, event ):
        event.Skip()

        if self.pixel_size[ 0 ] == 0 or self.pixel_size[ 1 ] == 0:
            return

        position = event.GetPosition()

        if event.LeftIsDown() is False or self.last_drag_position is None:
            self.window.SetCursor( self.background_cursor )
            return

        if not self.drag_enabled:
            self.window.SetCursor( self.background_cursor )
            return

        delta = (
            position[ 0 ] - self.last_drag_position[ 0 ],
            position[ 1 ] - self.last_drag_position[ 1 ],
        )

        render_delta = self.pixel_size_to_render_size( delta )

        self.render_origin = (
            self.render_origin[ 0 ] - render_delta[ 0 ],
            self.render_origin[ 1 ] - render_delta[ 1 ],
        )

        self.last_drag_position = position

        self.callback( self.render_origin, self.render_size, new_scale = False )

    def mouse_left( self, event ):
        self.window.SetCursor( wx.StockCursor( wx.CURSOR_ARROW ) )
        wx.SetCursor( wx.NullCursor )
        self.outbox.send(
            request = "mouse_left",
        )

    def mouse_wheel_scrolled( self, event ):
        rotation = event.GetWheelRotation()
        delta = event.GetWheelDelta()
        if delta == 0:
            return

        # For some reason, event.GetPosition() is incorrect the first time
        # this function is called. So manually calculate the mouse position
        # relative to the viewport window.
        mouse_state = wx.GetMouseState()
        window_screen_position = self.window.GetScreenPosition()

        focus_point = (
            mouse_state.GetX() - window_screen_position[ 0 ],
            mouse_state.GetY() - window_screen_position[ 1 ],
        )

        self.zoom(
            rotation / delta,
            focus_point = focus_point,
            center_viewport = False,
        )

    def zoom( self, change, focus_point = None, center_viewport = True ):
        # If there's no size, then we can't zoom.
        if 0.0 in self.render_size or change == 0:
            return

        ZOOM_FACTOR = 2.0

        center_point = (
            self.render_origin[ 0 ] + self.render_size[ 0 ] * 0.5,
            self.render_origin[ 1 ] + self.render_size[ 1 ] * 0.5,
        )

        if focus_point is not None:
            focus_point = self.pixel_to_render( focus_point )

        if change > 0:
            render_size = (
                self.render_size[ 0 ] / ZOOM_FACTOR,
                self.render_size[ 1 ] / ZOOM_FACTOR,
            )

            if center_viewport is True:
                if focus_point:
                    center_point = focus_point
            else:
                center_point = (
                    ( center_point[ 0 ] + focus_point[ 0 ] ) / ZOOM_FACTOR,
                    ( center_point[ 1 ] + focus_point[ 1 ] ) / ZOOM_FACTOR,
                )

            render_origin = (
                center_point[ 0 ] - render_size[ 0 ] * 0.5,
                center_point[ 1 ] - render_size[ 1 ] * 0.5,
            )

            reference_length = self.calculate_reference_render_length(
                render_origin,
                render_size,
            )

            if reference_length is None or reference_length <= self.ZOOM_IN_LIMIT:
                self.outbox.send(
                    request = "zoom_limit_reached",
                    message = "The maximum zoom level has been reached.",
                )
                return

        elif change < 0:
            render_size = (
                self.render_size[ 0 ] * ZOOM_FACTOR,
                self.render_size[ 1 ] * ZOOM_FACTOR,
            )

            if center_viewport is True:
                if focus_point:
                    center_point = focus_point
            else:
                center_point = (
                    center_point[ 0 ] * ZOOM_FACTOR - focus_point[ 0 ],
                    center_point[ 1 ] * ZOOM_FACTOR - focus_point[ 1 ],
                )

            render_origin = (
                center_point[ 0 ] - render_size[ 0 ] * 0.5,
                center_point[ 1 ] - render_size[ 1 ] * 0.5,
            )

            reference_length = self.calculate_reference_render_length(
                render_origin,
                render_size,
            )

            if reference_length is None or reference_length >= self.ZOOM_OUT_LIMIT:
                self.outbox.send(
                    request = "zoom_limit_reached",
                    message = "The minimum zoom level has been reached.",
                )
                return

        self.render_size = render_size
        self.render_origin = render_origin
        self.reference_length = reference_length

        self.callback( self.render_origin, self.render_size, new_scale = True )

        self.outbox.send(
            request = "zoomed",
        )

    def pixel_to_render( self, pixel ):
        # Note the reversing of the y coordinate. This is due to the fact that
        # viewport pixel coords are origin top-left, while render coords are
        # origin bottom-left.
        return (
            pixel[ 0 ] /
                ( self.pixel_size[ 0 ] / float( self.render_size[ 0 ] ) ) +
                self.render_origin[ 0 ],
            ( self.pixel_size[ 1 ] - pixel[ 1 ] ) /
                ( self.pixel_size[ 1 ] / float( self.render_size[ 1 ] ) ) +
                self.render_origin[ 1 ],
        )

    def pixel_to_geo( self, pixel, geo_projection ):
        if self.transformer is None or self.render_size[ 0 ] == 0 or \
           self.render_size[ 1 ] == 0:
            return None

        return self.transformer.reverse_transform(
            self.pixel_to_render( pixel ),
            geo_projection,
        )

    def pixel_size_to_render_size( self, pixel_size ):
        if self.render_size[ 0 ] == 0 or self.render_size[ 1 ] == 0:
            return ( 0, 0 )

        return (
            pixel_size[ 0 ] /
                ( self.pixel_size[ 0 ] / float( self.render_size[ 0 ] ) ),
            -pixel_size[ 1 ] /
                ( self.pixel_size[ 1 ] / float( self.render_size[ 1 ] ) ),
        )

    def pixel_sizes_to_render_sizes( self, pixel_widths = None,
                                     pixel_heights = None ):
        if pixel_widths is not None:
            pixel_widths = pixel_widths / \
                ( self.pixel_size[ 0 ] / float( self.render_size[ 0 ] ) )

        if pixel_heights is not None:
            pixel_heights = pixel_heights / \
                ( self.pixel_size[ 1 ] / float( self.render_size[ 1 ] ) )

        return ( pixel_widths, pixel_heights )

    def pixel_size_to_geo_size( self, pixel_size, geo_projection ):
        render_size = self.pixel_size_to_render_size( pixel_size )

        return self.geo_size(
            geo_projection, self.render_origin, render_size,
        )

    def reference_render_length( self ):
        return self.reference_length

    def update_reference_render_length( self ):
        self.reference_length = self.calculate_reference_render_length(
            self.render_origin,
            self.render_size,
        )

    def calculate_reference_render_length( self, render_origin, render_size ):
        """
        Calculate a reference length in render units based on the current
        viewport and return it. This is useful for determining whether to
        show or hide something based on the current viewport zoom level.
        """
        REFERENCE_PIXEL_LENGTH = 100.0

        # Convert the size to a standard reference projection so that the
        # reference_render_length is the same regardless of the actual render
        # projection. This is necessary because some render projections are in
        # meters while others are just plain lat-long coordinates.
        try:
            reference_size = self.geo_size(
                self.reference_projection,
                render_origin,
                render_size,
            )
        # This error can occur if the dimensions given to pyproj result in
        # invalid lat-long coordinates once transformed.
        except RuntimeError:
            return None

        return min(
            REFERENCE_PIXEL_LENGTH /
                abs( self.pixel_size[ 0 ] / float( reference_size[ 0 ] ) ),
            REFERENCE_PIXEL_LENGTH /
                abs( self.pixel_size[ 1 ] / float( reference_size[ 1 ] ) ),
        )

    def geo_origin( self, geo_projection, render_origin = None ):
        if render_origin is None:
            render_origin = self.render_origin

        return self.transformer.reverse_transform(
            render_origin,
            geo_projection,
        )

    def geo_size( self, geo_projection, render_origin = None,
                  render_size = None ):
        if render_size is None:
            render_size = self.render_size

        if render_origin is None:
            render_origin = self.render_origin

        geo_point = self.transformer.reverse_transform( (
            render_size[ 0 ] + render_origin[ 0 ],
            render_size[ 1 ] + render_origin[ 1 ],
        ), geo_projection )

        geo_origin = self.geo_origin( geo_projection, render_origin )

        return (
            geo_point[ 0 ] - geo_origin[ 0 ],
            geo_point[ 1 ] - geo_origin[ 1 ],
        )
