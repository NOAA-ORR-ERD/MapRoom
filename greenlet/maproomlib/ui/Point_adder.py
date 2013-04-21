import wx
import sys
import maproomlib.ui


class Point_adder:
    """
    Allows the user to add points and lines by clicking (when in the
    appropriate mode).
    """
    def __init__( self, window, viewport, root_layer, command_stack ):
        self.window = window
        self.viewport = viewport
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        self.mode = "pan_mode"
        self.return_to_mode = None
        self.hovered_renderer = None
        self.hovered_index = None

        app = wx.GetApp()

        self.window.Bind(
            wx.EVT_LEFT_DOWN,
            self.mouse_pressed,
        )
        self.window.Bind(
            wx.EVT_RIGHT_DOWN,
            self.mouse_right_pressed,
        )
        app.Bind(
            wx.EVT_KEY_DOWN,
            self.key_pressed,
        )
        app.Bind(
            wx.EVT_KEY_UP,
            self.key_released,
        )

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = (
                "pan_mode",
                "add_points_mode",
                "add_lines_mode",
            )
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "pan_mode",
                    "add_points_mode",
                    "add_lines_mode",
                )
            )

            self.mode = message.get( "request" )

    def mouse_pressed( self, event ):
        import maproomlib.plugin as plugin

        if self.mode == "pan_mode" or self.viewport.uninitialized():
            event.Skip()
            return

        render_position = self.viewport.pixel_to_render( event.GetPosition() )
        points = plugin.Point_set_layer.make_points( 1, exact = True )
        points.x[ 0 ] = render_position[ 0 ]
        points.y[ 0 ] = render_position[ 1 ]

        if self.mode == "add_points_mode":
            self.add_point( points )
        elif self.mode == "add_lines_mode":
            self.add_line( points )

        self.window.SetFocus()
        event.Skip()

    def mouse_right_pressed( self, event ):
        event.Skip()

        # On Mac OS X, ctrl-left-click fires a ctrl-right-click event!
        if event.ControlDown():
            self.mouse_pressed( event )

    def add_point( self, points ):
        self.command_stack.inbox.send( request = "start_command" )
        self.root_layer.inbox.send(
            request = "add_points_to_selected",
            points = points,
            projection = self.viewport.transformer.projection,
            to_layer = self.hovered_renderer.layer if self.hovered_renderer \
                       else None,
            to_index = self.hovered_index,
        )

    def add_line( self, points ):
        self.command_stack.inbox.send( request = "start_command" )
        self.root_layer.inbox.send(
            request = "add_lines_to_selected",
            points = points,
            projection = self.viewport.transformer.projection,
            to_layer = self.hovered_renderer.layer if self.hovered_renderer \
                       else None,
            to_index = self.hovered_index,
        )

    def key_pressed( self, event ):
        if self.return_to_mode is not None:
            event.Skip()
            return

        key_code = event.GetKeyCode()
        self.return_to_mode = self.mode

        if key_code == wx.WXK_ALT:
            self.root_layer.inbox.send(
                request = "pan_mode",
            )

            # Prevent Windows from interpreting the alt key as "go to menu".
            event.Skip( False )
        else:
            event.Skip()

    def key_released( self, event ):
        if event.GetKeyCode() == wx.WXK_ALT:
            event.Skip( False )
        else:
            event.Skip()

        if self.return_to_mode is None:
            return

        if event.GetKeyCode() == wx.WXK_ALT:
            self.root_layer.inbox.send(
                request = self.return_to_mode,
            )

            self.return_to_mode = None

    def set_hovered( self, hovered_renderer, hovered_index ):
        self.hovered_renderer = hovered_renderer
        self.hovered_index = hovered_index
