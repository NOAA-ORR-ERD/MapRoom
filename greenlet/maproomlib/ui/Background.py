import wx
import maproomlib.ui


class Background:
    """
    Represents the background of the canvas, the area with no points, lines,
    or other selectable items. Typically created to receive events last so
    that it will only receive clicks if nothing else handles them.
    """
    def __init__( self, window, root_layer, command_stack ):
        self.window = window
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        self.mode = "pan_mode"
        self.mouse_press_position = None

        self.window.Bind(
            wx.EVT_LEFT_DOWN,
            self.mouse_pressed,
        )
        self.window.Bind(
            wx.EVT_LEFT_UP,
            self.mouse_released,
        )
        self.window.Bind(
            wx.EVT_RIGHT_UP,
            self.mouse_right_released,
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
        event.Skip()
        self.mouse_press_position = event.GetPosition()

    def mouse_released( self, event ):
        event.Skip()

        # If the mouse was released, and it wasn't dragged while the mouse
        # button was down, then clear the current selection.
        if self.mode == "pan_mode" and \
           self.mouse_press_position == event.GetPosition():
            self.command_stack.inbox.send(
                request = "start_command"
            )
            self.root_layer.inbox.send(
                request = "clear_selection",
            )

    def mouse_right_released( self, event ):
        event.Skip()

        # If the right mouse button was released, then clear the current
        # selection.
        self.command_stack.inbox.send(
            request = "start_command"
        )
        self.root_layer.inbox.send(
            request = "clear_selection",
        )
