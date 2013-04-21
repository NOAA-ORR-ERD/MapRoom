import wx
import maproomlib.ui
import maproomlib.utility


class Box_zoomer:
    """
    Allows the user to zoom in by selecting a rectangular area.
    """
    def __init__( self, window, viewport, root_layer, command_stack ):
        self.window = window
        self.viewport = viewport
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        self.outbox = maproomlib.utility.Outbox()
        self.selecting = False
        self.start_position = None

        self.window.Bind( wx.EVT_LEFT_DOWN, self.mouse_pressed )
        self.window.Bind( wx.EVT_LEFT_UP, self.mouse_released )
        self.window.Bind( wx.EVT_MOTION, self.mouse_moved )

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = "start_zoom_box",
        )

        while True:
            message = self.inbox.receive(
                request = "start_zoom_box",
            )

            self.window.SetCursor( wx.StockCursor( wx.CURSOR_MAGNIFIER ) )
            self.selecting = True

    def mouse_pressed( self, event ):
        if not self.selecting:
            event.Skip()
            return

        if self.viewport.uninitialized():
            self.root_layer.inbox.send(
                request = "end_zoom_box",
            )
            return

        self.window.CaptureMouse()

        self.start_position = event.GetPosition()

        self.outbox.send(
            request = "start_box",
            position = self.start_position,
        )

        event.Skip( False )

    def mouse_released( self, event ):
        if not self.selecting:
            event.Skip()
            return

        if self.start_position is None:
            event.Skip( False )
            return

        self.window.ReleaseMouse()

        self.selecting = False
        end_position = event.GetPosition()

        self.root_layer.inbox.send(
            request = "end_zoom_box",
        )
        self.outbox.send(
            request = "end_box",
            position = end_position,
        )

        if self.start_position != end_position:
            render_start_position = self.viewport.pixel_to_render(
                self.start_position,
            )
            render_end_position = self.viewport.pixel_to_render(
                end_position,
            )

            # Take into account the fact that the start and end positions can
            # indicate any two corners of a box.
            render_origin = (
                min( render_start_position[ 0 ], render_end_position[ 0 ] ),
                min( render_start_position[ 1 ], render_end_position[ 1 ] ),
            )

            render_corner = (
                max( render_start_position[ 0 ], render_end_position[ 0 ] ),
                max( render_start_position[ 1 ], render_end_position[ 1 ] ),
            )

            render_size = (
                render_corner[ 0 ] - render_origin[ 0 ],
                render_corner[ 1 ] - render_origin[ 1 ],
            )

            if render_size[ 0 ] != 0 and render_size[ 1 ] != 0:
                self.viewport.jump_render_boundary(
                    render_origin,
                    render_size,
                )

        self.start_position = None
        self.selecting = False
        self.window.SetCursor( wx.StockCursor( wx.CURSOR_ARROW ) )
        event.Skip( False )

    def mouse_moved( self, event ):
        if not self.selecting:
            event.Skip()
            return

        if self.start_position is None:
            event.Skip( False )
            return

        position = event.GetPosition()

        self.outbox.send(
            request = "move_box",
            position = position,
        )
        event.Skip( False )
