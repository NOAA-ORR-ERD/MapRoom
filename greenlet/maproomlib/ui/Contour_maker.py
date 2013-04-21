import wx
import sys
import maproomlib
import maproomlib.ui


class Slider( wx.StaticBoxSizer ):
    SPACING = 5

    def __init__( self, parent, title, minimum, maximum, default,
                  formatter = lambda value: str( value ) ):
        self.value = default
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.formatter = formatter

        wx.StaticBoxSizer.__init__(
            self,
            wx.StaticBox(
                parent,
                label = title,
            ),
            orient = wx.VERTICAL,
        )

        self.slider = wx.Slider(
            parent,
            wx.ID_ANY,
            minValue = self.minimum,
            maxValue = self.maximum,
        )
        self.slider.SetValue( self.default )

        self.Add(
            self.slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.label_sizer = wx.BoxSizer( wx.HORIZONTAL )

        self.label_sizer.Add(
            wx.StaticText( 
                parent,
                wx.ID_ANY,
                self.formatter( self.minimum ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.value_label = wx.StaticText( 
            parent,
            wx.ID_ANY,
            self.formatter( self.value ),
            style = wx.ALIGN_CENTRE,
        )

        value_font = self.value_label.GetFont()
        value_font.SetWeight( wx.FONTWEIGHT_BOLD )
        self.value_label.SetFont( value_font )

        self.label_sizer.Add(
            self.value_label,
            1, wx.EXPAND,
        )

        self.label_sizer.Add(
            wx.StaticText( 
                parent,
                wx.ID_ANY,
                self.formatter( self.maximum ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.Add(
            self.label_sizer,
            1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            border = self.SPACING,
        )

        self.slider.Bind( wx.EVT_SCROLL, self.slider_moved )

    def slider_moved( self, event ):
        self.value = self.slider.GetValue()

        self.value_label.SetLabel( self.formatter( self.value ) )
        self.Layout()


class Contour_maker( wx.Dialog ):
    SPACING = 15
    SLIDER_MIN_WIDTH = 400

    def __init__( self, frame, root_layer, command_stack ):
        wx.Dialog.__init__(
            self, None, wx.ID_ANY, "Contour Layer",
            style = wx.DEFAULT_DIALOG_STYLE | wx.DIALOG_NO_PARENT,
        )
        self.SetIcon( frame.GetIcon() )

        self.frame = frame
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()

        self.outer_sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer = wx.BoxSizer( wx.VERTICAL )

        self.panel = wx.Panel( self, wx.ID_ANY )
        self.outer_sizer.Add( self.panel, 1, wx.EXPAND )

        self.buffer_slider = Slider(
            parent = self.panel,
            title = "Grid Buffer Factor",
            minimum = 0,
            maximum = 100,
            default = 50,
            formatter = lambda value: str( value ) + "%",
        )
        self.buffer_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.buffer_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.grid_steps_slider = Slider(
            parent = self.panel,
            title = "Grid Dimensions",
            minimum = 16,
            maximum = 256,
            default = 96,
            formatter = lambda value: "%sx%s" % (value, value ),
        )
        self.grid_steps_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.grid_steps_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.neighbor_count_slider = Slider(
            parent = self.panel,
            title = "Nearest Neighbor Count",
            minimum = 10,
            maximum = 100,
            default = 30,
            formatter = lambda value: "%s points" % value,
        )
        self.neighbor_count_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.neighbor_count_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.level_light_slider = Slider(
            parent = self.panel,
            title = "Contour Light Level",
            minimum = 0,
            maximum = 100,
            default = 3,
            formatter = lambda value: str( value ) + "%",
        )
        self.level_light_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.level_light_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.level_medium_slider = Slider(
            parent = self.panel,
            title = "Contour Medium Level",
            minimum = 0,
            maximum = 100,
            default = 90,
            formatter = lambda value: str( value ) + "%",
        )
        self.level_medium_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.level_medium_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.level_heavy_slider = Slider(
            parent = self.panel,
            title = "Contour Heavy Level",
            minimum = 0,
            maximum = 100,
            default = 98,
            formatter = lambda value: str( value ) + "%",
        )
        self.level_heavy_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.level_heavy_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.button_sizer = wx.BoxSizer( wx.HORIZONTAL )

        self.contour_button_id = wx.NewId()
        self.contour_button = wx.Button(
            self.panel,
            self.contour_button_id,
            "Contour",
        )
        self.contour_button.SetDefault()

        self.close_button_id = wx.NewId()
        self.close_button = wx.Button(
            self.panel,
            self.close_button_id,
            "Close",
        )

        # Dialog button ordering, by convention, is backwards on Windows.
        if sys.platform.startswith( "win" ):
            self.button_sizer.Add(
                self.contour_button, 0, wx.LEFT, border = self.SPACING,
            )
            self.button_sizer.Add(
                self.close_button, 0, wx.LEFT, border = self.SPACING,
            )
        else:
            self.button_sizer.Add(
                self.close_button, 0, wx.LEFT, border = self.SPACING,
            )
            self.button_sizer.Add(
                self.contour_button, 0, wx.LEFT, border = self.SPACING,
            )

        self.sizer.Add(
            self.button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL,
            border = self.SPACING,
        )
        self.contour_button.SetDefault()

        self.panel.SetSizer( self.sizer )
        self.SetSizer( self.outer_sizer )

        self.sizer.Layout()
        self.Fit()
        self.Show()

        self.contour_button.Bind( wx.EVT_BUTTON, self.contour )
        self.close_button.Bind( wx.EVT_BUTTON, self.close )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = ( "contour", "show", "close" ),
            )

            request = message.pop( "request" )

            if request == "contour":
                self.contour_button.Enable( False )
                self.contour_button.SetLabel( "Contouring..." )
                self.sizer.Layout()

                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( 
                    request = "contour_selected",
                    response_box = self.inbox,
                    **message
                )

                try:
                    self.inbox.receive()
                except ( NotImplementedError ), error:
                    self.contour_button.Enable( True )
                    self.contour_button.SetLabel( "Contour" )
                    self.sizer.Layout()
                    wx.MessageDialog(
                        self.frame,
                        message = str( error ),
                        style = wx.OK | wx.ICON_ERROR,
                    ).ShowModal()
                    continue

                self.contour_button.Enable( True )
                self.contour_button.SetLabel( "Contour" )
                self.sizer.Layout()
            elif request == "show":
                self.Show()
                self.Raise()
            elif request == "close":
                return

    def contour( self, event ):
        self.inbox.send(
            request = "contour",
            levels = (
                self.level_light_slider.value / 100.0,
                self.level_medium_slider.value / 100.0,
                self.level_heavy_slider.value / 100.0,
            ),
            neighbor_count = self.neighbor_count_slider.value,
            grid_steps = self.grid_steps_slider.value,
            buffer_factor = self.buffer_slider.value / 100.0,
        )

    def close( self, event ):
        event.Skip()
        self.Hide()
