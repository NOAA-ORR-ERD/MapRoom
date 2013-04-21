# coding=utf8

import sys
import math
import wx


class Distance_slider( wx.StaticBoxSizer ):
    SPACING = 5
    MINIMUM = 0.0
    MAXIMUM = 60.0
    INITIAL_VALUE = 0.6
    SLIDER_STEPS = 1000.0
    LOG_BASE = 10000.0
    MINUTE_TO_METERS = 1852 / 60.0

    def __init__( self, parent ):
        self.value = self.INITIAL_VALUE

        wx.StaticBoxSizer.__init__(
            self,
            wx.StaticBox(
                parent,
                label = "Distance Tolerance for Duplicate Points (lat minutes / nautical miles)",
            ),
            orient = wx.VERTICAL,
        )

        self.slider = wx.Slider(
            parent,
            wx.ID_ANY,
            minValue = 0,
            maxValue = self.SLIDER_STEPS,
        )
        self.slider.SetValue(
            math.log(
                ( self.value - self.MINIMUM ) /
                ( self.MAXIMUM - self.MINIMUM ) * self.LOG_BASE,
                self.LOG_BASE,
            ) * self.SLIDER_STEPS,
        )

        self.Add(
            self.slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.minutes_label_sizer = wx.BoxSizer( wx.HORIZONTAL )
        self.meters_label_sizer = wx.BoxSizer( wx.HORIZONTAL )

        self.minutes_label_sizer.Add(
            wx.StaticText( 
                parent,
                wx.ID_ANY,
                self.format_minutes( self.MINIMUM ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )
        self.meters_label_sizer.Add(
            wx.StaticText( 
                parent,
                wx.ID_ANY,
                self.format_meters( self.MINIMUM ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.minutes_value_label = wx.StaticText( 
            parent,
            wx.ID_ANY,
            self.format_minutes( self.value ),
            style = wx.ALIGN_CENTRE,
        )
        self.meters_value_label = wx.StaticText( 
            parent,
            wx.ID_ANY,
            self.format_meters( self.value ),
            style = wx.ALIGN_CENTRE,
        )

        value_font = self.minutes_value_label.GetFont()
        value_font.SetWeight( wx.FONTWEIGHT_BOLD )
        self.minutes_value_label.SetFont( value_font )

        self.minutes_label_sizer.Add(
            self.minutes_value_label,
            1, wx.EXPAND,
        )
        self.meters_label_sizer.Add(
            self.meters_value_label,
            1, wx.EXPAND,
        )

        self.minutes_label_sizer.Add(
            wx.StaticText(
                parent,
                wx.ID_ANY,
                self.format_minutes( self.MAXIMUM ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )
        self.meters_label_sizer.Add(
            wx.StaticText(
                parent,
                wx.ID_ANY,
                self.format_meters( self.MAXIMUM ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.Add(
            self.minutes_label_sizer,
            1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            border = self.SPACING,
        )
        self.Add(
            self.meters_label_sizer,
            1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            border = self.SPACING,
        )

        self.slider.Bind( wx.EVT_SCROLL, self.slider_moved )

    def slider_moved( self, event ):
        fraction = \
            self.LOG_BASE ** ( self.slider.GetValue() / self.SLIDER_STEPS ) \
            / self.LOG_BASE
        self.value = \
            ( self.MAXIMUM - self.MINIMUM ) * fraction + self.MINIMUM

        self.minutes_value_label.SetLabel( self.format_minutes( self.value ) )
        self.meters_value_label.SetLabel( self.format_meters( self.value ) )
        self.Layout()

    @staticmethod
    def format_minutes( value ):
        return ( u"%f" % value ).rstrip( u"0" ).rstrip( "." ) + u"′"

    @staticmethod
    def format_meters( value ):
        return u"%d m" % int( value * Distance_slider.MINUTE_TO_METERS )


class Depth_slider( wx.StaticBoxSizer ):
    SPACING = 5
    MINIMUM = 0
    MAXIMUM = 1000
    INITIAL_VALUE = 100

    def __init__( self, parent ):
        self.value = self.INITIAL_VALUE

        wx.StaticBoxSizer.__init__(
            self,
            wx.StaticBox(
                parent,
                label = "Depth Tolerance for Duplicate Points",
            ),
            orient = wx.VERTICAL,
        )

        self.slider = wx.Slider(
            parent,
            wx.ID_ANY,
            minValue = self.MINIMUM,
            maxValue = self.MAXIMUM,
        )
        self.slider.SetValue( self.INITIAL_VALUE )

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
                self.format_percent( self.MINIMUM ),
                style = wx.ALIGN_RIGHT,
            ),
            0, wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.value_label = wx.StaticText( 
            parent,
            wx.ID_ANY,
            self.format_percent( self.value ),
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
                self.format_percent( self.MAXIMUM ),
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

        self.value_label.SetLabel( self.format_percent( self.value ) )
        self.Layout()

    @staticmethod
    def format_percent( value ):
        return str( value ) + "%"


class Duplicate_finder( wx.Dialog ):
    SPACING = 15
    SLIDER_MIN_WIDTH = 400

    def __init__( self, parent_frame, root_layer, command_stack, viewport ):
        import maproomlib.ui as ui

        wx.Dialog.__init__(
            self, None, wx.ID_ANY, "Merge Duplicate Points",
            style = wx.DEFAULT_DIALOG_STYLE | wx.DIALOG_NO_PARENT,
        )
        self.SetIcon( parent_frame.GetIcon() )
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.viewport = viewport
        self.inbox = ui.Wx_inbox()
        self.list_points = []
        self.points_in_lines = None
        self.layer = None
        self.flag_layer = None

        self.outer_sizer = wx.BoxSizer( wx.VERTICAL )
        self.sizer = wx.BoxSizer( wx.VERTICAL )

        self.panel = wx.Panel( self, wx.ID_ANY )
        self.outer_sizer.Add( self.panel, 1, wx.EXPAND )

        self.distance_slider = Distance_slider( self.panel )
        self.distance_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.distance_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.depth_slider = Depth_slider( self.panel )
        self.depth_slider.SetMinSize( ( self.SLIDER_MIN_WIDTH, -1 ) )
        self.sizer.Add(
            self.depth_slider,
            0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.find_button_id = wx.NewId()
        self.find_button = wx.Button(
            self.panel,
            self.find_button_id,
            "Find Duplicates",
        )
        self.find_button.SetDefault()
        self.find_button_id = wx.NewId()
        self.sizer.Add(
            self.find_button, 0,
            wx.ALIGN_LEFT | wx.ALL,
            border = self.SPACING,
        )

        self.label = None

        self.panel.SetSizer( self.sizer )
        self.SetSizer( self.outer_sizer )

        self.sizer.Layout()
        self.Fit()
        self.Show()

        self.find_button.Bind( wx.EVT_BUTTON, self.find_duplicates )

    def find_duplicates( self, event ):
        self.inbox.send(
            request = "find_duplicates",
            distance_tolerance = self.distance_slider.value / 60.0,
            depth_tolerance = self.depth_slider.value,
        )

    def update_selection( self, event = None ):
        selected = self.list.GetFirstSelected()
        points = []

        while selected != -1:
            original_index = self.list.GetItem( selected ).GetData()
            if original_index != -1:
                points.extend( self.list_points[ original_index ] )
            selected = self.list.GetNextSelected( selected )

        if len( points ) == 0:
            self.remove_button.Enable( False )
            points = sum( [ list( d ) for d in self.list_points ], [] )
        else:
            self.remove_button.Enable( True )

        point_count = len( points )

        if point_count == 0:
            self.flag_layer.inbox.send(
                request = "clear_selection",
            )
            return

        self.flag_layer.inbox.send(
            request = "replace_selection",
            layer = self.layer,
            object_indices = tuple( set( points ) ),
        )

    def key_pressed( self, event ):
        key_code = event.GetKeyCode()

        if key_code == wx.WXK_DELETE:
            self.delete_current_groups()
        else:
            event.Skip()

    def delete_current_groups( self, event = None ):
        if len( self.list_points ) == 0:
            return

        selected = self.list.GetFirstSelected()
        to_delete = []

        while selected != -1:
            to_delete.append( selected )
            selected = self.list.GetNextSelected( selected )

        # Reversing is necessary so as not to screw up the indices of
        # subsequent selected groups when a previous group is deleted.
        for selected in reversed( to_delete ):
            original_index = self.list.GetItem( selected ).GetData()
            self.list.DeleteItem( selected )
            self.list_points[ original_index ] = []

        # If there's a focused group after the group deletions, then select
        # that group. This way the user can hit delete repeatedly to delete
        # a bunch of groups from the list.
        focused = self.list.GetFocusedItem()
        if focused != -1:
            self.list.Select( focused, on = True )

    def merge_clicked( self, event ):
        self.command_stack.inbox.send( request = "start_command" )
        self.layer.inbox.send(
            request = "merge_duplicates",
            indices = [ tuple( d ) for d in self.list_points if d ],
            points_in_lines = self.points_in_lines,
        )

        event.Skip()
        self.find_button.SetDefault()

    def close_clicked( self, event ):
        self.flag_layer.inbox.send(
            request = "clear_selection",
        )

        event.Skip()
        self.Hide()

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = ( "find_duplicates", "show", "close" ),
            )

            request = message.get( "request" )
            if request == "show":
                self.Show()
                self.Raise()
                if hasattr( self, "list" ):
                    self.update_selection()
                continue
            if request == "close":
                return

            self.find_button.Enable( False )
            self.find_button.SetLabel( "Finding..." )
            self.sizer.Layout()

            self.root_layer.inbox.send(
                request = "find_duplicates_in_selected",
                distance_tolerance = message.get( "distance_tolerance" ),
                depth_tolerance = message.get( "depth_tolerance" ),
                response_box = self.inbox,
            )

            try:
                message = self.inbox.receive(
                    request = ( "duplicates", "close" ),
                )
            except ( NotImplementedError, ImportError ), error:
                self.find_button.Enable( True )
                self.find_button.SetLabel( "Find Duplicates" )
                self.sizer.Layout()
                wx.MessageDialog(
                    self,
                    message = str( error ),
                    style = wx.OK | wx.ICON_ERROR,
                ).ShowModal()
                continue

            if message.get( "request" ) == "close":
                return

            duplicates = message.get( "duplicates" )
            self.points_in_lines = message.get( "points_in_lines" )
            self.layer = message.get( "layer" )
            self.flag_layer = message.get( "flag_layer" )

            # Display duplicate points in list.
            self.Freeze()
            self.create_results_area()
            self.display_results( duplicates, self.points_in_lines )

            self.find_button.Enable( True )
            self.find_button.SetLabel( "Find Duplicates" )
            self.sizer.Layout()
            self.Fit()
            self.Thaw()

    def create_results_area( self ):
        if self.label is not None:
            return

        self.label_text = \
            "Below is a list of possible duplicate points, " + \
            "grouped into pairs and displayed as point numbers. " + \
            "Click on a pair to highlight its points on the map."

        self.label = wx.StaticText(
            self.panel,
            wx.ID_ANY,
            self.label_text,
            style = wx.ST_NO_AUTORESIZE,
        )
        self.sizer.Add(
            self.label,
            0, wx.EXPAND | wx.LEFT | wx.RIGHT, border = self.SPACING,
        )

        self.list = wx.ListView( self.panel, wx.ID_ANY, style = wx.LC_LIST )
        self.list.SetMinSize( ( -1, 150 ) )

        self.sizer.Add(
            self.list, 1, wx.EXPAND | wx.ALL,
            border = self.SPACING,
        )

        self.remove_button_id = wx.NewId()
        self.remove_button = wx.Button(
            self.panel,
            self.remove_button_id,
            "Remove from Merge List",
        )
        self.remove_button.Enable( False )
        self.sizer.Add(
            self.remove_button, 0,
            wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.merge_label_text = \
            "Click Merge to merge each pair into a single point. " + \
            "Pairs that cannot be merged automatically are indicated " + \
            "in red and will be skipped during merging. (You can merge " + \
            "such points manually.)"

        self.merge_label = wx.StaticText(
            self.panel,
            wx.ID_ANY,
            self.merge_label_text,
            style = wx.ST_NO_AUTORESIZE,
        )
        self.sizer.Add(
            self.merge_label, 0,
            wx.ALIGN_LEFT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.button_sizer = wx.BoxSizer( wx.HORIZONTAL )

        self.merge_button_id = wx.NewId()
        self.merge_button = wx.Button(
            self.panel,
            self.merge_button_id,
            "Merge",
        )

        self.close_button_id = wx.NewId()
        self.close_button = wx.Button(
            self.panel,
            self.close_button_id,
            "Close",
        )

        # Dialog button ordering, by convention, is backwards on Windows.
        if sys.platform.startswith( "win" ):
            self.button_sizer.Add(
                self.merge_button, 0, wx.LEFT, border = self.SPACING,
            )
            self.button_sizer.Add(
                self.close_button, 0, wx.LEFT, border = self.SPACING,
            )
        else:
            self.button_sizer.Add(
                self.close_button, 0, wx.LEFT, border = self.SPACING,
            )
            self.button_sizer.Add(
                self.merge_button, 0, wx.LEFT, border = self.SPACING,
            )

        self.sizer.Add(
            self.button_sizer, 0, wx.ALIGN_RIGHT | wx.BOTTOM | wx.LEFT | wx.RIGHT,
            border = self.SPACING,
        )

        self.label.Wrap( self.sizer.GetSize()[ 0 ] - self.SPACING * 2 )
        self.merge_label.Wrap( self.sizer.GetSize()[ 0 ] - self.SPACING * 2 )

        self.list.Bind( wx.EVT_LIST_ITEM_SELECTED, self.update_selection )
        self.list.Bind( wx.EVT_LIST_ITEM_DESELECTED, self.update_selection )
        self.list.Bind( wx.EVT_LIST_KEY_DOWN, self.key_pressed )
        self.remove_button.Bind( wx.EVT_BUTTON, self.delete_current_groups )

        self.Bind(
            wx.EVT_BUTTON, self.merge_clicked, id = self.merge_button_id,
        )
        self.Bind(
            wx.EVT_BUTTON, self.close_clicked, id = self.close_button_id,
        )

    def display_results( self, duplicates, points_in_lines ):
        self.list.ClearAll()
        self.list_points = []
        MAX_POINTS_TO_LIST = 500
        pair_count = len( duplicates )

        if pair_count == 0:
            self.list.InsertStringItem( 0, "No duplicate points found." )
            self.list.SetItemData( 0, -1 )

            self.flag_layer.inbox.send(
                request = "clear_selection",
            )
            return

        if pair_count > MAX_POINTS_TO_LIST:
            self.list.InsertStringItem(
                0,
                "Too many duplicate points to list (%d pairs)." % pair_count
            )
            self.list.SetItemData( 0, -1 )

            self.list_points.extend( duplicates )
            self.update_selection()

            return

        for ( index, points ) in enumerate( duplicates ):
            # + 1 because points start from zero but users should see them
            # starting from one.
            formatted = ", ".join( [ str( p + 1 ) for p in points ] )
            self.list.InsertStringItem( index, formatted )

            # Associate the group's original index with it as list item data,
            # because its index can change if an item before it is deleted.
            self.list.SetItemData( index, index )
            self.list_points.append( points )

            # If each of the points in this group is within a line, then we
            # don't know how to merge it automatically. So distinguish it
            # from the groups we do know how to merge.
            for point in points:
                if point not in points_in_lines:
                    break
            else:
                self.list.SetItemTextColour( index, wx.RED )

        self.update_selection()
        self.merge_button.SetDefault()
