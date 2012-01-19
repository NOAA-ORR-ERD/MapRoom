import os
import os.path
import sys
import math
import wx
import wx.glcanvas as glcanvas
import pyproj
from ui.Menu_bar import Menu_bar
from ui.Tool_bar import Tool_bar
from ui.Properties_panel import Properties_panel
from ui.Triangle_dialog import Triangle_dialog
from ui.Merge_layers_dialog import Merge_layers_dialog
from ui.Merge_duplicate_points_dialog import Merge_duplicate_points_dialog
import Layer_manager
import Layer_tree_control
import Editor
import lon_lat_grid
import library.Opengl_renderer
import library.rect as rect
import app_globals

"""
    maproom to-do list (June 24, 2011)
    
    - finish panel items in Editor
    - save layer to .mrv file (using xml)
    - read layer from .mrv file
    - triangulation (basic workflow; just adds triangle objects to current layer)
    - read le file into multi-layer file
    - contouring (basic workflow; just adds polygons to current layer)
    - merge verdat points (and lines?) between to layers into a new layer
    - remove duplicate points in layer (within tolerance)
     
    - delete layer    
    - create new layer
"""

class Application( wx.App ):
    """
    The UI for the Maproom application.
    """
    DEFAULT_FRAME_SIZE = ( 1000, 750 )
    LEFT_SASH_POSITION = 200
    TOP_SASH_POSITION = 250
    IMAGE_PATH = "ui/images"
    ICON_FILENAME = os.path.join( IMAGE_PATH, "maproom.ico" )
    
    MODE_PAN = 0
    MODE_ZOOM_RECT = 1
    MODE_EDIT_POINTS = 2
    MODE_EDIT_LINES = 3
    
    frame = None
    menu_bar = None
    tool_bar = None
    renderer_splitter = None
    properties_splitter = None
    renderer = None # the glcanvas
    opengl_renderer = None # the library.Opengl_renderer.Opengl_renderer()
    layer_tree_panel = None
    properties_panel = None
    layer_tree_control = None
    status_bar = None
    
    triangle_dialog_box = None
    merge_duplicate_points_dialog_box = None
    
    mode = MODE_PAN
    hand_cursor = None
    hand_closed_cursor = None
    forced_cursor = None
    
    lon_lat_grid = None
    lon_lat_grid_shown = True
    
    mouse_is_down = False
    is_alt_key_down = False
    selection_box_is_being_defined = False;
    mouse_down_position = ( 0, 0 )
    mouse_move_position = ( 0, 0 )
    
    # two variables keep track of what's visible on the screen:
    # (1) the projected point at the center of the screen
    projected_point_center = ( 0, 0 )
    # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)
    projected_units_per_pixel = 10000
    # the projection to go from lon/lat points to projected points, and vice versa;
    # projection( lon, lat ) = ( x, y ) where ( x, y ) are meters from ( 0, 0 );
    #   for mercator projection
    #       the world is 40075016.6855801 meters across (its actual width), and infintely tall;
    #       89 degrees north = 30198185.169877 meters up, 89 degrees south = 30198185.169877 meters down;
    #       we treat 89 degrees north and 89 degrees south as the ends of the earth
    #   for lon/lat projection (i.e., "no projection")
    #       the world is 360 degrees across, and 180 degrees tall;
    #
    #   16004_1.KAP = mercator, meters (Alaska)
    #           Upper Left  ( -157413.563,11979777.582) (157d24'50.65"W, 72d43'40.06"N) (-157.414069444444, 72.7277944444444 )
    #           Lower Left  ( -157413.250,10239431.419) (157d24'50.64"W, 67d25'38.94"N) (-157.414066666667, 67.4274833333333 )
    #           Upper Right ( 1987191.738,11979774.130) (138d 8'55.51"W, 72d43'40.03"N) (-138.148752777778, 72.7277861111111 )
    #           Lower Right ( 1987192.051,10239427.967) (138d 8'55.50"W, 67d25'38.90"N) (-138.14875, 67.4274722222222 )
    #   Center      (  914889.244,11109602.774) (147d46'53.08"W, 70d14'52.25"N)    #   14771_1.KAP = polyconic
    #   gs_09apr16_0227_mult_geo.png = longlat
    #   NOAA18649.png = longlat
    # we default to mercator projection and only switch to longlat projection if we load a longlat raster and
    # don't already have a mercator raster loaded
    projection = pyproj.Proj( "+proj=merc +units=m +over" )
    # for longlat projection, apparently someone decided that since the projection
    # is the identity, it might as well do something and so it returns the coordinates as
    # radians instead of degrees; so here we use this variable to avoid using the longlat projection
    projection_is_identity = False
    
    is_initialized = False
    is_closing = False
    
    def __init__( self, init_filenames ):
        app_globals.application = self
        app_globals.layer_manager = Layer_manager.Layer_manager()
        app_globals.editor = Editor.Editor()
        self.init_filenames = init_filenames
        wx.App.__init__( self, False )
        self.refresh()
    
    def OnInit( self ):
        self.frame = wx.Frame( None, wx.ID_ANY, "Maproom" )
        self.frame.SetIcon( wx.Icon( self.ICON_FILENAME, wx.BITMAP_TYPE_ICO ) )
        self.frame.SetSizeHints( 250, 250 )
        # self.log_viewer = Log_viewer( self.frame )
        
        p = os.path.join( self.IMAGE_PATH, "cursors", "hand.ico" )
        self.hand_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
        p = os.path.join( self.IMAGE_PATH, "cursors", "hand_closed.ico" )
        self.hand_closed_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
        
        self.lon_lat_grid = lon_lat_grid.Lon_lat_grid()
        
        self.renderer_splitter = wx.SplitterWindow(
            self.frame,
            wx.ID_ANY,
            style = wx.SP_3DSASH,
        )
        self.renderer_splitter.SetMinimumPaneSize( 20 )
        
        self.properties_splitter = wx.SplitterWindow(
            self.renderer_splitter,
            wx.ID_ANY,
            style = wx.SP_3DSASH,
        )
        self.properties_splitter.SetMinimumPaneSize( 20 )
        
        self.menu_bar = Menu_bar()
        self.frame.SetMenuBar( self.menu_bar )
        
        self.tool_bar = Tool_bar()
        self.frame.SetToolBar( self.tool_bar )
        
        self.status_bar = self.frame.CreateStatusBar()
        
        """
        self.progress_tracker = ui.Progress_tracker(
            self.status_bar,
            self.root_layer,
            self.renderer,
        )
        """
        
        """
        self.mouse_tracker = ui.Mouse_tracker(
            self.root_layer,
            self.status_bar,
            self.renderer.viewport,
        )
        scheduler.add( self.mouse_tracker.run )

        self.flag_jumper = ui.Flag_jumper(
            self.root_layer,
            self.renderer.viewport,
        )
        scheduler.add( self.flag_jumper.run )
        """
        
        self.renderer = glcanvas.GLCanvas(
                self.renderer_splitter,
                attribList = ( glcanvas.WX_GL_RGBA,
                               glcanvas.WX_GL_DOUBLEBUFFER,
                               glcanvas.WX_GL_MIN_ALPHA, 8, ) )
        
        self.renderer_splitter.SplitVertically(
            self.properties_splitter,
            self.renderer,
            self.LEFT_SASH_POSITION,
        )
        
        # self.layer_tree_panel = wx.Control( self.properties_splitter )
        self.layer_tree_control = Layer_tree_control.Layer_tree_control( self.properties_splitter )
        self.properties_panel = Properties_panel( self.properties_splitter )
        
        self.properties_splitter.SplitHorizontally(
            self.layer_tree_control,
            self.properties_panel,
            self.TOP_SASH_POSITION,
        )
        
        self.frame.Bind( wx.EVT_CLOSE, self.close )
        self.frame.Bind( wx.EVT_MENU, self.close, id = wx.ID_EXIT )
        self.frame.Bind( wx.EVT_ACTIVATE, self.refresh )
        self.frame.Bind( wx.EVT_ACTIVATE_APP, self.refresh )
        self.frame.Bind( wx.EVT_MOVE, self.refresh )
        self.frame.Bind( wx.EVT_IDLE, self.on_idle )
        # self.frame.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )
        
        self.renderer_splitter.Bind( wx.EVT_SPLITTER_SASH_POS_CHANGING, self.splitter_size_changed )
        self.renderer_splitter.Bind( wx.EVT_SPLITTER_SASH_POS_CHANGED, self.splitter_size_changed )
        # self.renderer.Bind( wx.EVT_PAINT, self.draw_render_pane )
        self.renderer.Bind( wx.EVT_SIZE, self.resize_render_pane )
        self.renderer.Bind( wx.EVT_LEFT_DOWN, self.on_mouse_down )
        self.renderer.Bind( wx.EVT_MOTION, self.on_mouse_motion )
        self.renderer.Bind( wx.EVT_LEFT_UP, self.on_mouse_up )
        self.renderer.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )
        self.renderer.Bind( wx.EVT_LEAVE_WINDOW, self.on_mouse_leave )
        self.renderer.Bind( wx.EVT_CHAR, self.on_key_char )
        self.renderer.Bind( wx.EVT_KEY_DOWN, self.on_key_down )
        self.renderer.Bind( wx.EVT_KEY_DOWN, self.on_key_up )
        # Prevent flashing on Windows by doing nothing on an erase background event.
        self.renderer.Bind( wx.EVT_ERASE_BACKGROUND, lambda event: None )
        self.renderer.SetFocus()
        
        self.SetTopWindow( self.frame )
        self.frame.SetSize( self.DEFAULT_FRAME_SIZE )
        self.frame.Show( True )
        
        """
        for filename in self.init_filenames:
            file_opener = ui.File_opener(
                self.frame, self.root_layer, self.command_stack, filename,
            )

            file_opener.outbox.subscribe(
                self.inbox,
                request = "file_opener_done",
            )

            scheduler.add( file_opener.run )

            # To ensure that layers show up in a deterministic order, wait
            # until the layer has been successfully added before adding another.
            self.inbox.receive( request = "file_opener_done" )
            file_opener.outbox.unsubscribe( self.inbox )
        """
        
        self.layer_tree_control.rebuild()
        
        print "( 0, 0 ) -> " + str( self.get_projected_point_from_world_point( ( 0, 0 ) ) )
        print "( 1, 1 ) -> " + str( self.get_projected_point_from_world_point( ( 1, 1 ) ) )
        print "( 179, 89 ) -> " + str( self.get_projected_point_from_world_point( ( 180, 89 ) ) )
        print "( 10, 10 ) -> " + str( self.get_projected_point_from_world_point( ( 10, 10 ) ) )
        
        print "( 0, 0 ) <- " + str( self.get_world_point_from_projected_point( ( 0, 0 ) ) )
        print "( 1000000, 1000000 ) <- " + str( self.get_world_point_from_projected_point( ( 1000000, 1000000 ) ) )
        print "( 1000000, -1000000 ) <- " + str( self.get_world_point_from_projected_point( ( 1000000, -1000000 ) ) )
        print "( -1000000, 1000000 ) <- " + str( self.get_world_point_from_projected_point( ( -1000000, 1000000 ) ) )
        print "( -1000000, -1000000 ) <- " + str( self.get_world_point_from_projected_point( ( -1000000, -1000000 ) ) )
        
        self.is_initialized = True
        
        return True
    
    def on_mouse_down( self, event ):
        self.renderer.SetFocus()
        
        self.get_effective_tool_mode( event ) # update alt key state
        self.forced_cursor = None
        self.mouse_is_down = True
        self.selection_box_is_being_defined = False
        self.mouse_down_position = event.GetPosition()
        self.mouse_move_position = self.mouse_down_position
        self.renderer.CaptureMouse()
        
        if ( self.get_effective_tool_mode( event ) == self.MODE_PAN ):
            return
        
        e = app_globals.editor
        lm = app_globals.layer_manager
        
        if ( e.clickable_object_mouse_is_over != None ):
            ( layer_index, type, subtype, object_index ) = e.parse_clickable_object( e.clickable_object_mouse_is_over )
            layer = lm.get_layer_by_flattened_index( layer_index )
            if ( lm.is_layer_selected( layer ) ):
                if ( e.clickable_object_is_ugrid_point() ):
                    e.clicked_on_point( event, layer, object_index )
                if ( e.clickable_object_is_ugrid_line() ):
                    world_point = self.get_world_point_from_screen_point( event.GetPosition() )
                    e.clicked_on_line_segment( event, layer, object_index, world_point )
        else:
            layer = self.layer_tree_control.get_selected_layer()
            if ( layer != None ):
                if ( event.ControlDown() or event.ShiftDown() ):
                    self.selection_box_is_being_defined = True
                else:
                    world_point = self.get_world_point_from_screen_point( event.GetPosition() )
                    e.clicked_on_empty_space( event, layer, world_point )
    
    def release_mouse( self ):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        self.renderer.ReleaseMouse()
    
    def on_mouse_motion( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        
        if ( not self.mouse_is_down ):
            o = self.opengl_renderer.picker.get_object_at_mouse_position( event.GetPosition() )
            if ( o != None ):
                ( layer_index, type, subtype, object_index ) = app_globals.editor.parse_clickable_object( o )
                layer = app_globals.layer_manager.get_layer_by_flattened_index( layer_index )
                if ( app_globals.layer_manager.is_layer_selected( layer ) ):
                    app_globals.editor.clickable_object_mouse_is_over = o
            else:
                app_globals.editor.clickable_object_mouse_is_over = None
        
        if ( self.mouse_is_down ):
            p = event.GetPosition()
            d_x = p[ 0 ] - self.mouse_down_position[ 0 ]
            d_y = self.mouse_down_position[ 1 ] - p[ 1 ]
            # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
            if ( self.get_effective_tool_mode( event ) == self.MODE_PAN ):
                if ( d_x != 0 or d_y != 0 ):
                    # the user has panned the map
                    d_x_p = d_x * self.projected_units_per_pixel
                    d_y_p = d_y * self.projected_units_per_pixel
                    self.projected_point_center = ( self.projected_point_center[ 0 ] - d_x_p,
                                                    self.projected_point_center[ 1 ] - d_y_p )
                    self.mouse_down_position = p
                    self.refresh()
            elif ( self.get_effective_tool_mode( event ) == self.MODE_ZOOM_RECT or self.selection_box_is_being_defined ):
                self.mouse_move_position = event.GetPosition()
                self.refresh()
            else:
                if ( d_x != 0 or d_y != 0 ):
                    w_p0 = self.get_world_point_from_screen_point( self.mouse_down_position )
                    w_p1 = self.get_world_point_from_screen_point( p )
                    app_globals.editor.dragged( w_p1[ 0 ] - w_p0[ 0 ], w_p1[ 1 ] - w_p0[ 1 ] )
                    self.mouse_down_position = p
                    self.refresh()
    
    def on_mouse_up( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        
        self.forced_cursor = None
        
        if ( not self.mouse_is_down ):
            self.selection_box_is_being_defined = False
            
            return
        
        self.mouse_is_down = False
        self.renderer.ReleaseMouse()
        if ( self.get_effective_tool_mode( event ) == self.MODE_ZOOM_RECT ):
            self.mouse_move_position = event.GetPosition()
            ( x1, y1, x2, y2 ) = rect.get_normalized_coordinates( self.mouse_down_position,
                                                                  self.mouse_move_position )
            d_x = x2 - x1
            d_y = y2 - y1
            if ( d_x >= 5 and d_y >= 5 ):
                p_r = self.get_projected_rect_from_screen_rect( ( ( x1, y1 ), ( x2, y2 ) ) )
                self.projected_point_center = rect.center( p_r )
                s_r = self.get_screen_rect()
                ratio_h = float( d_x ) / float( rect.width( s_r ) )
                ratio_v = float( d_y ) / float( rect.height( s_r ) )
                self.projected_units_per_pixel *= max( ratio_h, ratio_v )
                self.constrain_zoom()
                self.refresh()
        elif ( self.get_effective_tool_mode( event ) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode( event ) == self.MODE_EDIT_LINES ):
            if ( self.selection_box_is_being_defined ):
                self.mouse_move_position = event.GetPosition()
                ( x1, y1, x2, y2 ) = rect.get_normalized_coordinates( self.mouse_down_position,
                                                                      self.mouse_move_position )
                p_r = self.get_projected_rect_from_screen_rect( ( ( x1, y1 ), ( x2, y2 ) ) )
                w_r = self.get_world_rect_from_projected_rect( p_r )
                layer = self.layer_tree_control.get_selected_layer()
                if ( layer != None ):
                    if ( self.get_effective_tool_mode( event ) == self.MODE_EDIT_POINTS ):
                        layer.select_points_in_rect( event.ControlDown(), event.ShiftDown(), w_r )
                    else:
                        layer.select_line_segments_in_rect( event.ControlDown(), event.ShiftDown(), w_r )
                self.selection_box_is_being_defined = False
                self.refresh()
            else:
                app_globals.editor.finished_drag( self.mouse_down_position, self.mouse_move_position )
        self.selection_box_is_being_defined = False
    
    def on_mouse_wheel_scroll( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        
        rotation = event.GetWheelRotation()
        delta = event.GetWheelDelta()
        if ( delta == 0 ):
            return
        
        amount = rotation / delta
        
        screen_point_clicked = event.GetPosition()
        projected_point_clicked = self.get_projected_point_from_screen_point( screen_point_clicked )
        d_projected_units = ( projected_point_clicked[ 0 ] - self.projected_point_center[ 0 ],
                              projected_point_clicked[ 1 ] - self.projected_point_center[ 1 ] )
        
        if ( amount < 0 ):
            self.projected_units_per_pixel *= 2
        else:
            self.projected_units_per_pixel /= 2
        self.constrain_zoom()
        
        # compensate to keep the same projected point under the mouse
        if ( amount < 0 ):
            self.projected_point_center = ( self.projected_point_center[ 0 ] - d_projected_units[ 0 ],
                                            self.projected_point_center[ 1 ] - d_projected_units[ 1 ] )
        else:
            self.projected_point_center = ( self.projected_point_center[ 0 ] + d_projected_units[ 0 ] / 2,
                                            self.projected_point_center[ 1 ] + d_projected_units[ 1 ] / 2 )
        
        self.refresh()
    
    def on_mouse_leave( self, event ):
        self.frame.SetCursor( wx.StockCursor( wx.CURSOR_ARROW ) )
        # this messes up object dragging when the mouse goes outside the window
        # app_globals.editor.clickable_object_mouse_is_over = None
    
    def on_key_down( self, event ):
        self.get_effective_tool_mode( event )
    
    def on_key_up( self, event ):
        self.get_effective_tool_mode( event )
    
    def on_key_char( self, event ):
        self.get_effective_tool_mode( event )
        self.set_cursor()
        
        if ( self.mouse_is_down and self.get_effective_tool_mode( event ) == self.MODE_ZOOM_RECT ):
            if ( event.GetKeyCode() == wx.WXK_ESCAPE ):
                self.mouse_is_down = False
                self.renderer.ReleaseMouse()
                self.refresh()
        else:
            if ( event.GetKeyCode() == wx.WXK_ESCAPE ):
                app_globals.editor.esc_key_pressed()
    
    def on_idle( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        # print self.mouse_is_down
        self.set_cursor()
    
    def set_cursor( self ):
        if ( self.forced_cursor != None ):
            self.frame.SetCursor( self.forced_cursor )
            #
            return
        
        if ( app_globals.editor.clickable_object_mouse_is_over != None and
             ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode( None ) == self.MODE_EDIT_LINES ) ):
            if ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS and app_globals.editor.clickable_object_is_ugrid_line() ):
                self.frame.SetCursor( wx.StockCursor( wx.CURSOR_BULLSEYE ) )
            else:
                self.frame.SetCursor( wx.StockCursor( wx.CURSOR_HAND ) )
            #
            return
        
        if ( self.mouse_is_down ):
            if ( self.get_effective_tool_mode( None ) == self.MODE_PAN ):
                self.frame.SetCursor( self.hand_closed_cursor )
            #
            return
        
        w = wx.FindWindowAtPointer()
        if ( w == self.renderer ):
            c = wx.StockCursor( wx.CURSOR_ARROW )
            if ( self.get_effective_tool_mode( None ) == self.MODE_PAN ):
                c = self.hand_cursor
            if ( self.get_effective_tool_mode( None ) == self.MODE_ZOOM_RECT ):
                c = wx.StockCursor( wx.CURSOR_CROSS )
            if ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode( None ) == self.MODE_EDIT_LINES ):
                c = wx.StockCursor( wx.CURSOR_PENCIL )
            self.frame.SetCursor( c )
    
    def get_effective_tool_mode( self, event ):
        if ( event != None ):
            try:
                self.is_alt_key_down = event.AltDown()
                # print self.is_alt_key_down
            except:
                pass
        if ( self.is_alt_key_down ):
            return self.MODE_PAN
        #
        return self.mode
    
    def layer_tree_selection_changed( self ):
        self.menu_bar.enable_disable_menu_items()
        self.tool_bar.enable_disable_tools()
        self.refresh()
    
    def show_triangle_dialog_box( self ):
        if ( self.triangle_dialog_box == None ):
            self.triangle_dialog_box = Triangle_dialog()
        self.triangle_dialog_box.Show()
        self.triangle_dialog_box.SetFocus()
    
    def show_merge_layers_dialog_box( self ):
        Merge_layers_dialog().show()
    
    def show_merge_duplicate_points_dialog_box( self ):
        if ( self.merge_duplicate_points_dialog_box == None ):
            self.merge_duplicate_points_dialog_box = Merge_duplicate_points_dialog()
        self.merge_duplicate_points_dialog_box.Show()
        self.merge_duplicate_points_dialog_box.SetFocus()
    
    def points_were_deleted( self, layer ):
        # when points are deleted from a layer the indexes of the points in the existing merge dialog box
        # become invalid; so force the user to re-find duplicates in order to create a valid list again
        if ( self.merge_duplicate_points_dialog_box != None and self.merge_duplicate_points_dialog_box.layer == layer ):
            self.merge_duplicate_points_dialog_box.clear_results()
    
    def MacReopenApp( self ):
        """
        Invoked by wx when the Maproom dock icon is clicked on Mac OS X.
        """
        self.GetTopWindow().Raise()
    
    def close( self, event ):
        # self.renderer.shutdown()
        # self.shutdown()
        self.is_closing = True
        app_globals.layer_manager.destroy()
        self.opengl_renderer.destroy()
        self.frame.Destroy()
    
    """
    # see http://wiki.wxwidgets.org/WxGLCanvas#Tooltips_from_upper_widgets_disappearing_or_cut_when_drawn_over_wxGLCanvas_on_Windows
    # http://trac.wxwidgets.org/ticket/10520 says it's fixed, but apparently not in our copy of wxpython
    def draw_render_pane( self, event ):
        # self.point_adder.set_hovered( *self.picker.hovered )
        dc = wx.PaintDC( self.renderer )
        # well, it was difficult to get the window to paint in all cases, so for now
        # we put in this refresh() call to go ahead and paint in all cases, which again messes up the tooltips (on Windows)
        self.refresh( event )
    """
    
    def refresh( self, event = None, rebuild_layer_tree_control = False ):
        if ( self.is_closing ):
            return
        
        if ( rebuild_layer_tree_control and self.layer_tree_control != None ):
            self.layer_tree_control.rebuild()
        self.render( event )
        if ( self.layer_tree_control != None and self.properties_panel != None ):
            layer = self.layer_tree_control.get_selected_layer()
            # note that the following call only does work if the properties for the layer have changed
            self.properties_panel.display_panel_for_layer( layer )
        if ( self.is_initialized ):
            self.menu_bar.enable_disable_menu_items()
            self.tool_bar.enable_disable_tools()
    
    def render( self, event ):
        if ( self.is_closing ):
            return
        if ( self.renderer == None ):
            return
        
        """
        import traceback
        traceback.print_stack();
        import code; code.interact( local = locals() )
        """
        
        self.renderer.SetCurrent()
        
        # this has to be here because the window has to exist before making the renderer
        if ( self.opengl_renderer == None ):
            self.opengl_renderer = library.Opengl_renderer.Opengl_renderer( True )
        
        s_r = self.get_screen_rect()
        # print "s_r = " + str( s_r )
        p_r = self.get_projected_rect_from_screen_rect( s_r )
        # print "p_r = " + str( p_r )
        w_r = self.get_world_rect_from_projected_rect( p_r )
        # print "w_r = " + str( w_r )
        
        if ( not self.opengl_renderer.prepare_to_render_projected_objects( p_r, s_r ) ):
            return
        
        """
        self.root_renderer.render()
        self.set_screen_projection_matrix()
        self.box_overlay.render()
        self.set_render_projection_matrix()
        """
        
        app_globals.layer_manager.render()
        
        if ( not self.opengl_renderer.prepare_to_render_screen_objects( s_r ) ):
            return
        
        # we use a try here since we must call done_rendering_screen_objects() below
        # to pop the gl stack
        try:
            if ( self.lon_lat_grid_shown ):
                self.lon_lat_grid.draw( self.opengl_renderer, w_r, p_r, s_r )
            if ( ( self.get_effective_tool_mode( event ) == self.MODE_ZOOM_RECT or self.selection_box_is_being_defined ) and self.mouse_is_down ):
                ( x1, y1, x2, y2 ) = rect.get_normalized_coordinates( self.mouse_down_position,
                                                                      self.mouse_move_position )
                # self.opengl_renderer.draw_screen_rect( ( ( 20, 50 ), ( 300, 200 ) ), 1.0, 1.0, 0.0, alpha = 0.25 )
                rects = self.get_surrounding_screen_rects( ( ( x1, y1 ), ( x2, y2 ) ) )
                for r in rects:
                    if ( r != rect.EMPTY_RECT ):
                        self.opengl_renderer.draw_screen_rect( r, 0.0, 0.0, 0.0, 0.25 )
                # small adjustments to make stipple overlap gray rects perfectly
                y1 -= 1
                x2 += 1
                self.opengl_renderer.draw_screen_line( ( x1, y1 ), ( x2, y1 ), 1.0, 0, 0, 0, 1.0, 1, 0x00FF )
                self.opengl_renderer.draw_screen_line( ( x1, y1 ), ( x1, y2 ), 1.0, 0, 0, 0, 1.0, 1, 0x00FF )
                self.opengl_renderer.draw_screen_line( ( x2, y1 ), ( x2, y2 ), 1.0, 0, 0, 0, 1.0, 1, 0x00FF )
                self.opengl_renderer.draw_screen_line( ( x1, y2 ), ( x2, y2 ), 1.0, 0, 0, 0, 1.0, 1, 0x00FF )
        except Exception as inst:
            print "error during rendering of screen objects: " + str( inst )
        
        self.opengl_renderer.done_rendering_screen_objects()
        
        self.renderer.SwapBuffers()
        
        self.opengl_renderer.prepare_to_render_picker( s_r )
        app_globals.layer_manager.render( pick_mode = True )
        self.opengl_renderer.done_rendering_picker()
        
        if ( event != None ):
            event.Skip()
    
    def splitter_size_changed( self, event ):
        if not self.renderer.GetContext():
            return
        event.Skip()
        # self.renderer.Refresh()
        # self.renderer.Update()
        # self.renderer_splitter.Refresh()
        # self.renderer_splitter.Update()
        self.resize_render_pane( event )
    
    def resize_render_pane( self, event ):
        if not self.renderer.GetContext():
            return
        
        event.Skip()
        
        """
        # Make sure the frame is shown before calling SetCurrent().
        self.renderer.Show()
        self.renderer.SetCurrent()
        """
        self.refresh( event )
    
    ####### functions related to world coordinates, projected coordinates, and screen coordinates
    
    def get_screen_size( self ):
        return self.renderer.GetClientSize()
    
    def get_screen_rect( self ):
        size = self.get_screen_size()
        #
        return ( ( 0, 0 ), ( size[ 0 ], size[ 1 ] ) )
    
    def get_projected_point_from_screen_point( self, screen_point ):
        c = rect.center( self.get_screen_rect() )
        d = ( screen_point[ 0 ] - c[ 0 ], screen_point[ 1 ] - c[ 1 ] )
        d_p = ( d[ 0 ] * self.projected_units_per_pixel, d[ 1 ] * self.projected_units_per_pixel )
        #
        return ( self.projected_point_center[ 0 ] + d_p[ 0 ],
                 self.projected_point_center[ 1 ] - d_p[ 1 ] )
    
    def get_projected_rect_from_screen_rect( self, screen_rect ):
        left_bottom = ( screen_rect[ 0 ][ 0 ], screen_rect[ 1 ][ 1 ] )
        right_top = ( screen_rect[ 1 ][ 0 ], screen_rect[ 0 ][ 1 ] )
        #
        return ( self.get_projected_point_from_screen_point( left_bottom ),
                 self.get_projected_point_from_screen_point( right_top ) )
    
    def get_screen_point_from_projected_point( self, projected_point ):
        d_p = ( projected_point[ 0 ] - self.projected_point_center[ 0 ],
                projected_point[ 1 ] - self.projected_point_center[ 1 ] )
        d = ( d_p[ 0 ] / self.projected_units_per_pixel, d_p[ 1 ] / self.projected_units_per_pixel )
        r = self.get_screen_rect()
        c = rect.center( r )
        #
        return ( c[ 0 ] + d[ 0 ], c[ 1 ] - d[ 1 ] )
    
    def get_screen_rect_from_projected_rect( self, projected_rect ):
        left_top = ( projected_rect[ 0 ][ 0 ], projected_rect[ 1 ][ 1 ] )
        right_bottom = ( projected_rect[ 1 ][ 0 ], projected_rect[ 0 ][ 1 ] )
        #
        return ( self.get_screen_point_from_projected_point( left_top ),
                 self.get_screen_point_from_projected_point( right_bottom ) )
    
    def get_world_point_from_projected_point( self, projected_point ):
        if ( self.projection_is_identity ):
            return projected_point
        else:
            return self.projection( projected_point[ 0 ], projected_point[ 1 ], inverse = True )
    
    def get_world_rect_from_projected_rect( self, projected_rect ):
        return ( self.get_world_point_from_projected_point( projected_rect[ 0 ] ),
                 self.get_world_point_from_projected_point( projected_rect[ 1 ] ) )
    
    def get_projected_point_from_world_point( self, world_point ):
        if ( self.projection_is_identity ):
            return world_point
        else:
            return self.projection( world_point[ 0 ], world_point[ 1 ] )
    
    def get_projected_rect_from_world_rect( self, world_rect ):
        return ( self.get_projected_point_from_world_point( world_rect[ 0 ] ),
                 self.get_projected_point_from_world_point( world_rect[ 1 ] ) )
    
    def get_world_point_from_screen_point( self, screen_point ):
        return self.get_world_point_from_projected_point( self.get_projected_point_from_screen_point( screen_point ) )
    
    def get_world_rect_from_screen_rect( self, screen_rect ):
        return self.get_world_rect_from_projected_rect( self.get_projected_rect_from_screen_rect( screen_rect ) )
    
    def get_screen_point_from_world_point( self, world_point ):
        return self.get_screen_point_from_projected_point( self.get_projected_point_from_world_point( world_point ) )
    
    def get_screen_rect_from_world_rect( self, world_rect ):
        return self.get_screen_rect_from_projected_rect( self.get_projected_rect_from_world_rect( world_rect ) )
    
    def zoom_in( self ):
        self.projected_units_per_pixel /= 2.0;
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str( self.projected_units_per_pixel )
        self.refresh()
    
    def zoom_out( self ):
        self.projected_units_per_pixel *= 2.0;
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str( self.projected_units_per_pixel )
        self.refresh()
    
    def zoom_to_fit( self ):
        w_r = app_globals.layer_manager.accumulate_layer_rects()
        if ( w_r != rect.NONE_RECT ):
            self.zoom_to_world_rect( w_r )
    
    def zoom_to_world_rect( self, w_r ):
        p_r = self.get_projected_rect_from_world_rect( w_r )
        size = self.get_screen_size()
        pixels_h = rect.width( p_r ) / self.projected_units_per_pixel
        pixels_v = rect.height( p_r ) / self.projected_units_per_pixel
        # print "pixels_h = {0}, pixels_v = {1}".format( pixels_h, pixels_v )
        ratio_h = float( pixels_h ) / float( size[ 0 ] )
        ratio_v = float( pixels_v ) / float( size[ 1 ] )
        ratio = max( ratio_h, ratio_v )
        
        self.projected_point_center = rect.center( p_r )
        self.projected_units_per_pixel *= ratio
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str( self.projected_units_per_pixel )
        
        self.refresh()
    
    def zoom_to_include_world_rect( self, w_r ):
        view_w_r = self.get_world_rect_from_screen_rect( self.get_screen_rect() )
        if ( not rect.contains_rect( view_w_r, w_r ) ):
            # first try just panning
            p_r = self.get_projected_rect_from_world_rect( w_r )
            self.projected_point_center = rect.center( p_r )
            view_w_r = self.get_world_rect_from_screen_rect( self.get_screen_rect() )
            if ( not rect.contains_rect( view_w_r, w_r ) ):
                # otherwise we have to zoom (i.e., zoom out because panning didn't work)
                self.zoom_to_world_rect( w_r )
    
    def reproject_all( self, srs ):
        s_r = self.get_screen_rect()
        s_c = rect.center( s_r )
        w_c = self.get_world_point_from_screen_point( s_c )
        was_identity = self.projection_is_identity
        
        # print "self.projected_units_per_pixel A = " + str( self.projected_units_per_pixel )
        self.projection = pyproj.Proj( srs )
        self.projection_is_identity = self.projection.srs.find( "+proj=longlat" ) != -1
        app_globals.layer_manager.reproject_all( self.projection, self.projection_is_identity )
        # print "self.projected_units_per_pixel B = " + str( self.projected_units_per_pixel )
        
        ratio = 1.0
        if ( was_identity and not self.projection_is_identity ):
            ratio = 40075016.6855801 / 360.0
        if ( not was_identity and self.projection_is_identity ):
            ratio = 360.0 / 40075016.6855801
        self.projected_units_per_pixel *= ratio
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str( self.projected_units_per_pixel )
        # import code; code.interact( local = locals() )
        
        self.projected_point_center = self.get_projected_point_from_world_point( w_c )
        
        self.refresh()
    
    def constrain_zoom( self ):
        if ( self.projection_is_identity ):
            min_val = 0.00001
            max_val = 1
        else:
            min_val = 1
            max_val = 80000
        self.projected_units_per_pixel = max( self.projected_units_per_pixel, min_val )
        self.projected_units_per_pixel = min( self.projected_units_per_pixel, max_val )
    
    def get_surrounding_screen_rects( self, r ):
        # return four disjoint rects surround r on the screen
        sr = self.get_screen_rect()
        
        if ( r[ 0 ][ 1 ] <= sr[ 0 ][ 1 ] ):
            above = rect.EMPTY_RECT
        else:
            above = ( sr[ 0 ], ( sr[ 1 ][ 0 ], r[ 0 ][ 1 ] ) )
        
        if ( r[ 1 ][ 1 ] >= sr[ 1 ][ 1 ] ):
            below = rect.EMPTY_RECT
        else:
            below = ( ( sr[ 0 ][ 0 ], r[ 1 ][ 1 ] ), sr[ 1 ] )
        
        if ( r[ 0 ][ 0 ] <= sr[ 0 ][ 0 ] ):
            left = rect.EMPTY_RECT
        else:
            left = ( ( sr[ 0 ][ 0 ], r[ 0 ][ 1 ] ), ( r[ 0 ][ 0 ], r[ 1 ][ 1 ] ) )
        
        if ( r[ 1 ][ 0 ] >= sr[ 1 ][ 0 ] ):
            right = rect.EMPTY_RECT
        else:
            right = ( ( r[ 1 ][ 0 ], r[ 0 ][ 1 ] ), ( sr[ 1 ][ 0 ], r[ 1 ][ 1 ] ) )
        
        return [ above, below, left, right ]
    
    """
    def get_degrees_lon_per_pixel( self, reference_latitude = None ):
        if ( reference_latitude == None ):
            reference_latitude = self.world_point_center[ 1 ]
        factor = math.cos( math.radians( reference_latitude ) )
        ###
        return self.degrees_lat_per_pixel * factor
    
    def get_lon_dist_from_screen_dist( self, screen_dist ):
        return self.get_degrees_lon_per_pixel() * screen_dist
    
    def get_lat_dist_from_screen_dist( self, screen_dist ):
        return self.degrees_lat_per_pixel * screen_dist
    """
