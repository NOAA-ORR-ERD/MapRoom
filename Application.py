import os
import os.path
import sys
import math
import wx
#import wx.glcanvas as glcanvas
#import pyproj
from ui.Menu_bar import Menu_bar
from ui.Tool_bar import Tool_bar
from ui.Properties_panel import Properties_panel
from ui.Triangle_dialog import Triangle_dialog
from ui.Merge_layers_dialog import Merge_layers_dialog
from ui.Merge_duplicate_points_dialog import Merge_duplicate_points_dialog
import ui.File_opener
import Layer_manager
import Layer_tree_control
import Editor
#import lon_lat_grid
#import library.Opengl_renderer
#import library.rect as rect
import app_globals
import preferences

from ui.RenderWindow import RenderWindow

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
    
    NAME = "Maproom"
    
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
    
    #mouse_is_down = False
    is_alt_key_down = False
    selection_box_is_being_defined = False;
    #mouse_down_position = ( 0, 0 )
    #mouse_move_position = ( 0, 0 )
    
    # two variables keep track of what's visible on the screen:
    # (1) the projected point at the center of the screen
    #projected_point_center = ( 0, 0 )
    # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)
    #projected_units_per_pixel = 10000
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
    #projection = pyproj.Proj( "+proj=merc +units=m +over" )
    # for longlat projection, apparently someone decided that since the projection
    # is the identity, it might as well do something and so it returns the coordinates as
    # radians instead of degrees; so here we use this variable to avoid using the longlat projection
    #projection_is_identity = False
    
    is_initialized = False
    is_closing = False
    
    def __init__( self, init_filenames ):
        print "in application.__init__"
        app_globals.application = self
        app_globals.layer_manager = Layer_manager.Layer_manager()
        app_globals.editor = Editor.Editor()
        self.init_filenames = init_filenames
        wx.App.__init__( self, False )
        self.refresh()
    
    def OnInit( self ):
        print "in application.OnInit"
        self.SetAppName(self.NAME)
        
        data_dir = wx.StandardPaths.Get().GetUserDataDir()
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        app_globals.preferences = preferences.MaproomPreferences(os.path.join(data_dir, "MaproomPrefs.json"))
                
        self.frame = wx.Frame( None, wx.ID_ANY, self.NAME )
        self.frame.SetIcon( wx.Icon( self.ICON_FILENAME, wx.BITMAP_TYPE_ICO ) )
        self.frame.SetSizeHints( 250, 250 )
        # self.log_viewer = Log_viewer( self.frame )
        
#        p = os.path.join( self.IMAGE_PATH, "cursors", "hand.ico" )
#        self.hand_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
#        p = os.path.join( self.IMAGE_PATH, "cursors", "hand_closed.ico" )
#        self.hand_closed_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
#        
#        self.lon_lat_grid = lon_lat_grid.Lon_lat_grid()
        
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
        # On Mac, we need to call Realize after setting the frame's toolbar.
        self.tool_bar.Realize()
        
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
        
        self.renderer = RenderWindow( self.renderer_splitter)

        self.renderer_splitter.SplitVertically(
            self.properties_splitter,
            self.renderer,
            self.LEFT_SASH_POSITION,
        )
        
        
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

        self.renderer.SetFocus()
        
        self.SetTopWindow( self.frame )
        self.frame.SetSize( self.DEFAULT_FRAME_SIZE )
        self.frame.Show( True )
        
        if self.init_filenames:
            for filename in self.init_filenames:
                print "opening:", filename
                ui.File_opener.open_file(filename)
        else: # just so there is something there.
            pass #app_globals.layer_manager.add_folder( name = "folder_a" )
        
        print "selected layer", self.layer_tree_control.get_selected_layer()
        
        #self.layer_tree_control.rebuild()
        print "after rebuilding: selected layer", self.layer_tree_control.get_selected_layer()
        
        self.renderer.SetFocus()
        self.is_initialized = True
        
        return True
    
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
        app_globals.preferences.save()
        #self.opengl_renderer.destroy()
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
        print "refresh called"
        ## fixme: this shouldn't be required!
        if ( self.is_closing ):
            return
        
        if ( rebuild_layer_tree_control and self.layer_tree_control != None ):
            self.layer_tree_control.rebuild()
        if self.renderer is not None:
        #    self.renderer.render()
            # On Mac this is neither necessary nor desired.
            if not sys.platform.startswith('darwin'):
                self.renderer.Update()
            self.renderer.Refresh()
        if ( self.layer_tree_control != None and self.properties_panel != None ):
            layer = self.layer_tree_control.get_selected_layer()
            # note that the following call only does work if the properties for the layer have changed
            self.properties_panel.display_panel_for_layer( layer )
        if ( self.is_initialized ):
            self.menu_bar.enable_disable_menu_items()
            self.tool_bar.enable_disable_tools()
    
    
#    def splitter_size_changed( self, event ):
#        if not self.renderer.GetContext():
#            return
#        event.Skip()
#        # self.renderer.Refresh()
#        # self.renderer.Update()
#        # self.renderer_splitter.Refresh()
#        # self.renderer_splitter.Update()
#        self.resize_render_pane( event )
#    
#    def resize_render_pane( self, event ):
#        if not self.renderer.GetContext():
#            return
#        
#        event.Skip()
#        
#        """
#        # Make sure the frame is shown before calling SetCurrent().
#        self.renderer.Show()
#        self.renderer.SetCurrent()
#        """
#        self.refresh( event )
#    
