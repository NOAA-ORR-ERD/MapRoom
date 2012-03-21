import os
#import os.path
#import sys
#import math
import wx
import wx.glcanvas as glcanvas
import pyproj
#from ui.Menu_bar import Menu_bar
#from ui.Tool_bar import Tool_bar
#from ui.Properties_panel import Properties_panel
#from ui.Triangle_dialog import Triangle_dialog
#from ui.Merge_layers_dialog import Merge_layers_dialog
#from ui.Merge_duplicate_points_dialog import Merge_duplicate_points_dialog
#import Layer_manager
#import Layer_tree_control
#import Editor
import lon_lat_grid
import library.Opengl_renderer
import library.rect as rect
import app_globals

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

class RenderWindow( glcanvas.GLCanvas ):
    """
    The core rendering class for MapRoom app.
    """

    ##fixme: this should be in app_globals, or something like that
    IMAGE_PATH = "ui/images"

    MODE_PAN = 0
    MODE_ZOOM_RECT = 1
    MODE_EDIT_POINTS = 2
    MODE_EDIT_LINES = 3

    mode = MODE_PAN
    hand_cursor = None
    hand_closed_cursor = None
    forced_cursor = None

    lon_lat_grid = None
    lon_lat_grid_shown = True
    
    opengl_renderer = None
    
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
    projection = pyproj.Proj( "+proj=merc +units=m +over" )
    # for longlat projection, apparently someone decided that since the projection
    # is the identity, it might as well do something and so it returns the coordinates as
    # radians instead of degrees; so here we use this variable to avoid using the longlat projection
    projection_is_identity = False
    
    #is_initialized = False
    #is_closing = False
    
    def __init__(self, *args, **kwargs):
        kwargs[ 'attribList' ]= ( glcanvas.WX_GL_RGBA,
                                glcanvas.WX_GL_DOUBLEBUFFER,
                                glcanvas.WX_GL_MIN_ALPHA, 8, )
        glcanvas.GLCanvas.__init__( self, *args, **kwargs)
        
        self.context = glcanvas.GLContext(self)
        
        p = os.path.join( self.IMAGE_PATH, "cursors", "hand.ico" )
        self.hand_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
        p = os.path.join( self.IMAGE_PATH, "cursors", "hand_closed.ico" )
        self.hand_closed_cursor = wx.Cursor( p, wx.BITMAP_TYPE_ICO, 16, 16 )
        
        self.lon_lat_grid = lon_lat_grid.Lon_lat_grid()
        
        #self.frame.Bind( wx.EVT_MOVE, self.refresh )
        #self.frame.Bind( wx.EVT_IDLE, self.on_idle )
        # self.frame.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )
        
        self.Bind( wx.EVT_IDLE, self.on_idle ) # not sure about this -- but it's where the cursors are set.
        self.Bind( wx.EVT_PAINT, self.render )
        self.Bind( wx.EVT_SIZE, self.resize_render_pane )
        self.Bind( wx.EVT_LEFT_DOWN, self.on_mouse_down )
        self.Bind( wx.EVT_MOTION, self.on_mouse_motion )
        self.Bind( wx.EVT_LEFT_UP, self.on_mouse_up )
        self.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )
        self.Bind( wx.EVT_LEAVE_WINDOW, self.on_mouse_leave )
        self.Bind( wx.EVT_CHAR, self.on_key_char )
        self.Bind( wx.EVT_KEY_DOWN, self.on_key_down )
        self.Bind( wx.EVT_KEY_DOWN, self.on_key_up )
        # Prevent flashing on Windows by doing nothing on an erase background event.
        self.Bind( wx.EVT_ERASE_BACKGROUND, lambda event: None )
        #self.is_initialized = True
        
    
    def on_mouse_down( self, event ):
        #self.SetFocus() # why would it not be focused?
        print "in on_mouse_down"
        self.get_effective_tool_mode( event ) # update alt key state
        self.forced_cursor = None
        self.mouse_is_down = True
        self.selection_box_is_being_defined = False
        self.mouse_down_position = event.GetPosition()
        self.mouse_move_position = self.mouse_down_position
        #self.CaptureMouse()
        
        if ( self.get_effective_tool_mode( event ) == self.MODE_PAN ):
            self.CaptureMouse()
            return
        
        e = app_globals.editor
        lm = app_globals.layer_manager
        
        if ( e.clickable_object_mouse_is_over != None ): # the mouse is on a clickable object
            ( layer_index, type, subtype, object_index ) = e.parse_clickable_object( e.clickable_object_mouse_is_over )
            layer = lm.get_layer_by_flattened_index( layer_index )
            if ( lm.is_layer_selected( layer ) ):
                self.CaptureMouse()
                if ( e.clickable_object_is_ugrid_point() ):
                    e.clicked_on_point( event, layer, object_index )
                if ( e.clickable_object_is_ugrid_line() ):
                    world_point = self.get_world_point_from_screen_point( event.GetPosition() )
                    e.clicked_on_line_segment( event, layer, object_index, world_point )
        else: # the mouse is not on a clickable object
            ##fixme: there should be a reference to the layer manager in the RenderWindow
            ##       and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = app_globals.application.layer_tree_control.get_selected_layer()
            if ( layer != None ):
                if ( event.ControlDown() or event.ShiftDown() ):
                    self.selection_box_is_being_defined = True
                    self.CaptureMouse()
                else:
                    world_point = self.get_world_point_from_screen_point( event.GetPosition() )
                    e.clicked_on_empty_space( event, layer, world_point )
    
    def release_mouse( self ):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        self.ReleaseMouse()
    
    def on_mouse_motion( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        
        if ( not self.mouse_is_down ):
            #print "mouse is not down"
            o = self.opengl_renderer.picker.get_object_at_mouse_position( event.GetPosition() )
            #print "object that is under mouse:", o
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
                    self.render(event)
            elif ( self.get_effective_tool_mode( event ) == self.MODE_ZOOM_RECT or self.selection_box_is_being_defined ):
                self.mouse_move_position = event.GetPosition()
                self.render(event)
            else:
                if ( d_x != 0 or d_y != 0 ):
                    w_p0 = self.get_world_point_from_screen_point( self.mouse_down_position )
                    w_p1 = self.get_world_point_from_screen_point( p )
                    app_globals.editor.dragged( w_p1[ 0 ] - w_p0[ 0 ], w_p1[ 1 ] - w_p0[ 1 ] )
                    self.mouse_down_position = p
                    self.render(event)
    
    def on_mouse_up( self, event ):
        self.get_effective_tool_mode( event ) # update alt key state
        
        self.forced_cursor = None
        
        if ( not self.mouse_is_down ):
            self.selection_box_is_being_defined = False
            
            return
        
        self.mouse_is_down = False
        if self.HasCapture(): # it's hard to know for sure when the mouse may be captured
            self.ReleaseMouse()
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
                self.render()
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
                self.render()
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
        
        self.render()
    
    def on_mouse_leave( self, event ):
        self.SetCursor( wx.StockCursor( wx.CURSOR_ARROW ) )
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
                self.ReleaseMouse()
                self.render()
        else:
            if ( event.GetKeyCode() == wx.WXK_ESCAPE ):
                app_globals.editor.esc_key_pressed()
    
    def on_idle( self, event ):
        #self.get_effective_tool_mode( event ) # update alt key state (not needed, it gets called in set_cursor anyway
        # print self.mouse_is_down
        self.set_cursor()
    
    def set_cursor( self ):
        if ( self.forced_cursor != None ):
            self.SetCursor( self.forced_cursor )
            #
            return
        
        if ( app_globals.editor.clickable_object_mouse_is_over != None and
             ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode( None ) == self.MODE_EDIT_LINES ) ):
            if ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS and app_globals.editor.clickable_object_is_ugrid_line() ):
                self.SetCursor( wx.StockCursor( wx.CURSOR_BULLSEYE ) )
            else:
                self.SetCursor( wx.StockCursor( wx.CURSOR_HAND ) )
            #
            return
        
        if ( self.mouse_is_down ):
            if ( self.get_effective_tool_mode( None ) == self.MODE_PAN ):
                self.SetCursor( self.hand_closed_cursor )
            #
            return
        
        #w = wx.FindWindowAtPointer() is this needed?
        #if ( w == self.renderer ):
        c = wx.StockCursor( wx.CURSOR_ARROW )
        if ( self.get_effective_tool_mode( None ) == self.MODE_PAN ):
            c = self.hand_cursor
        if ( self.get_effective_tool_mode( None ) == self.MODE_ZOOM_RECT ):
            c = wx.StockCursor( wx.CURSOR_CROSS )
        if ( self.get_effective_tool_mode( None ) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode( None ) == self.MODE_EDIT_LINES ):
            c = wx.StockCursor( wx.CURSOR_PENCIL )
        self.SetCursor( c )
    
    def get_effective_tool_mode( self, event ):
        if ( event != None ):
            try:
                self.is_alt_key_down = event.AltDown()
                # print self.is_alt_key_down
            except:
                pass
        if ( self.is_alt_key_down ):
            return self.MODE_PAN
        return self.mode
    
    def layer_tree_selection_changed( self ):
        self.render()

    def render( self, event=None ):
        
        """
        import traceback
        traceback.print_stack();
        import code; code.interact( local = locals() )
        """

        self.SetCurrent(self.context)
        
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
        
        app_globals.layer_manager.render(self)
        
        if ( not self.opengl_renderer.prepare_to_render_screen_objects( s_r ) ):
            return
        
        # we use a try here since we must call done_rendering_screen_objects() below
        # to pop the gl stack
        try:
            if ( self.lon_lat_grid_shown ):
                self.lon_lat_grid.draw( self, w_r, p_r, s_r )
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
            raise
            #print "error during rendering of screen objects: " + str( inst )
        
        self.opengl_renderer.done_rendering_screen_objects()
        
        self.SwapBuffers()
        
        self.opengl_renderer.prepare_to_render_picker( s_r )
        app_globals.layer_manager.render(self,  pick_mode = True )
        self.opengl_renderer.done_rendering_picker()
        
        if ( event != None ):
            event.Skip()
    
    def resize_render_pane( self, event ):
        if not self.GetContext():
            return
        
        event.Skip()
        self.render( event )
    
    ####### functions related to world coordinates, projected coordinates, and screen coordinates
    
    def get_screen_size( self ):
        return self.GetClientSize()
    
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
        self.render()
    
    def zoom_out( self ):
        self.projected_units_per_pixel *= 2.0;
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str( self.projected_units_per_pixel )
        self.render()
    
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
        
        self.render()
    
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
        
        self.render()
    
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
