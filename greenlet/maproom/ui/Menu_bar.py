import wx
import os
import sys
import maproomlib.ui as ui
from About_dialog import About_dialog
from Help_window import Help_window
from Add_layer_menu import Add_layer_menu


class Menu_bar( wx.MenuBar ):
    """
    Main application menu bar.
    """
    IMAGE_PATH = "ui/images"

    def __init__( self, frame, renderer, root_layer, layer_tree,
                  command_stack, viewport, log_viewer, version ):
        wx.MenuBar.__init__( self )
        self.frame = frame
        self.renderer = renderer
        self.root_layer = root_layer
        self.layer_tree = layer_tree
        self.command_stack = command_stack
        self.viewport = viewport
        self.log_viewer = log_viewer
        self.version = version
        self.inbox = ui.Wx_inbox()
        self.duplicate_finder = None
        self.contour_maker = None
        self.triangle_maker = None

        image_path = self.IMAGE_PATH
        if os.path.basename( os.getcwd() ) != "maproom":
            image_path = os.path.join( "maproom", image_path )

        self.SetAutoWindowMenu( False )

        self.frame.Bind( wx.EVT_MENU,
            lambda event: wx.MessageDialog(
                frame, "Not implemented yet! Suggested solution: Wait a few weeks, install a new version, and try again.", style = wx.OK | wx.ICON_ERROR
            ).ShowModal()
        )

        self.file_menu = wx.Menu()
        self.Append( self.file_menu, "File" )

        self.open_item = wx.MenuItem(
            self.file_menu,
            wx.ID_OPEN,
            "Open...\tCtrl-O",
        )

        self.open_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "open.png" ) ),
        )
        self.file_menu.AppendItem( self.open_item )

        self.file_menu.AppendSeparator()

        self.save_item = wx.MenuItem( self.file_menu, wx.ID_SAVE )
        save_bitmap = wx.Bitmap( os.path.join( image_path, "save.png" ) )
        self.save_item.SetBitmap( save_bitmap )
        self.file_menu.AppendItem( self.save_item )
        self.save_item.Enable( False )

        self.save_as_item = wx.MenuItem( self.file_menu, wx.ID_SAVEAS )
        self.save_as_item.SetBitmap( save_bitmap )
        self.file_menu.AppendItem( self.save_as_item )
        self.save_as_item.Enable( False )

        self.save_image_id = wx.NewId()
        self.save_image_item = wx.MenuItem( self.file_menu, self.save_image_id, "Save as Image..." )
        save_image_bitmap = wx.Bitmap( os.path.join( image_path, "save_image.png" ) )
        self.save_image_item.SetBitmap( save_image_bitmap )
        self.file_menu.AppendItem( self.save_image_item )

        self.file_menu.AppendSeparator()

#        self.print_item = wx.MenuItem( self.file_menu, wx.ID_PRINT )
#        self.print_item.SetBitmap(
#            art.GetBitmap( wx.ART_PRINT, wx.ART_MENU )
#        )
#        self.file_menu.AppendItem( self.print_item )
#
#        self.file_menu.AppendSeparator()

        self.exit_item = wx.MenuItem( self.file_menu, wx.ID_EXIT )
        self.exit_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "quit.png" ) ),
        )
        self.file_menu.AppendItem( self.exit_item )

        self.edit_menu = wx.Menu()
        self.Append( self.edit_menu, "Edit" )
        
        # For some inexplicable reason, if wx.ID_UNDO is used here instead of
        # making a custom id, then the Undo menu entry is almost always
        # disabled (grayed out), and no amount of Enable( True ) will change
        # that. This only occurs on certain Mac OS X machines.
        self.undo_id = wx.NewId()
        self.undo_item = wx.MenuItem( self.edit_menu, self.undo_id, "Undo\tCtrl-Z" )
        self.undo_bitmap = wx.Bitmap( os.path.join( image_path, "undo.png" ) )
        self.undo_item.SetBitmap( self.undo_bitmap )
        self.edit_menu.AppendItem( self.undo_item )
        self.undo_item.Enable( False )

        self.redo_id = wx.NewId()
        self.redo_item = wx.MenuItem(
            self.edit_menu, self.redo_id, "Redo\tShift-Ctrl-Z",
        )
        self.redo_bitmap = wx.Bitmap( os.path.join( image_path, "redo.png" ) )
        self.redo_item.SetBitmap( self.redo_bitmap )
        self.edit_menu.AppendItem( self.redo_item )
        self.redo_item.Enable( False )

        self.edit_menu.AppendSeparator()

        self.clear_selection = wx.MenuItem(
            self.edit_menu,
            wx.ID_CLEAR,
            "&Clear Selection",
        )
        self.clear_selection.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "clear_selection.png" ) ),
        )
        self.edit_menu.AppendItem( self.clear_selection )
        self.clear_selection.Enable( False )

        self.delete_selection = wx.MenuItem(
            self.edit_menu,
            wx.ID_DELETE,
            "&Delete Selection\tDel",
        )
        self.delete_selection.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "delete_selection.png" ) ),
        )
        self.edit_menu.AppendItem( self.delete_selection )
        self.delete_selection.Enable( False )

        self.layer_menu = wx.Menu()
        self.Append( self.layer_menu, "Layer" )

        add_layer = wx.MenuItem(
            self.layer_menu,
            wx.ID_ANY,
            "Add Layer",
            subMenu = \
                Add_layer_menu( image_path, root_layer, command_stack, frame ),
        )
        add_layer.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "add_layer.png" ) ),
        )

        self.layer_menu.AppendItem( add_layer )
        self.layer_menu.AppendSeparator()

        self.raise_item = wx.MenuItem(
            self.layer_menu,
            wx.ID_UP,
            "Raise Layer",
        )
        self.raise_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "raise.png" ) ),
        )
        self.layer_menu.AppendItem( self.raise_item )
        self.raise_item.Enable( False )

        self.lower_item = wx.MenuItem(
            self.layer_menu,
            wx.ID_DOWN,
            "Lower Layer",
        )
        self.lower_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "lower.png" ) ),
        )
        self.layer_menu.AppendItem( self.lower_item )
        self.lower_item.Enable( False )

        self.layer_menu.AppendSeparator()

        self.triangulate_id = wx.NewId()
        self.triangulate_item = wx.MenuItem(
            self.layer_menu,
            self.triangulate_id,
            "Triangulate Layer\tCtrl-T",
        )
        self.triangulate_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "triangulate.png" ) ),
        )
        self.layer_menu.AppendItem( self.triangulate_item )
        
        try:
            import pytriangle
        except ImportError:
            self.triangulate_item.Enable( False )

        self.contour_id = wx.NewId()
        self.contour_item = wx.MenuItem(
            self.layer_menu,
            self.contour_id,
            "Contour Layer...",
        )
        self.contour_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "contour.png" ) ),
        )
        self.layer_menu.AppendItem( self.contour_item )

        self.merge_layers_id = wx.NewId()
        self.merge_layers_item = wx.MenuItem(
            self.layer_menu,
            self.merge_layers_id,
            "Merge Layers...\tCtrl-M",
        )
        self.merge_layers_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "merge.png" ) ),
        )
        self.layer_menu.AppendItem( self.merge_layers_item )

        self.layer_menu.AppendSeparator()

        self.delete_layer_item = wx.MenuItem(
            self.layer_menu,
            wx.ID_REMOVE,
            "Delete Layer\tCtrl-D",
        )
        self.delete_layer_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "delete_layer.png" ) ),
        )
        self.layer_menu.AppendItem( self.delete_layer_item )
        self.delete_layer_item.Enable( False )

        self.layer_menu.AppendSeparator()

        self.pan_mode_id = wx.NewId()
        self.pan_mode_item = wx.MenuItem(
            self.layer_menu,
            self.pan_mode_id,
            "Pan",
            kind = wx.ITEM_RADIO,
        )
        if sys.platform == "darwin":
            self.pan_mode_item.SetBitmap(
                wx.Bitmap( os.path.join( image_path, "pan.png" ) ),
            )
        self.layer_menu.AppendItem( self.pan_mode_item )
        self.layer_menu.Check( self.pan_mode_id, True )

        self.add_points_mode_id = wx.NewId()
        self.add_points_mode_item = wx.MenuItem(
            self.layer_menu,
            self.add_points_mode_id,
            "Edit Points",
            kind = wx.ITEM_RADIO,
        )
        if sys.platform == "darwin":
            self.add_points_mode_item.SetBitmap(
                wx.Bitmap( os.path.join( image_path, "add_points.png" ) ),
            )
        self.layer_menu.AppendItem( self.add_points_mode_item )

        self.add_lines_mode_id = wx.NewId()
        self.add_lines_mode_item = wx.MenuItem(
            self.layer_menu,
            self.add_lines_mode_id,
            "Edit Lines",
            kind = wx.ITEM_RADIO,
        )
        if sys.platform == "darwin":
            self.add_lines_mode_item.SetBitmap(
                wx.Bitmap( os.path.join( image_path, "add_lines.png" ) ),
            )
        self.layer_menu.AppendItem( self.add_lines_mode_item )

        self.view_menu = wx.Menu()
        self.Append( self.view_menu, "View" )

        self.zoom_in_item = wx.MenuItem(
            self.view_menu,
            wx.ID_ZOOM_IN,
            "&Zoom In\tPageUp",
        )
        self.zoom_in_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "zoom_in.png" ) ),
        )
        self.view_menu.AppendItem( self.zoom_in_item )

        self.zoom_out_item = wx.MenuItem(
            self.view_menu,
            wx.ID_ZOOM_OUT,
            "&Zoom Out\tPageDown",
        )
        self.zoom_out_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "zoom_out.png" ) ),
        )
        self.view_menu.AppendItem( self.zoom_out_item )

        self.zoom_box_item = wx.MenuItem(
            self.view_menu,
            wx.ID_ZOOM_100,
            "&Zoom to Box",
        )
        self.zoom_box_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "zoom_box.png" ) ),
        )
        self.view_menu.AppendItem( self.zoom_box_item )

        self.zoom_fit_item = wx.MenuItem(
            self.view_menu,
            wx.ID_ZOOM_FIT,
            "&Zoom to Fit",
        )
        self.zoom_fit_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "zoom_fit.png" ) ),
        )
        self.view_menu.AppendItem( self.zoom_fit_item )

        self.view_menu.AppendSeparator()

        self.grid_id = wx.NewId()
        self.grid_item = wx.MenuItem(
            self.view_menu,
            self.grid_id,
            "Grid Lines\tCtrl-G",
            kind = wx.ITEM_CHECK,
        )
        if sys.platform == "darwin":
            self.grid_item.SetBitmap(
                wx.Bitmap( os.path.join( image_path, "grid.png" ) ),
            )
        self.view_menu.AppendItem( self.grid_item )
        self.view_menu.Check( self.grid_id, True )

        self.view_menu.AppendSeparator()

        self.jump_id = wx.NewId()
        self.jump_item = wx.MenuItem(
            self.view_menu,
            self.jump_id,
            "Jump to Coordinates...\tCtrl-J",
        )
        self.jump_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "jump.png" ) ),
        )
        self.view_menu.AppendItem( self.jump_item )

        self.tools_menu = wx.Menu()
        self.Append( self.tools_menu, "Tools" )
#        self.tools_menu.Append( wx.ID_PREFERENCES )

        self.merge_duplicates_id = wx.NewId()
        self.merge_duplicates_item = wx.MenuItem(
            self.tools_menu,
            self.merge_duplicates_id,
            "Merge Duplicate Points...",
        )
        self.merge_duplicates_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "merge_duplicates.png" ) ),
        )
        self.tools_menu.AppendItem( self.merge_duplicates_item )
        self.merge_duplicates_item.Enable( False )

        self.debug_menu = wx.Menu()
        debug = wx.MenuItem(
            self.tools_menu,
            wx.ID_ANY,
            "Debug",
            subMenu = self.debug_menu,
        )
        debug.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "debug.png" ) ),
        )

        self.tools_menu.AppendItem( debug )

        self.log_id = wx.NewId()
        self.log_item = wx.MenuItem(
            self.debug_menu,
            self.log_id,
            "View Log Messages...",
        )
        self.log_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "view_log.png" ) ),
        )
        self.debug_menu.AppendItem( self.log_item )

        self.help_menu = wx.Menu()
        self.Append( self.help_menu, "&Help" )

        self.help_id = wx.NewId()
        self.help_item = wx.MenuItem(
            self.help_menu,
            self.help_id,
            "Maproom Help...\tF1",
        )
        self.help_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "help.png" ) ),
        )
        self.help_menu.AppendItem( self.help_item )

        self.help_menu.AppendSeparator()

        self.about_item = wx.MenuItem(
            self.help_menu,
            wx.ID_ABOUT,
            "&About Maproom...",
        )
        self.about_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "maproom.png" ) ),
        )
        self.help_menu.AppendItem( self.about_item )

        self.command_stack.outbox.subscribe(
            self.inbox,
            request = "undo_updated",
        )
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = (
                "selection_updated",
                "pan_mode",
                "add_points_mode",
                "add_lines_mode",
            ),
        )

        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "open_file" ),
            id = wx.ID_OPEN,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "save_file" ),
            id = wx.ID_SAVE,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "save_as_file" ),
            id = wx.ID_SAVEAS,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "save_image" ),
            id = self.save_image_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "undo" ),
            id = self.undo_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "redo" ),
            id = self.redo_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "undo" ),
            id = wx.ID_UNDO,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "redo" ),
            id = wx.ID_REDO,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "raise" ),
            id = wx.ID_UP,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "lower" ),
            id = wx.ID_DOWN,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "clear" ),
            id = wx.ID_CLEAR,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "delete" ),
            id = wx.ID_DELETE,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "delete_layer" ),
            id = wx.ID_REMOVE,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "contour_layer" ),
            id = self.contour_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "merge_layers" ),
            id = self.merge_layers_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "pan_mode_clicked" ),
            id = self.pan_mode_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "add_points_mode_clicked" ),
            id = self.add_points_mode_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "add_lines_mode_clicked" ),
            id = self.add_lines_mode_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            lambda event: self.viewport.zoom( 1 ),
            id = wx.ID_ZOOM_IN,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            lambda event: self.viewport.zoom( -1 ),
            id = wx.ID_ZOOM_OUT,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "zoom_box" ),
            id = wx.ID_ZOOM_100,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "zoom_fit" ),
            id = wx.ID_ZOOM_FIT,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "toggle_grid" ),
            id = self.grid_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "triangulate" ),
            id = self.triangulate_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "jump" ),
            id = self.jump_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "merge_duplicates" ),
            id = self.merge_duplicates_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "view_log" ),
            id = self.log_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "help" ),
            id = self.help_id,
        )
        self.frame.Bind(
            wx.EVT_MENU,
            ui.Wx_handler( self.inbox, "about" ),
            id = wx.ID_ABOUT,
        )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = (
                    "open_file", "save_file", "save_as_file", "save_image",
                    "undo", "redo",
                    "raise", "lower",
                    "clear", "delete", "undo_updated",
                    "selection_updated",
                    "triangulate",
                    "delete_layer", "contour_layer", "merge_layers",
                    "pan_mode", "add_points_mode", "add_lines_mode",
                    "pan_mode_clicked", "add_points_mode_clicked",
                    "add_lines_mode_clicked",
                    "zoom_fit", "zoom_box", "toggle_grid",
                    "jump", "merge_duplicates", "view_log", "about", "help",
                ),
            )
            request = message.pop( "request" )
            
            if request == "open_file":
                scheduler.add(
                    ui.File_opener(
                        self.frame,
                        self.root_layer,
                        self.command_stack,
                    ).run
                )
            elif request == "save_file":
                scheduler.add(
                    ui.File_saver(
                        self.frame,
                        self.root_layer,
                        self.command_stack,
                        always_prompt_for_filename = False,
                    ).run
                )
            elif request == "save_as_file":
                scheduler.add(
                    ui.File_saver(
                        self.frame,
                        self.root_layer,
                        self.command_stack,
                        always_prompt_for_filename = True,
                    ).run
                )
            elif request == "save_image":
                scheduler.add(
                    ui.Screenshot_saver(
                        self.frame,
                        self.renderer,
                    ).run
                )
            elif request == "undo":
                # Don't allow undo or redo if a mouse button is down, because
                # the user could be in the middle of dragging a point, and
                # undoing at that point would screw up the undo stack.
                if wx.GetMouseState().LeftDown():
                    continue
                self.command_stack.inbox.send( request = "undo" )
            elif request == "redo":
                if wx.GetMouseState().LeftDown():
                    continue
                self.command_stack.inbox.send( request = "redo" )
            elif request == "raise":
                self.command_stack.inbox.send( request = "start_command" )
                self.layer_tree.inbox.send( request = "raise_layer" )
            elif request == "lower":
                self.command_stack.inbox.send( request = "start_command" )
                self.layer_tree.inbox.send( request = "lower_layer" )
            elif request == "clear":
                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( request = "clear_selection" )
            elif request == "delete":
                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( request = "delete_selection" )
            elif request == "undo_updated":
                self.undo_updated( **message )
            elif request == "selection_updated":
                self.selection_updated( **message )
            elif request == "delete_layer":
                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( request = "delete_layer" )
            elif request == "triangulate":
                """
                self.command_stack.inbox.send( request = "start_command" )
                scheduler.add(
                    ui.Triangulator(
                        self.frame,
                        self.root_layer,
                        self.renderer.transformer,
                    ).run,
                )
                """
                if self.triangle_maker is None:
                    self.triangle_maker = ui.Triangle_maker(
                        self.frame,
                        self.root_layer,
                        self.renderer.transformer,
                        self.command_stack
                    )
                    scheduler.add( self.triangle_maker.run )
                else:
                    self.triangle_maker.inbox.send( request = "show" )
            elif request == "contour_layer":
                if self.contour_maker is None:
                    self.contour_maker = ui.Contour_maker(
                        self.frame,
                        self.root_layer,
                        self.command_stack
                    )
                    scheduler.add( self.contour_maker.run )
                else:
                    self.contour_maker.inbox.send( request = "show" )
            elif request == "merge_layers":
                self.command_stack.inbox.send( request = "start_command" )
                scheduler.add(
                    ui.Merge_layers_dialog( self.frame, self.root_layer ).run,
                )
            elif request == "pan_mode_clicked":
                self.root_layer.inbox.send( request = "pan_mode" )
            elif request == "add_points_mode_clicked":
                self.root_layer.inbox.send( request = "add_points_mode" )
            elif request == "add_lines_mode_clicked":
                self.root_layer.inbox.send( request = "add_lines_mode" )
            elif request == "pan_mode":
                self.layer_menu.Check( self.pan_mode_id, True )
            elif request == "add_points_mode":
                self.layer_menu.Check( self.add_points_mode_id, True )
            elif request == "add_lines_mode":
                self.command_stack.inbox.send( request = "start_command" )
                self.root_layer.inbox.send( request = "clear_selection" )
                self.layer_menu.Check( self.add_lines_mode_id, True )
            elif request == "zoom_box":
                self.root_layer.inbox.send( request = "start_zoom_box" )
            elif request == "zoom_fit":
                self.root_layer.inbox.send(
                    request = "get_dimensions",
                    response_box = self.inbox,
                )

                message = self.inbox.receive( request = "dimensions" )

                self.viewport.jump_geo_boundary( 
                    message.get( "origin" ),
                    message.get( "size" ),
                    message.get( "projection" ),
                )
            elif request == "toggle_grid":
                self.renderer.toggle_grid_lines()
            elif request == "jump":
                scheduler.add(
                    ui.Coordinate_jumper( self.frame, self.viewport ).run,
                )
            elif request == "merge_duplicates":
                if self.duplicate_finder is None:
                    self.duplicate_finder = ui.Duplicate_finder(
                        self.frame, self.root_layer, self.command_stack,
                        self.viewport,
                    )
                    scheduler.add( self.duplicate_finder.run )
                else:
                    self.duplicate_finder.inbox.send( request = "show" )
            elif request == "view_log":
                self.log_viewer.inbox.send( request = "show" )
            elif request == "about":
                scheduler.add( About_dialog( self.version ).run )
            elif request == "help":
                Help_window( self.frame )

    def undo_updated( self, next_undo_description, next_redo_description ):
        # Deleting and re-adding these menu items, because if we don't, then
        # trying to update the item by calling SetItemLabel() makes the item
        # bitmap disappear.
        self.edit_menu.Delete( self.undo_id )
        self.edit_menu.Delete( self.redo_id )

        if next_undo_description is None:
            undo_label = "Undo\tCtrl-Z"
            undo_enabled = False
        else:
            undo_label = "Undo %s\tCtrl-Z" % next_undo_description
            undo_enabled = True

        if next_redo_description is None:
            redo_label = "Redo\tShift-Ctrl-Z"
            redo_enabled = False
        else:
            redo_label = "Redo %s\tShift-Ctrl-Z" % next_redo_description
            redo_enabled = True

        self.redo_item = wx.MenuItem(
            self.edit_menu, self.redo_id, redo_label,
        )
        self.redo_item.SetBitmap( self.redo_bitmap )
        self.edit_menu.PrependItem( self.redo_item )
        self.redo_item.Enable( redo_enabled )

        self.undo_item = wx.MenuItem( self.edit_menu, self.undo_id, undo_label )
        self.undo_item.SetBitmap( self.undo_bitmap )
        self.edit_menu.PrependItem( self.undo_item )
        self.undo_item.Enable( undo_enabled )

    def selection_updated( self, selections = None, raisable = False,
                           lowerable = False, deletable = False,
                           layer_deletable = None ):
        if raisable is not None:
            self.raise_item.Enable( raisable )
        if lowerable is not None:
            self.lower_item.Enable( lowerable )

        self.clear_selection.Enable( deletable )
        self.delete_selection.Enable( deletable )

        if layer_deletable is not None:
            self.delete_layer_item.Enable( layer_deletable )
            self.save_item.Enable( layer_deletable )
            self.save_as_item.Enable( layer_deletable )
            self.merge_duplicates_item.Enable( layer_deletable )

            if self.duplicate_finder:
                self.duplicate_finder.panel.Enable( layer_deletable )
