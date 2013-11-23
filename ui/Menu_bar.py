import wx
import os
import sys
from wx.lib.pubsub import pub

from About_dialog import About_dialog
from Jump_coords_dialog import JumpCoordsDialog
from Find_point_dialog import FindPointDialog
from Preferences_dialog import PreferencesDialog
from Triangle_dialog import Triangle_dialog
# from Help_window import Help_window
# from Add_layer_menu import Add_layer_menu
import File_opener
import File_saver
import app_globals
import Layer
import library.coordinates as coordinates


class Menu_bar(wx.MenuBar):

    """
    Main application menu bar.
    """
    IMAGE_PATH = "ui/images"

    def __init__(self, controller, frame):
        wx.MenuBar.__init__(self)

        self.controller = controller
        self.frame = frame

        image_path = self.IMAGE_PATH

        self.SetAutoWindowMenu(False)

        self.frame.Bind(wx.EVT_MENU,
                        lambda event: wx.MessageDialog(
                            self.frame, "Not implemented yet.", style=wx.OK | wx.ICON_ERROR
                        ).ShowModal()
                        )

        self.file_menu = wx.Menu()

        self.new_map_item = wx.MenuItem(self.file_menu, wx.NewId(), "New Map...\tCtrl-N")
        self.file_menu.AppendItem(self.new_map_item)

        self.open_item = wx.MenuItem(self.file_menu, wx.ID_OPEN, "Open...\tCtrl-O", )
        self.open_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "open.png")))
        self.file_menu.AppendItem(self.open_item)
        
        recent = wx.Menu()
        pub.sendMessage(('recent_files', 'config'), menu=recent)
        self.file_menu.AppendMenu(wx.ID_ANY, "&Recent Files", recent)

        self.file_menu.AppendSeparator()

        self.save_item = wx.MenuItem(self.file_menu, wx.ID_SAVE)
        save_bitmap = wx.Bitmap(os.path.join(image_path, "save.png"))
        self.save_item.SetBitmap(save_bitmap)
        self.file_menu.AppendItem(self.save_item)

        self.save_as_item = wx.MenuItem(self.file_menu, wx.ID_SAVEAS)
        self.save_as_item.SetBitmap(save_bitmap)
        self.file_menu.AppendItem(self.save_as_item)

        self.save_image_id = wx.NewId()
        self.save_image_item = wx.MenuItem(self.file_menu, self.save_image_id, "Save as Image...")
        save_image_bitmap = wx.Bitmap(os.path.join(image_path, "save_image.png"))
        self.save_image_item.SetBitmap(save_image_bitmap)
        self.file_menu.AppendItem(self.save_image_item)

        self.file_menu.AppendSeparator()

        # self.print_item = wx.MenuItem( self.file_menu, wx.ID_PRINT )
        # self.print_item.SetBitmap(
        #   art.GetBitmap( wx.ART_PRINT, wx.ART_MENU )
        # )
        # self.file_menu.AppendItem( self.print_item )
        #
        # self.file_menu.AppendSeparator()

        self.exit_item = wx.MenuItem(self.file_menu, wx.ID_EXIT)
        self.exit_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "quit.png")))
        self.file_menu.AppendItem(self.exit_item)

        # wait to add the File menu till after all the menu items are added
        # so that the Quit item will be moved out of the File menu to the app
        # menu as per Apple style guideles
        self.Append(self.file_menu, "&File")
        
        #

        self.edit_menu = wx.Menu()

        # For some inexplicable reason, if wx.ID_UNDO is used here instead of
        # making a custom id, then the Undo menu entry is almost always
        # disabled (grayed out), and no amount of Enable( True ) will change
        # that. This only occurs on certain Mac OS X machines.
        self.undo_id = wx.NewId()
        self.undo_item = wx.MenuItem(self.edit_menu, self.undo_id, "Undo\tCtrl-Z")
        self.undo_bitmap = wx.Bitmap(os.path.join(image_path, "undo.png"))
        self.undo_item.SetBitmap(self.undo_bitmap)
        self.edit_menu.AppendItem(self.undo_item)

        self.redo_id = wx.NewId()
        self.redo_item = wx.MenuItem(self.edit_menu, self.redo_id, "Redo\tCtrl-Y")
        self.redo_bitmap = wx.Bitmap(os.path.join(image_path, "redo.png"))
        self.redo_item.SetBitmap(self.redo_bitmap)
        self.edit_menu.AppendItem(self.redo_item)

        self.edit_menu.AppendSeparator()

        self.delete_selection = wx.MenuItem(self.edit_menu, wx.ID_DELETE, "&Delete Selection\tDel")
        self.delete_selection.SetBitmap(wx.Bitmap(os.path.join(image_path, "delete_selection.png")))
        self.edit_menu.AppendItem(self.delete_selection)

        self.edit_menu.AppendSeparator()

        self.clear_selection = wx.MenuItem(self.edit_menu, wx.ID_CLEAR, "&Clear Selection\tEsc", )
        self.clear_selection.SetBitmap(wx.Bitmap(os.path.join(image_path, "clear_selection.png")))
        self.edit_menu.AppendItem(self.clear_selection)

        self.edit_menu.AppendSeparator()

        self.find_point_id = wx.NewId()
        self.find_point_item = wx.MenuItem(self.edit_menu, self.find_point_id, "Find Points...\tCtrl-F")
        self.edit_menu.AppendItem(self.find_point_item)

        self.edit_menu.AppendSeparator()
        prefs_shortcut = ""  # does Windows have a standard shortcut for Preferences?
        if sys.platform.startswith('darwin'):
            prefs_shortcut = "\tCTRL+,"
        self.preferences = wx.MenuItem(self.edit_menu, wx.ID_PREFERENCES, "Preferences" + prefs_shortcut)
        self.edit_menu.AppendItem(self.preferences)
        self.Append(self.edit_menu, "Edit")

        #

        self.layer_menu = wx.Menu()
        self.Append(self.layer_menu, "Layer")

        add_layer = wx.MenuItem(
            self.layer_menu,
            wx.ID_ANY,
            "New Grid Layer",
            # subMenu = Add_layer_menu( image_path, root_layer, command_stack, frame )
        )
        add_layer.SetBitmap(wx.Bitmap(os.path.join(image_path, "add_layer.png")))
        self.layer_menu.AppendItem(add_layer)

        add_folder = wx.MenuItem(
            self.layer_menu,
            wx.ID_ANY,
            "New Folder",
            # subMenu = Add_layer_menu( image_path, root_layer, command_stack, frame )
        )
        add_folder.SetBitmap(wx.Bitmap(os.path.join(image_path, "add_folder.png")))
        # FIXME: folders are still not implemented, so disabling UI
        add_folder.Enable(False)
        self.layer_menu.AppendItem(add_folder)

        self.layer_menu.AppendSeparator()

        self.raise_item = wx.MenuItem(self.layer_menu, wx.ID_UP, "Raise Layer", )
        self.raise_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "raise.png")))
        self.layer_menu.AppendItem(self.raise_item)

        self.lower_item = wx.MenuItem(self.layer_menu, wx.ID_DOWN, "Lower Layer", )
        self.lower_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "lower.png")))
        self.layer_menu.AppendItem(self.lower_item)

        self.layer_menu.AppendSeparator()

        self.triangulate_id = wx.NewId()
        self.triangulate_item = wx.MenuItem(self.layer_menu, self.triangulate_id, "Triangulate Layer\tCtrl-T")
        self.triangulate_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "triangulate.png")))
        self.layer_menu.AppendItem(self.triangulate_item)

        self.contour_id = wx.NewId()
        self.contour_item = wx.MenuItem(self.layer_menu, self.contour_id, "Contour Layer...")
        self.contour_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "contour.png")))
        self.layer_menu.AppendItem(self.contour_item)

        self.merge_layers_id = wx.NewId()
        self.merge_layers_item = wx.MenuItem(self.layer_menu, self.merge_layers_id, "Merge Layers...\tCtrl-M")
        self.merge_layers_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "merge.png")))
        self.layer_menu.AppendItem(self.merge_layers_item)

        self.merge_duplicate_points_id = wx.NewId()
        self.merge_duplicate_points_item = wx.MenuItem(self.layer_menu, self.merge_duplicate_points_id, "Merge Duplicate Points...")
        self.merge_duplicate_points_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "merge_duplicates.png")))
        self.layer_menu.AppendItem(self.merge_duplicate_points_item)

        self.layer_menu.AppendSeparator()

        self.delete_layer_item = wx.MenuItem(self.layer_menu, wx.ID_REMOVE, "Delete Layer\tCtrl-D")
        self.delete_layer_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "delete_layer.png")))
        self.layer_menu.AppendItem(self.delete_layer_item)

        self.layer_menu.AppendSeparator()

        self.zoom_box_item = wx.MenuItem(self.layer_menu, wx.ID_ZOOM_100, "&Zoom to Box")
        self.zoom_box_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "zoom_box.png")))
        self.layer_menu.AppendItem(self.zoom_box_item)

        self.pan_mode_id = wx.NewId()
        self.pan_mode_item = wx.MenuItem(self.layer_menu, self.pan_mode_id, "Pan", kind=wx.ITEM_RADIO)
        if sys.platform == "darwin":
            self.pan_mode_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "pan.png")))
        self.layer_menu.AppendItem(self.pan_mode_item)

        self.add_points_mode_id = wx.NewId()
        self.add_points_mode_item = wx.MenuItem(self.layer_menu, self.add_points_mode_id, "Edit Points", kind=wx.ITEM_RADIO)
        if sys.platform == "darwin":
            self.add_points_mode_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "add_points.png")))
        self.layer_menu.AppendItem(self.add_points_mode_item)

        self.add_lines_mode_id = wx.NewId()
        self.add_lines_mode_item = wx.MenuItem(self.layer_menu, self.add_lines_mode_id, "Edit Lines", kind=wx.ITEM_RADIO)
        if sys.platform == "darwin":
            self.add_lines_mode_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "add_lines.png")))
        self.layer_menu.AppendItem(self.add_lines_mode_item)

        self.layer_menu.AppendSeparator()

        self.check_valid_verdat_id = wx.NewId()
        self.check_valid_verdat_item = wx.MenuItem(self.layer_menu, self.check_valid_verdat_id, "Check for errors")
        self.layer_menu.AppendItem(self.check_valid_verdat_item)

        #

        self.view_menu = wx.Menu()
        self.Append(self.view_menu, "View")

        self.zoom_in_item = wx.MenuItem(self.view_menu, wx.ID_ZOOM_IN, "&Zoom In\tPageUp")
        self.zoom_in_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "zoom_in.png")))
        self.view_menu.AppendItem(self.zoom_in_item)

        self.zoom_out_item = wx.MenuItem(self.view_menu, wx.ID_ZOOM_OUT, "&Zoom Out\tPageDown")
        self.zoom_out_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "zoom_out.png")))
        self.view_menu.AppendItem(self.zoom_out_item)

        self.zoom_fit_item = wx.MenuItem(self.view_menu, wx.ID_ZOOM_FIT, "&Zoom to Fit", )
        self.zoom_fit_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "zoom_fit.png")))
        self.view_menu.AppendItem(self.zoom_fit_item)

        self.zoom_layer_id = wx.NewId()
        self.zoom_layer_item = wx.MenuItem(self.view_menu, self.zoom_layer_id, "&Zoom to Selected Layer", )
        self.view_menu.AppendItem(self.zoom_layer_item)

        self.view_menu.AppendSeparator()

        self.grid_id = wx.NewId()
        self.grid_item = wx.MenuItem(self.view_menu, self.grid_id, "Grid Lines\tCtrl-G", kind=wx.ITEM_CHECK)
        if sys.platform == "darwin":
            self.grid_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "grid.png")))
        self.view_menu.AppendItem(self.grid_item)

        self.view_menu.AppendSeparator()

        self.bbox_id = wx.NewId()
        self.bbox_item = wx.MenuItem(self.view_menu, self.bbox_id, "Layer Bounding Boxes", kind=wx.ITEM_CHECK)
        self.view_menu.AppendItem(self.bbox_item)

        self.view_menu.AppendSeparator()

        self.jump_id = wx.NewId()
        self.jump_item = wx.MenuItem(self.view_menu, self.jump_id, "Jump to Coordinates...\tCtrl-J")
        self.jump_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "jump.png")))
        self.view_menu.AppendItem(self.jump_item)

        #

        self.tools_menu = wx.Menu()
        self.Append(self.tools_menu, "Tools")
        # self.tools_menu.Append( wx.ID_PREFERENCES )

        self.debug_menu = wx.Menu()
        debug = wx.MenuItem(self.tools_menu, wx.ID_ANY, "Debug", subMenu=self.debug_menu)
        debug.SetBitmap(wx.Bitmap(os.path.join(image_path, "debug.png")))
        self.tools_menu.AppendItem(debug)

        self.log_id = wx.NewId()
        self.log_item = wx.MenuItem(self.debug_menu, self.log_id, "View Log Messages...")
        self.log_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "view_log.png")))
        self.debug_menu.AppendItem(self.log_item)

        #

        self.help_menu = wx.Menu()

        self.help_id = wx.NewId()
        self.help_item = wx.MenuItem(self.help_menu, self.help_id, "Maproom Help...\tF1")
        self.help_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "help.png")))
        self.help_menu.AppendItem(self.help_item)

        self.help_menu.AppendSeparator()

        self.about_item = wx.MenuItem(self.help_menu, wx.ID_ABOUT, "&About Maproom...")
        self.about_item.SetBitmap(wx.Bitmap(os.path.join(image_path, "maproom.png")))
        self.help_menu.AppendItem(self.about_item)

        self.Append(self.help_menu, "&Help")

        #

        f = self.frame
        f.Bind(wx.EVT_MENU, self.do_new_map, self.new_map_item)
        f.Bind(wx.EVT_MENU, self.do_open_file, id=wx.ID_OPEN)
        f.Bind(wx.EVT_MENU, self.do_save, id=wx.ID_SAVE)
        f.Bind(wx.EVT_MENU, self.do_save_as, id=wx.ID_SAVEAS)
        f.Bind(wx.EVT_MENU, self.do_save_image, id=self.save_image_id)
        f.Bind(wx.EVT_MENU, self.do_undo, id=self.undo_id)
        f.Bind(wx.EVT_MENU, self.do_redo, id=self.redo_id)
        f.Bind(wx.EVT_MENU, self.do_raise_layer, id=wx.ID_UP)
        f.Bind(wx.EVT_MENU, self.do_lower_layer, id=wx.ID_DOWN)
        f.Bind(wx.EVT_MENU, self.do_clear, id=wx.ID_CLEAR)
        f.Bind(wx.EVT_MENU, self.do_delete, id=wx.ID_DELETE)
        f.Bind(wx.EVT_MENU, self.do_add_layer, id=add_layer.Id)
        f.Bind(wx.EVT_MENU, self.do_add_folder, id=wx.ID_ADD)
        f.Bind(wx.EVT_MENU, self.do_delete_layer, id=wx.ID_REMOVE)
        f.Bind(wx.EVT_MENU, self.do_countour_layer, id=self.contour_id)
        # f.Bind( wx.EVT_MENU, self.do_merge_layers, id = self.merge_layers_id )
        f.Bind(wx.EVT_MENU, self.do_set_zoom_to_box_mode, id=wx.ID_ZOOM_100)
        f.Bind(wx.EVT_MENU, self.do_set_pan_mode, id=self.pan_mode_id)
        f.Bind(wx.EVT_MENU, self.do_set_add_points_mode, id=self.add_points_mode_id)
        f.Bind(wx.EVT_MENU, self.do_set_add_lines_mode, id=self.add_lines_mode_id)
        f.Bind(wx.EVT_MENU, self.do_zoom_in, id=wx.ID_ZOOM_IN)
        f.Bind(wx.EVT_MENU, self.do_zoom_out, id=wx.ID_ZOOM_OUT)
        f.Bind(wx.EVT_MENU, self.do_zoom_fit, id=wx.ID_ZOOM_FIT)
        f.Bind(wx.EVT_MENU, self.do_zoom_layer, id=self.zoom_layer_id)
        f.Bind(wx.EVT_MENU, self.do_toggle_grid, id=self.grid_id)
        f.Bind(wx.EVT_MENU, self.do_toggle_bbox, id=self.bbox_id)
        f.Bind(wx.EVT_MENU, self.do_triangulate, id=self.triangulate_id)
        f.Bind(wx.EVT_MENU, self.do_merge_layers, id=self.merge_layers_id)
        f.Bind(wx.EVT_MENU, self.do_check_for_errors, id=self.check_valid_verdat_id)
        f.Bind(wx.EVT_MENU, self.do_jump, id=self.jump_id)
        f.Bind(wx.EVT_MENU, self.do_find_point, id=self.find_point_id)
        f.Bind(wx.EVT_MENU, self.do_merge_duplicate_points, id=self.merge_duplicate_points_id)
        f.Bind(wx.EVT_MENU, self.do_view_log, id=self.log_id)
        f.Bind(wx.EVT_MENU, self.do_show_help, id=self.help_id)
        f.Bind(wx.EVT_MENU, self.do_show_about, id=wx.ID_ABOUT)
        f.Bind(wx.EVT_MENU, self.do_show_preferences, id=wx.ID_PREFERENCES)
        f.Bind(wx.EVT_MENU_RANGE, self.on_file_history, id=wx.ID_FILE1, id2=wx.ID_FILE1+app_globals.preferences["Number of Recent Files"])

    def enable_disable_menu_items(self):
        raisable = self.controller.layer_tree_control.is_selected_layer_raisable()
        self.raise_item.Enable(raisable)
        lowerable = self.controller.layer_tree_control.is_selected_layer_lowerable()
        self.lower_item.Enable(lowerable)
        self.updated_undo_redo()
        
        layer = self.controller.layer_tree_control.get_selected_layer()
        enabled = layer is not None and layer.has_points()
        self.find_point_item.Enable(enabled)

        self.grid_item.Check(self.controller.renderer.lon_lat_grid_shown)
        self.bbox_item.Check(self.controller.renderer.bounding_boxes_shown)
        """
        self.save_item.Enable( False )
        self.save_as_item.Enable( False )
        self.clear_selection.Enable( False )
        self.redo_item.Enable( False )
        self.undo_item.Enable( False )
        self.delete_selection.Enable( False )
        self.triangulate_item.Enable( False )
        self.delete_layer_item.Enable( False )
        self.layer_menu.Check( self.pan_mode_id, True )
        self.view_menu.Check( self.grid_id, True )
        
        self.clear_selection.Enable( deletable )
        self.delete_selection.Enable( deletable )

        if layer_deletable is not None:
            self.delete_layer_item.Enable( layer_deletable )
            self.save_item.Enable( layer_deletable )
            self.save_as_item.Enable( layer_deletable )
            self.merge_duplicates_item.Enable( layer_deletable )

            if self.duplicate_finder:
                self.duplicate_finder.panel.Enable( layer_deletable )
        
        self.EnableTool( wx.ID_SAVE, False )
        self.EnableTool( wx.ID_UNDO, False )
        self.EnableTool( wx.ID_REDO, False )
        self.EnableTool( self.triangulate_id, False )
        self.EnableTool( wx.ID_REMOVE, False )
        """

    def do_new_map(self, event):
        app_globals.application.new_map()

    def do_open_file(self, event):
        File_opener.show()

    def do_save(self, event):
        layer = self.controller.layer_tree_control.get_selected_layer()
        if layer is None:
            return

        if layer.file_path == "":
            File_saver.show()
            return
        app_globals.layer_manager.save_layer(layer, layer.file_path)

    def do_save_as(self, event):
        File_saver.show()

    def do_save_image(self, event):
        dialog = wx.FileDialog(
            app_globals.current_frame,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            message="Save as Image",
            wildcard="PNG Image (*.png) |*.png"
        )

        if dialog.ShowModal() == wx.ID_OK:
            image = self.controller.renderer.get_canvas_as_image()
            image.SaveFile(dialog.GetPath(), wx.BITMAP_TYPE_PNG)
        dialog.Destroy()

    def do_undo(self, event):
        app_globals.editor.undo()

    def do_redo(self, event):
        app_globals.editor.redo()

    def do_raise_layer(self, event):
        layer = self.controller.layer_tree_control.get_selected_layer()
        self.controller.layer_tree_control.raise_selected_layer()
        self.controller.layer_tree_control.select_layer(layer)

    def do_lower_layer(self, event):
        layer = self.controller.layer_tree_control.get_selected_layer()
        self.controller.layer_tree_control.lower_selected_layer()
        self.controller.layer_tree_control.select_layer(layer)

    def do_clear(self, event):
        app_globals.editor.esc_key_pressed()

    def do_delete(self, event):
        app_globals.editor.delete_key_pressed()

    def do_add_layer(self, event):
        app_globals.layer_manager.add_layer()

    def do_add_folder(self, event):
        app_globals.layer_manager.add_folder()

    def do_delete_layer(self, event):
        app_globals.layer_manager.delete_selected_layer()

    def do_countour_layer(self, event):
        pass

    def do_set_zoom_to_box_mode(self, event):
        print "in zoom to box handler"
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_POINTS):
            app_globals.editor.point_tool_deselected()
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_LINES):
            app_globals.editor.line_tool_deselected()
        self.controller.renderer.mode = self.controller.MODE_ZOOM_RECT
        print "calling refresh on application"
        self.controller.refresh()

    def do_set_pan_mode(self, event):
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_POINTS):
            app_globals.editor.point_tool_deselected()
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_LINES):
            app_globals.editor.line_tool_deselected()
        self.controller.renderer.mode = self.controller.MODE_PAN

    def do_set_add_points_mode(self, event):
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_LINES):
            app_globals.editor.line_tool_deselected()
        self.controller.renderer.mode = self.controller.MODE_EDIT_POINTS
        app_globals.editor.point_tool_selected()

    def do_set_add_lines_mode(self, event):
        if (self.controller.renderer.mode == self.controller.MODE_EDIT_POINTS):
            app_globals.editor.point_tool_deselected()
        self.controller.renderer.mode = self.controller.MODE_EDIT_LINES
        app_globals.editor.line_tool_selected()

    def do_zoom_in(self, event):
        self.controller.renderer.zoom_in()

    def do_zoom_out(self, event):
        self.controller.renderer.zoom_out()

    def do_zoom_fit(self, event):
        self.controller.renderer.zoom_to_fit()

    def do_zoom_layer(self, event):
        self.controller.render_controller.zoom_to_selected_layer()

    def do_toggle_grid(self, event):
        self.controller.renderer.lon_lat_grid_shown = not self.controller.renderer.lon_lat_grid_shown
        self.controller.refresh()

    def do_toggle_bbox(self, event):
        self.controller.renderer.bounding_boxes_shown = not self.controller.renderer.bounding_boxes_shown
        self.controller.refresh()

    def do_triangulate(self, event):
        self.controller.show_triangle_dialog_box()

    def do_merge_layers(self, event):
        self.controller.show_merge_layers_dialog_box()

    def do_check_for_errors(self, event):
        layer = self.controller.layer_tree_control.get_selected_layer()
        layer.check_for_errors()

    def do_merge_duplicate_points(self, event):
        self.controller.show_merge_duplicate_points_dialog_box()

    def do_jump(self, event):
        dialog = JumpCoordsDialog(None, wx.ID_ANY, "Jump to Coordinates")
        if dialog.ShowModal() == wx.ID_OK:
            lat_lon = coordinates.lat_lon_from_format_string(dialog.coords_text.Value)
            app = wx.GetApp()
            renderer = app.current_map.renderer
            renderer.projected_point_center = renderer.get_projected_point_from_world_point(lat_lon)
            app.refresh()
        dialog.Destroy()

    def do_find_point(self, event):
        dialog = FindPointDialog(None, wx.ID_ANY, "Find Points")
        if dialog.ShowModal() == wx.ID_OK:
            try:
                values, error = dialog.get_values()
                layer = dialog.layer
                if len(values) > 0 and layer.has_points():
                    index = values[0]
                    point = layer.points[index]
                    print "point[%d]=%s" % (index, str(point))
                    app = wx.GetApp()
                    layer.clear_all_point_selections()
                    for index in values:
                        layer.select_point(index)
                    renderer = app.current_map.renderer
                    w_r = layer.compute_bounding_rect_of_points(Layer.STATE_SELECTED)
                    renderer.zoom_to_include_world_rect(w_r)
                    app.refresh()
                if error:
                    tlw = wx.GetApp().GetTopWindow()
                    tlw.SetStatusText(error)
            except IndexError:
                tlw = wx.GetApp().GetTopWindow()
                tlw.SetStatusText(u"No point #%s in this layer" % value)
            except ValueError:
                tlw = wx.GetApp().GetTopWindow()
                tlw.SetStatusText(u"Point number must be an integer, not '%s'" % value)
            except:
                raise
        dialog.Destroy()

    def do_view_log(self, event):
        pass

    def do_show_help(self, event):
        pass

    def do_show_about(self, event):
        About_dialog().show()

    def do_show_preferences(self, event):
        PreferencesDialog(None, wx.ID_ANY, "Maproom Preferences").Show()
        
    def on_file_history(self, event):
        index = event.GetId() - wx.ID_FILE1
        pub.sendMessage(('recent_files', 'open'), index=index)

    def updated_undo_redo(self):
        u = app_globals.editor.get_current_undoable_operation_text()
        r = app_globals.editor.get_current_redoable_operation_text()

        # must delete and re-add these menu items;
        # trying to update the item by calling SetItemLabel() makes the item bitmap disappear
        self.edit_menu.Delete(self.undo_id)
        self.edit_menu.Delete(self.redo_id)

        if (u == ""):
            undo_label = "Undo\tCtrl-Z"
            undo_enabled = False
        else:
            undo_label = "Undo {0}\tCtrl-Z".format(u)
            undo_enabled = True

        if (r == ""):
            redo_label = "Redo\tCtrl-Y"
            redo_enabled = False
        else:
            redo_label = "Redo {0}\tCtrl-Y".format(r)
            redo_enabled = True

        self.redo_item = wx.MenuItem(
            self.edit_menu, self.redo_id, redo_label,
        )
        self.redo_item.SetBitmap(self.redo_bitmap)
        self.edit_menu.PrependItem(self.redo_item)
        self.redo_item.Enable(redo_enabled)

        self.undo_item = wx.MenuItem(self.edit_menu, self.undo_id, undo_label)
        self.undo_item.SetBitmap(self.undo_bitmap)
        self.edit_menu.PrependItem(self.undo_item)
        self.undo_item.Enable(undo_enabled)

"""
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
    
    def
        self.frame.Bind( wx.EVT_MENU, lambda event: self.viewport.zoom( 1 ), id = wx.ID_ZOOM_IN )
        self.frame.Bind( wx.EVT_MENU, lambda event: self.viewport.zoom( -1 ), id = wx.ID_ZOOM_OUT )

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
"""
