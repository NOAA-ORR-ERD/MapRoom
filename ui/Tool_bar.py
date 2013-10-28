import wx
import os
import sys
import Menu_bar
import pytriangle
import app_globals


class Tool_bar(wx.ToolBar):

    """
    Main application menu bar.
    """
    IMAGE_PATH = "ui/images"

    def __init__(self, controller):
        self.controller = controller
        self.frame = controller.frame

        wx.ToolBar.__init__(self, self.frame, style=wx.TB_FLAT)

        image_path = self.IMAGE_PATH
        self.toolbar_image_path = os.path.join(image_path, "toolbar")

        self.AddLabelTool(wx.ID_OPEN, "Open File", self.bmp("open"), shortHelp="Open File")
        self.AddLabelTool(wx.ID_SAVE, "Save File", self.bmp("save"), shortHelp="Save File")

        self.AddSeparator()

        self.AddLabelTool(wx.ID_UNDO, "Undo", self.bmp("undo"), shortHelp="Undo")
        self.AddLabelTool(wx.ID_REDO, "Redo", self.bmp("redo"), shortHelp="Redo")
        self.AddLabelTool(wx.ID_CLEAR, "Clear Selection", self.bmp("clear_selection"), shortHelp="Clear Selection")
        self.AddLabelTool(wx.ID_DELETE, "Delete Selection", self.bmp("delete_selection"), shortHelp="Delete Selection")

        self.AddSeparator()

        self.AddLabelTool(wx.ID_NEW, "New Layer", self.bmp("add_layer"), shortHelp="New Layer", )
        self.AddLabelTool(wx.ID_ADD, "New Folder", self.bmp("add_folder"), shortHelp="New Folder", )

        self.AddLabelTool(wx.ID_UP, "Raise Layer", self.bmp("raise"), shortHelp="Raise Layer")
        self.AddLabelTool(wx.ID_DOWN, "Lower Layer", self.bmp("lower"), shortHelp="Lower Layer")

        self.triangulate_id = wx.NewId()
        self.AddLabelTool(self.triangulate_id, "Triangulate Layer", self.bmp("triangulate"), shortHelp="Triangulate Layer", )

        self.contour_id = wx.NewId()
        self.AddLabelTool(self.contour_id, "Contour Layer", self.bmp("contour"), shortHelp="Contour Layer")

        self.AddLabelTool(wx.ID_REMOVE, "Delete Layer", self.bmp("delete_layer"), shortHelp="Delete Layer")

        self.AddSeparator()

        self.AddLabelTool(wx.ID_ZOOM_FIT, "Zoom to Fit", self.bmp("zoom_fit"), shortHelp="Zoom to Fit Visible Layers")
        self.AddLabelTool(wx.ID_ZOOM_IN, "Zoom In", self.bmp("zoom_in"), shortHelp="Zoom In")
        self.AddLabelTool(wx.ID_ZOOM_OUT, "Zoom Out", self.bmp("zoom_out"), shortHelp="Zoom Out")

        self.AddSeparator()

        self.AddRadioLabelTool(wx.ID_ZOOM_100, "Zoom to Box", self.bmp("zoom_box"), shortHelp="Zoom to Box")

        self.pan_id = wx.NewId()
        self.AddRadioLabelTool(self.pan_id, "Pan", self.bmp("pan"), shortHelp="Pan (Alt)")

        self.add_points_id = wx.NewId()
        self.AddRadioLabelTool(self.add_points_id, "Edit Points", self.bmp("add_points"), shortHelp="Edit Points")

        self.add_lines_id = wx.NewId()
        self.AddRadioLabelTool(self.add_lines_id, "Edit Lines", self.bmp("add_lines"), shortHelp="Edit Lines")

        # self.selection_updated()

        f = self.frame
        # f.Bind( wx.EVT_TOOL, ui.Wx_handler( self.inbox, "triangulate" ), id = self.triangulate_id )

        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_triangulate(event), id=self.triangulate_id)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_set_pan_mode(event), id=self.pan_id)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_set_add_points_mode(event), id=self.add_points_id)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_set_add_lines_mode(event), id=self.add_lines_id)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_add_layer(event), id=wx.ID_NEW)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_add_folder(event), id=wx.ID_ADD)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_undo(event), id=wx.ID_UNDO)
        f.Bind(wx.EVT_TOOL, lambda event: self.controller.menu_bar.do_redo(event), id=wx.ID_REDO)
        # f.Bind( wx.EVT_TOOL, ui.Wx_handler( self.inbox, "contour_layer" ), id = self.contour_id )

        # make the pan tool selected by default
        self.ToggleTool(self.pan_id, True)

    def enable_disable_tools(self):
        raisable = self.controller.layer_tree_control.is_selected_layer_raisable()
        self.EnableTool(wx.ID_UP, raisable)
        lowerable = self.controller.layer_tree_control.is_selected_layer_lowerable()
        self.EnableTool(wx.ID_DOWN, lowerable)
        self.updated_undo_redo()

    def updated_undo_redo(self):
        u = app_globals.editor.get_current_undoable_operation_text()
        r = app_globals.editor.get_current_redoable_operation_text()

        if (u == ""):
            self.SetToolShortHelp(wx.ID_UNDO, "Undo")
            self.EnableTool(wx.ID_UNDO, False)
        else:
            self.SetToolShortHelp(wx.ID_UNDO, "Undo {0}".format(u))
            self.EnableTool(wx.ID_UNDO, True)

        if (r == ""):
            self.SetToolShortHelp(wx.ID_REDO, "Redo")
            self.EnableTool(wx.ID_REDO, False)
        else:
            self.SetToolShortHelp(wx.ID_REDO, "Redo {0}".format(r))
            self.EnableTool(wx.ID_REDO, True)

    """
            if request == "undo_updated":
                self.undo_updated( **message )
            elif request == "selection_updated":
                self.selection_updated( **message )
            elif request == "add_layer":
                # Position the popup menu beneath the toolbar near where the
                # click occurred.
                position = (
                    self.ScreenToClient( wx.GetMousePosition() )[ 0 ] -
                        self.GetToolSize()[ 0 ] / 2,
                    0 if sys.platform == "darwin" else self.GetToolSize()[ 1 ],
                )

                self.PopupMenu(
                    Add_layer_menu(
                        self.image_path, self.root_layer, self.command_stack,
                    ),
                    position,
                ),
            elif request == "triangulate":
                self.triangulate( scheduler )
            elif request == "contour_layer":
                self.contour( scheduler )
            elif request == "pan_mode":
                self.ToggleTool( self.pan_id, True )
            elif request == "add_points_mode":
                self.ToggleTool( self.add_points_id, True )
            elif request == "add_lines_mode":
                self.ToggleTool( self.add_lines_id, True )
            elif request == "start_zoom_box":
                self.ToggleTool( wx.ID_ZOOM_100, True )
            elif request == "end_zoom_box":
                self.ToggleTool( wx.ID_ZOOM_100, False )

    def triangulate( self, scheduler ):
        if self.menu_bar.triangle_maker is None:
            self.menu_bar.triangle_maker = ui.Triangle_maker(
                self.frame,
                self.root_layer,
                self.renderer.transformer,
                self.command_stack
            )
            scheduler.add( self.menu_bar.triangle_maker.run )
        else:
            self.menu_bar.triangle_maker.inbox.send( request = "show" )

    def contour( self, scheduler ):
        if self.menu_bar.contour_maker is None:
            self.menu_bar.contour_maker = ui.Contour_maker(
                self.frame,
                self.root_layer,
                self.command_stack
            )
            scheduler.add( self.menu_bar.contour_maker.run )
        else:
            self.menu_bar.contour_maker.inbox.send( request = "show" )

    def selection_updated( self, selections = None, raisable = False,
                           lowerable = False, deletable = False,
                           layer_deletable = None ):
        if raisable is not None:
            self.EnableTool( wx.ID_UP, raisable )
        if lowerable is not None:
            self.EnableTool( wx.ID_DOWN, lowerable )

        self.EnableTool( wx.ID_CLEAR, deletable )
        self.EnableTool( wx.ID_DELETE, deletable )

        if layer_deletable is not None:
            self.EnableTool( wx.ID_SAVE, layer_deletable )
            self.EnableTool( wx.ID_REMOVE, layer_deletable )
    """

    def bmp(self, base_name):
        return wx.Bitmap(os.path.join(self.toolbar_image_path, base_name + ".png"))
