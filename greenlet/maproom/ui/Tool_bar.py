# coding=utf8
import wx
import os
import sys
import maproomlib.ui as ui
from Add_layer_menu import Add_layer_menu


class Tool_bar( wx.ToolBar ):
    """
    Main application menu bar.
    """
    IMAGE_PATH = "ui/images"

    def __init__( self, frame, renderer, root_layer, layer_tree,
                  command_stack, menu_bar ):
        wx.ToolBar.__init__( self, frame, style = wx.TB_FLAT )
        self.frame = frame
        self.renderer = renderer
        self.root_layer = root_layer
        self.layer_tree = layer_tree
        self.command_stack = command_stack
        self.menu_bar = menu_bar
        self.inbox = ui.Wx_inbox()
        self.update_time = None

        image_path = self.IMAGE_PATH
        if os.path.basename( os.getcwd() ) != "maproom":
            image_path = os.path.join( "maproom", image_path )

        self.image_path = image_path
        toolbar_image_path = os.path.join( image_path, "toolbar" )
        self.toolbar_image_path = toolbar_image_path

        self.AddLabelTool(
            wx.ID_OPEN, "Open File",
            wx.Bitmap( os.path.join( toolbar_image_path, "open.png" ) ),
            shortHelp = "Open File",
        )

        self.AddLabelTool(
            wx.ID_SAVE, "Save File",
            wx.Bitmap( os.path.join( toolbar_image_path, "save.png" ) ),
            shortHelp = "Save File",
        )
        self.EnableTool( wx.ID_SAVE, False )

        self.AddSeparator()

        self.AddLabelTool(
            wx.ID_UNDO, "Undo",
            wx.Bitmap( os.path.join( toolbar_image_path, "undo.png" ) ),
            shortHelp = "Undo",
        )
        self.EnableTool( wx.ID_UNDO, False )
        self.AddLabelTool(
            wx.ID_REDO, "Redo",
            wx.Bitmap( os.path.join( toolbar_image_path, "redo.png" ) ),
            shortHelp = "Redo",
        )
        self.AddLabelTool(
            wx.ID_CLEAR, "Clear Selection",
            wx.Bitmap( os.path.join( toolbar_image_path, "clear_selection.png" ) ),
            shortHelp = "Clear Selection",
        )
        self.AddLabelTool(
            wx.ID_DELETE, "Delete Selection",
            wx.Bitmap( os.path.join( toolbar_image_path, "delete_selection.png" ) ),
            shortHelp = "Delete Selection",
        )

        self.AddSeparator()

        add_layer_bitmap = \
            wx.Bitmap( os.path.join( toolbar_image_path, "add_layer.png" ) )

        self.AddLabelTool(
            wx.ID_NEW, "Add Layer",
            add_layer_bitmap,
            shortHelp = "Add Layer",
        )

        self.AddLabelTool(
            wx.ID_UP, "Raise Layer",
            wx.Bitmap( os.path.join( toolbar_image_path, "raise.png" ) ),
            shortHelp = "Raise Layer",
        )
        self.AddLabelTool(
            wx.ID_DOWN, "Lower Layer",
            wx.Bitmap( os.path.join( toolbar_image_path, "lower.png" ) ),
            shortHelp = "Lower Layer",
        )
        self.EnableTool( wx.ID_REDO, False )

        self.triangulate_id = wx.NewId()
        self.AddLabelTool(
            self.triangulate_id, "Triangulate Layer",
            wx.Bitmap( os.path.join( toolbar_image_path, "triangulate.png" ) ),
            shortHelp = "Triangulate Layer",
        )
        try:
            import pytriangle
        except ImportError:
            self.EnableTool( self.triangulate_id, False )

        self.contour_id = wx.NewId()
        self.AddLabelTool(
            self.contour_id, "Contour Layer",
            wx.Bitmap( os.path.join( toolbar_image_path, "contour.png" ) ),
            shortHelp = "Contour Layer",
        )

        self.AddLabelTool(
            wx.ID_REMOVE, "Delete Layer",
            wx.Bitmap( os.path.join( toolbar_image_path, "delete_layer.png" ) ),
            shortHelp = "Delete Layer",
        )
        self.EnableTool( wx.ID_REMOVE, False )

        self.AddSeparator()

        self.AddLabelTool(
            wx.ID_ZOOM_FIT, "Zoom to Fit",
            wx.Bitmap( os.path.join( toolbar_image_path, "zoom_fit.png" ) ),
            shortHelp = "Zoom to Fit",
        )

        self.AddLabelTool(
            wx.ID_ZOOM_IN, "Zoom In",
            wx.Bitmap( os.path.join( toolbar_image_path, "zoom_in.png" ) ),
            shortHelp = "Zoom In",
        )

        self.AddLabelTool(
            wx.ID_ZOOM_OUT, "Zoom Out",
            wx.Bitmap( os.path.join( toolbar_image_path, "zoom_out.png" ) ),
            shortHelp = "Zoom Out",
        )

        self.AddCheckLabelTool(
            wx.ID_ZOOM_100, "Zoom to Box",
            wx.Bitmap( os.path.join( toolbar_image_path, "zoom_box.png" ) ),
            shortHelp = "Zoom to Box",
        )

        self.AddSeparator()

        self.pan_id = wx.NewId()
        self.AddRadioLabelTool(
            self.pan_id, "Pan",
            wx.Bitmap( os.path.join( toolbar_image_path, "pan.png" ) ),
            shortHelp = "Pan (Alt)",
        )

        self.add_points_id = wx.NewId()
        self.AddRadioLabelTool(
            self.add_points_id, "Edit Points",
            wx.Bitmap( os.path.join( toolbar_image_path, "add_points.png" ) ),
            shortHelp = "Edit Points",
        )

        self.add_lines_id = wx.NewId()
        self.AddRadioLabelTool(
            self.add_lines_id, "Edit Lines",
            wx.Bitmap( os.path.join( toolbar_image_path, "add_lines.png" ) ),
            shortHelp = "Edit Lines",
        )

        self.selection_updated()
        self.Realize()

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
                "start_zoom_box",
                "end_zoom_box",
            ),
        )

        self.frame.Bind(
            wx.EVT_TOOL,
            ui.Wx_handler( self.inbox, "triangulate" ),
            id = self.triangulate_id,
        )
        self.frame.Bind(
            wx.EVT_TOOL,
            lambda event: \
                self.root_layer.inbox.send( request = "pan_mode" ),
            id = self.pan_id,
        )
        self.frame.Bind(
            wx.EVT_TOOL,
            lambda event: \
                self.root_layer.inbox.send( request = "add_points_mode" ),
            id = self.add_points_id,
        )
        self.frame.Bind(
            wx.EVT_TOOL,
            lambda event: \
                self.root_layer.inbox.send( request = "add_lines_mode" ),
            id = self.add_lines_id,
        )
        self.frame.Bind(
            wx.EVT_TOOL,
            ui.Wx_handler( self.inbox, "add_layer" ),
            id = wx.ID_NEW,
        )
        self.frame.Bind(
            wx.EVT_TOOL,
            ui.Wx_handler( self.inbox, "contour_layer" ),
            id = self.contour_id,
        )

    def run( self, scheduler ):
        while True:
            message = self.inbox.receive(
                request = (
                    "undo_updated",
                    "selection_updated",
                    "add_layer",
                    "triangulate",
                    "contour_layer",
                    "pan_mode", "add_points_mode", "add_lines_mode",
                    "start_zoom_box", "end_zoom_box",
                ),
            )
            request = message.pop( "request" )
            
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
                    0 if sys.platform == "darwin" else \
                        self.GetToolSize()[ 1 ],
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

    def undo_updated( self, next_undo_description, next_redo_description ):
        if next_undo_description is None:
            self.SetToolShortHelp( wx.ID_UNDO, "Undo" )
            self.EnableTool( wx.ID_UNDO, False )
        else:
            self.SetToolShortHelp( wx.ID_UNDO, "Undo %s" %
                                   next_undo_description )
            self.EnableTool( wx.ID_UNDO, True )

        if next_redo_description is None:
            self.SetToolShortHelp( wx.ID_REDO, "Redo" )
            self.EnableTool( wx.ID_REDO, False )
        else:
            self.SetToolShortHelp( wx.ID_REDO, "Redo %s" %
                                   next_redo_description )
            self.EnableTool( wx.ID_REDO, True )

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
