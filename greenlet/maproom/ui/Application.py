import os
import os.path
import wx
import maproomlib.ui as ui
from Menu_bar import Menu_bar
from Tool_bar import Tool_bar


class Application( ui.Scheduled_application ):
    """
    The UI for the Maproom application.
    """
    DEFAULT_FRAME_SIZE = ( 800, 600 )
    LEFT_SASH_POSITION = 200
    TOP_SASH_POSITION = 250
    ICON_FILENAME = "ui/images/maproom.ico"

    def __init__( self, root_layer, command_stack, plugin_loader, scheduler,
                  version, init_filenames ):
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.plugin_loader = plugin_loader
        self.scheduler = scheduler
        self.version = version
        self.init_filenames = init_filenames
        self.inbox = ui.Wx_inbox()

        ui.Scheduled_application.__init__( self, False )

    def MainLoop( self, scheduler ):
        icon_filename = self.ICON_FILENAME
        if os.path.basename( os.getcwd() ) != "maproom":
            icon_filename = os.path.join( "maproom", icon_filename )

        self.frame = wx.Frame( None, wx.ID_ANY, "Maproom" )
        self.frame.SetIcon( wx.Icon( icon_filename, wx.BITMAP_TYPE_ICO ) )
        self.log_viewer = ui.Log_viewer( self.frame )

        self.renderer_splitter = wx.SplitterWindow(
            self.frame,
            wx.ID_ANY,
            style = wx.SP_3DSASH,
        )

        self.properties_splitter = wx.SplitterWindow(
            self.renderer_splitter,
            wx.ID_ANY,
            style = wx.SP_3DSASH,
        )

        self.layer_tree = ui.Layer_tree(
            self.properties_splitter,
            self.root_layer,
            self.command_stack,
        )

        self.properties_panel = ui.Properties_panel(
            self.properties_splitter,
            self.root_layer,
            self.layer_tree,
            self.frame,
            self.command_stack,
        )

        self.plugin_loader.inbox.send(
            request = "get_plugin",
            plugin_type = "renderer",
            plugin_name = "Opengl_renderer",
            response_box = self.inbox,
            # Request the class, not an instance. That way we can instantiate
            # it ourself.
            skip_call = True,
        )

        message = self.inbox.receive( request = "plugin" )
        self.renderer = message.get( "plugin" )(
            parent = self.renderer_splitter,
            root_layer = self.root_layer,
            command_stack = self.command_stack,
        )
        scheduler.add( self.renderer.run )

        self.menu_bar = Menu_bar(
            self.frame, self.renderer, self.root_layer, self.layer_tree,
            self.command_stack, self.renderer.viewport, self.log_viewer,
            self.version,
        )
        self.frame.SetMenuBar( self.menu_bar )

        self.tool_bar = Tool_bar(
            self.frame,
            self.renderer,
            self.root_layer,
            self.layer_tree,
            self.command_stack,
            self.menu_bar,
        )
        self.frame.SetToolBar( self.tool_bar )

        self.status_bar = self.frame.CreateStatusBar()
        self.progress_tracker = ui.Progress_tracker(
            self.status_bar,
            self.root_layer,
            self.renderer,
        )

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

        self.renderer_splitter.SplitVertically(
            self.properties_splitter,
            self.renderer,
            self.LEFT_SASH_POSITION,
        )

        self.properties_splitter.SplitHorizontally(
            self.layer_tree,
            self.properties_panel,
            self.TOP_SASH_POSITION,
        )

        self.SetTopWindow( self.frame )
        self.frame.SetSize( self.DEFAULT_FRAME_SIZE )
        self.frame.Show( True )

        self.frame.Bind( wx.EVT_CLOSE, self.close )
        self.frame.Bind( wx.EVT_MENU, self.close, id = wx.ID_EXIT )

        scheduler.add( self.menu_bar.run )
        scheduler.add( self.tool_bar.run )
        scheduler.add( self.layer_tree.run )
        scheduler.add( self.properties_panel.run )
        scheduler.add( self.progress_tracker.run )
        scheduler.add( self.log_viewer.run )

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

        ui.Scheduled_application.MainLoop( self, scheduler )

    def MacReopenApp( self ):
        """
        Invoked by wx when the Maproom dock icon is clicked on Mac OS X.
        """
        self.GetTopWindow().Raise()

    def close( self, event ):
        self.scheduler.shutdown()
        self.renderer.shutdown()
        self.shutdown()
        self.frame.Destroy()
