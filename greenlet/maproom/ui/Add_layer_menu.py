import wx
import os
import maproomlib.utility as utility


class Add_layer_menu( wx.Menu ):
    def __init__( self, image_path, root_layer, command_stack, frame = None ):
        import maproomlib.ui as ui

        wx.Menu.__init__( self )
        self.root_layer = root_layer
        self.command_stack = command_stack
        self.inbox = ui.Wx_inbox()

        self.add_verdat_id = wx.NewId()
        self.add_verdat_item = wx.MenuItem(
            self,
            self.add_verdat_id,
            "Verdat (points and lines)",
        )
        self.add_verdat_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "add_layer_verdat.png" ) ),
        )
        self.AppendItem( self.add_verdat_item )

        self.add_bna_id = wx.NewId()
        self.add_bna_item = wx.MenuItem(
            self,
            self.add_bna_id,
            "BNA (polygons)",
        )
        self.add_bna_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "add_layer_bna.png" ) ),
        )
        self.AppendItem( self.add_bna_item )

        self.add_gps_id = wx.NewId()
        self.add_gps_item = wx.MenuItem(
            self,
            self.add_gps_id,
            "GPS (positions from GPS device)",
        )
        self.add_gps_item.SetBitmap(
            wx.Bitmap( os.path.join( image_path, "add_layer_verdat.png" ) ),
        )
        self.AppendItem( self.add_gps_item )

        try:
            import garmin
        except ImportError:
            self.add_gps_item.Enable( False )

        bind_to = self if frame is None else frame

        bind_to.Bind(
            wx.EVT_MENU,
            self.add_verdat_layer,
            id = self.add_verdat_id,
        )
        bind_to.Bind(
            wx.EVT_MENU,
            self.add_bna_layer,
            id = self.add_bna_id,
        )
        bind_to.Bind(
            wx.EVT_MENU,
            self.add_gps_layer,
            id = self.add_gps_id,
        )

    def add_verdat_layer( self, event ):
        self.command_stack.inbox.send( request = "start_command" )
        self.root_layer.inbox.send(
            request = "create_layer",
            plugin_name = "Line_point_layer",
        )

    def add_bna_layer( self, event ):
        self.command_stack.inbox.send( request = "start_command" )
        self.root_layer.inbox.send(
            request = "create_layer",
            plugin_name = "Polygon_point_layer",
        )

    def add_gps_layer( self, event ):
        # TODO: If a GPS layer already exits, refuse to make another one.

        import serial
        scheduler = utility.Scheduler.current()

        # TODO: In addition to creating a Verdat layer, connect it to the
        # incoming GPS point data, and perhaps jump to the initial point.
        try:
            gps_reader = utility.Gps_reader()
        except serial.serialutil.SerialException:
            # TODO: Display an error message to the user.
            return

        self.root_layer.outbox.subscribe(
            self.inbox,
            request = "layer_added",
        )

        # Add a layer for the GPS path.
        layer_name = "GPS path"
        self.command_stack.inbox.send( request = "start_command" )
        self.root_layer.inbox.send(
            request = "create_layer",
            plugin_name = "Line_point_layer",
            name = layer_name,
        )


        # Wait for a message about that layer being added.
        layer = None
        while True:
            message = self.inbox.receive( request = "layer_added" )
            layer = message.get( "layer" )

            if layer and str( layer.name ) == layer_name:
                break

        self.root_layer.outbox.unsubscribe( self.inbox )

        # Let the GPS reader know about the layer.
        scheduler.add( gps_reader.run )
        gps_reader.inbox.send(
            request = "target_layer",
            target_layer = layer,
        )
