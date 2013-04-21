import wx
import logging
import functools
import maproomlib.ui
import maproomlib.utility as utility


class Properties_panel( wx.Panel ):
    """
    A panel for displaying and manipulating the properties of a layer.
    """
    LABEL_SPACING = 2
    VALUE_SPACING = 10
    SIDE_SPACING = 5
    SET_PROPERTY_TIMEOUT = 1.0

    def __init__( self, parent, root_layer, layer_tree, frame,
                  command_stack ):
        wx.Panel.__init__( self, parent )
        self.root_layer = root_layer
        self.layer_tree = layer_tree
        self.frame = frame
        self.command_stack = command_stack
        self.inbox = maproomlib.ui.Wx_inbox()
        self.outbox = utility.Outbox()
        self.subscribed_outbox = None
        self.fields = {} # property -> value field control
        self.logger = logging.getLogger( __name__ )
        self.start_command = False
        self.error_field = None

        static_box = wx.StaticBox( self, label = "Properties" )
        self.sizer = wx.StaticBoxSizer( static_box, wx.VERTICAL )
        self.SetSizer( self.sizer )

        self.sizer.Add( wx.Panel( self ) )

    def run( self, scheduler ):
        self.root_layer.outbox.subscribe(
            self.inbox,
            request = (
                "selection_updated",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "property_updated",
                    "set_property",
                    "selection_updated",
                ),
            )
            request = message.pop( "request" )

            if request == "selection_updated":
                selections = message.get( "selections" )
                self.update_panel( selections )
            elif request in ( "clear_selection", "delete_selection" ):
                self.update_panel( selections = None )
            elif request == "property_updated":
                self.property_updated( **message )
            elif request == "set_property":
                self.set_property( **message )

    def update_panel( self, selections = None ):
        if selections is None:
            selections = ()

        self.Freeze()
        self.sizer.Clear( True )
        first_text_ctrl = None

        for ( layer, indices ) in selections:
            if not hasattr( layer, "inbox" ):
                properties = ()
            elif hasattr( layer, "wrapped_layer" ) and \
                 layer.wrapped_layer == self.root_layer:
                properties = ()
            else:
                layer.inbox.send(
                    request = "get_properties",
                    indices = indices or None,
                    response_box = self.inbox,
                )

                message = self.inbox.receive( request = "properties" )
                properties = message.get( "properties" )

            if self.subscribed_outbox:
                self.subscribed_outbox.unsubscribe( self.inbox )
                self.subscribed_outbox = None

            if hasattr( layer, "outbox" ):
                layer.outbox.subscribe(
                    self.inbox,
                    request = "property_updated",
                )
                self.subscribed_outbox = layer.outbox

            bold_font = self.GetFont()
            bold_font.SetWeight( weight = wx.FONTWEIGHT_BOLD )
            
            # If the value field has an explicit save method, use that, since we
            # can't rely on the focus kill handler to fire before the field is
            # destroyed.
            for value_field in self.fields.values():
                if hasattr( value_field, "save" ):
                    value_field.save()

            self.fields = {}

            self.sizer.AddSpacer( self.LABEL_SPACING )

            for property in properties:
                label = wx.StaticText( self, label = property.name )
                label.SetFont( bold_font )
                self.sizer.Add( label, 0, wx.LEFT | wx.RIGHT, self.SIDE_SPACING )
                self.sizer.AddSpacer( self.LABEL_SPACING )
                expand = 0

                if property.mutable:
                    if property.choices:
                        value_field = wx.Choice(
                            self,
                            choices = property.choices,
                        )
                        value_field.SetSelection(
                            property.choices.index( property.value )
                        )
                        value_field.Bind(
                            wx.EVT_CHOICE,
                            functools.partial(
                                self.choice_changed,
                                layer = layer,
                                property = property,
                                value_field = value_field,
                            )
                        )
                        value_field.Bind(
                            wx.EVT_SET_FOCUS,
                            self.value_focused,
                        )
                    else:
                        value_field = wx.TextCtrl(
                            self,
                            value = str( property ),
                        )
                        if property.type is None:
                            expand = wx.EXPAND

                        if first_text_ctrl is None:
                            first_text_ctrl = value_field

                        value_field.Bind(
                            wx.EVT_TEXT,
                            functools.partial(
                                self.text_changed,
                                layer = layer,
                                property = property,
                                value_field = value_field,
                            )
                        )
                        value_field.Bind(
                            wx.EVT_SET_FOCUS,
                            self.value_focused,
                        )
                        value_field.save = functools.partial(
                            self.text_changed,
                            event = None,
                            layer = layer,
                            property = property,
                            value_field = value_field,
                        )
                else:
                    value_field = wx.StaticText( self, label = str( property ) )

                self.fields[ property ] = value_field
                self.sizer.Add(
                    value_field, 0, expand | wx.LEFT | wx.RIGHT,
                    self.SIDE_SPACING,
                )
                self.sizer.AddSpacer( self.VALUE_SPACING )

        self.sizer.Layout()
        self.sizer.FitInside( self )

        if first_text_ctrl is not None:
            first_text_ctrl.SetSelection( -1, -1 )
            first_text_ctrl.SetFocus()

        self.Thaw()
        self.Update()
        self.Refresh()

    def set_property( self, layer, property, value ):
        if self.start_command is True:
            self.start_command = False
            self.command_stack.inbox.send(
                request = "start_command",
            )

        layer.inbox.send(
            request = "set_property",
            property = property,
            value = value,
            response_box = self.inbox,
        )

        try:
            message = self.inbox.receive(
                request = "property_updated",
                layer = layer.wrapped_layer if \
                        hasattr( layer, "wrapped_layer" ) else layer,
                timeout = self.SET_PROPERTY_TIMEOUT,
            )
        except ValueError, error:
            if self.error_field: self.error_field.Destroy()

            self.Freeze()
            error_font = self.GetFont()
            error_font.SetWeight( weight = wx.FONTWEIGHT_BOLD )
            self.error_field = wx.StaticText( self, label = str( error ) )
            self.error_field.SetFont( error_font )
            self.error_field.SetForegroundColour( wx.RED )

            self.sizer.Insert(
                0, self.error_field, 0, wx.EXPAND | wx.LEFT | wx.RIGHT,
                self.SIDE_SPACING,
            )
            self.sizer.Layout()
            self.Thaw()
            self.Refresh()
            return
        except utility.Timeout_error:
            if self.error_field:
                self.Freeze()
                self.error_field.Destroy()
                self.error_field = None
                self.sizer.Layout()
                self.Thaw()
                self.Refresh()

            # If we're still subscribed to the layer containing the property,
            # then log a timeout error.
            if self.subscribed_outbox == layer.outbox:
                self.logger.error(
                    'A timeout occurred when setting property "%s".' % \
                    property.name,
                )
            return

        if self.error_field:
            self.Freeze()
            self.error_field.Destroy()
            self.error_field = None
            self.sizer.Layout()
            self.Thaw()
            self.Refresh()

        message.pop( "request" )
        self.property_updated( **message )

    def property_updated( self, layer, property ):
        value_field = self.fields.get( property )
        if value_field is None:
            return

        # Bail if the field is focused to avoid replacing text the user is
        # editing.
        if wx.Window.FindFocus() == value_field:
            return

        if hasattr( value_field, "ChangeValue" ):
            value_field.ChangeValue( str( property ) )
        elif hasattr( value_field, "SetStringSelection" ):
            value_field.SetStringSelection( str( property ) )

    def value_focused( self, event ):
        self.start_command = True

        event.Skip()

    def choice_changed( self, event, layer, property, value_field ):
        value = value_field.GetString( value_field.GetSelection() )

        # Send this as a message to ourself so that it's handled in the
        # context of our standard message handler, rather than from within
        # a wx event handler.
        self.inbox.send(
            request = "set_property",
            layer = layer,
            property = property,
            value = value,
        )

    def text_changed( self, event, layer, property, value_field ):
        value = value_field.GetValue()

        self.inbox.send(
            request = "set_property",
            layer = layer,
            property = property,
            value = value,
        )
