import maproomlib.utility as utility
from maproomlib.plugin.Selection_layer import Selection_layer


class Flag_layer( Selection_layer ):
    """
    A layer for flagging objects, such as points and lines. This is distinct
    from the standard selection layer so that objects can remain flagged
    regardless of what the user does with the current selection.
    """
    PLUGIN_TYPE = "layer"
    FLAG_COLOR = utility.color_to_int( 1.0, 0.9, 0, 1 )

    def __init__( self, command_stack, plugin_loader ):
        # No parent means it can't subscribe to standard selection messages,
        # and so it can only be manipulated directly.
        Selection_layer.__init__(
            self, command_stack, plugin_loader, parent = None,
            color = self.FLAG_COLOR,
        )

    def replace_selection( self, layer, object_indices, record_undo = False ):
        if ( self.parent and layer not in self.parent.children ) or \
           not hasattr( layer, "inbox" ):
            self.clear_selection( record_undo = record_undo )
            return

        layer.inbox.send(
            request = "make_flag",
            object_indices = object_indices,
            color = self.color,
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "flag" )
        flag = message.get( "layer" )
        if flag is None:
            return

        # Necessary to prevent subscription leaks, since each flag layer
        # subscribes itself to its points layer.
        for child in self.children:
            child.inbox.send( request = "close" )

        self.replace_layers(
            flag,
            description = "Set %s" % flag.name,
            record_undo = record_undo,
        )

    def clear_selection( self, record_undo = False ):
        if len( self.children ) == 0:
            return

        for child in self.children:
            child.inbox.send( request = "close" )

        self.replace_layers(
            layer = None,
            description = "Clear Selection",
            record_undo = record_undo,
        )

    def selection_updated( self ):
        # Since this is a flag layer, manipulating it shouldn't alter what's
        # selected. So prevent this layer from issuing a selection_updated
        # message. Instead, it should send a flags_updated message.
        flags = []
        flag_count = 0

        for child in self.children:
            child.inbox.send(
                request = "get_indices",
                response_box = self.inbox,
            )

            message = self.inbox.receive(
                request = "indices",
            )

            indices = message.get( "indices" )
            flag_count += len( indices )
            flags.append( ( child, indices ) )

        ( origin, size, projection ) = self.get_dimensions()

        self.outbox.send(
            request = "flags_updated",
            flags = flags,
            flag_count = flag_count,
            origin = origin,
            size = size,
            projection = projection,
        )
