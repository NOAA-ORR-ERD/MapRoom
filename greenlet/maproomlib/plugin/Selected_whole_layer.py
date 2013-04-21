import maproomlib.utility as utility


class Selected_whole_layer:
    """
    A selected layer in its entirety, as opposed to just some points or lines
    selected within a layer.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "composite"

    def __init__( self, command_stack, name, wrapped_layer ):
        self.command_stack = command_stack
        self.name = name
        self.wrapped_layer = wrapped_layer
        self.projection = wrapped_layer.projection
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()

    def run( self, scheduler ):
        self.wrapped_layer.outbox.subscribe(
            self.inbox,
            request = (
                "property_updated",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "delete",
                    "undelete",
                    "add_points",
                    "add_lines",
                    "find_duplicates",
                    "triangulate",
                    "contour",
                    "save",
                    "set_property",
                    "get_properties",
                    "property_updated",
                    "cleaned_up_undo",
                    "close",
                ),
            )
            request = message.get( "request" )

            if request == "property_updated":
                self.outbox.send( **message )
            elif request == "cleaned_up_undo":
                if len( self.outbox.inboxes.values() ) > 1:
                    continue
                self.wrapped_layer.outbox.unsubscribe( self.inbox )
                return
            elif request == "close":
                self.wrapped_layer.outbox.unsubscribe( self.inbox )
                return
            else:
                self.wrapped_layer.inbox.send( **message )

    origin = property( lambda self: self.wrapped_layer.origin )
    size = property( lambda self: self.wrapped_layer.size )
    supported_savers = \
        property( lambda self: self.wrapped_layer.supported_savers )
