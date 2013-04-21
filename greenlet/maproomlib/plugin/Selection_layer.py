import maproomlib.utility as utility
from maproomlib.plugin.Composite_layer import Composite_layer


class Selection_layer( Composite_layer ):
    """
    A layer for representing and manipulating the current selection, e.g. the
    points and lines that the user has selected.
    """
    PLUGIN_TYPE = "layer"

    def __init__( self, command_stack, plugin_loader, parent, color = None,
                  depth_unit = None ):
        Composite_layer.__init__(
            self, command_stack, plugin_loader, parent, "Selection",
            child_subscribe_requests = [
                "points_added",
                "line_points_added",
                "points_deleted",
                "points_updated",
                "lines_added",
                "lines_deleted",
            ]
        )

        self.color = color
        self.depth_unit = depth_unit or "unknown"

    def run( self, scheduler ):
        # Subscribe to our own outbox, so we can issue selection messages in
        # response to certain layer manipulation messages.
        self.outbox.subscribe(
            self.inbox,
            request = (
                "layer_added",
                "layer_removed",
            ),
        )

        Composite_layer.run(
            self,
            scheduler,
            replace_selection = self.replace_selection,
            add_selection = self.add_selection,
            clear_selection = self.clear_selection,
            move_selection = self.move_selection,
            delete_selection = self.delete_selection,
            triangulate_selected = self.triangulate_selected,
            contour_selected = self.contour_selected,
            delete_layer = self.delete_layer,
            add_points_to_selected = self.add_points_to_selected,
            add_lines_to_selected = self.add_lines_to_selected,
            get_savers_for_selected = self.get_savers_for_selected,
            save_selected = self.save_selected,
            find_duplicates_in_selected = self.find_duplicates_in_selected,
            property_updated = self.property_updated,
            layer_added = lambda *args, **kwargs: self.selection_updated(),
            points_added = lambda *args, **kwargs: self.selection_updated(),
            line_points_added = lambda *args, **kwargs: self.selection_updated(),
            lines_added = lambda *args, **kwargs: self.selection_updated(),
            lines_deleted = lambda *args, **kwargs: self.selection_updated(),
            update_selection = lambda *args, **kwargs: self.selection_updated(),
        )

        if self.parent is not None:
            self.parent.outbox.unsubscribe( self.inbox )

    def replace_selection( self, layer, object_indices, record_undo = True ):
        if ( self.parent and layer not in self.parent.children ) or \
           not hasattr( layer, "inbox" ) or len( object_indices ) == 0:
            self.clear_selection( record_undo = record_undo )
            return

        layer.inbox.send(
            request = "make_selection",
            object_indices = object_indices,
            color = self.color,
            depth_unit = self.depth_unit,
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "selection" )
        selection = message.get( "layer" )
        if selection is None:
            self.clear_selection( record_undo = record_undo )
            return

        self.replace_layers(
            selection,
            description = "Set %s" % selection.name,
            record_undo = record_undo,
        )

    def add_selection( self, layer, object_indices, include_range = False,
                       record_undo = True ):
        if ( self.parent and layer not in self.parent.children ) or \
           not hasattr( layer, "inbox" ):
            return

        # If there is already at least one selection layer, then try to add to
        # extend each layer. Upon success, return.
        for child in self.children:
            child.inbox.send(
                request = "extend",
                layer = layer,
                indices = object_indices,
                include_range = include_range,
                record_undo = record_undo,
                response_box = self.inbox,
            )

            message = self.inbox.receive( request = "selected" )
            if message.get( "indices" ):
                return

        # Otherwise, fall back to making a new selection.
        layer.inbox.send(
            request = "make_selection",
            object_indices = object_indices,
            color = self.color,
            depth_unit = self.depth_unit,
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "selection" )
        selection = message.get( "layer" )
        if selection is None:
            return

        self.add_layer(
            selection,
            description = "Add %s" % selection.name,
            record_undo = record_undo,
        )

    def clear_selection( self, record_undo = True ):
        if len( self.children ) == 0:
            return

        self.replace_layers(
            layer = None,
            description = "Clear Selection",
            record_undo = record_undo,
        )

    def move_selection( self, movement_vector, cumulative = False ):
        for child in self.children:
            child.inbox.send(
                request = "move",
                movement_vector = movement_vector,
                cumulative = cumulative,
            )

    def delete_selection( self ):
        for child in list( self.children ):
            child.inbox.send(
                request = "delete",
            )

            self.remove_layer( child )

    def triangulate_selected( self, q, a, transformer, response_box ):
        pass

    def contour_selected( self, levels, neighbor_count, grid_steps,
                          buffer_factor, response_box ):
        pass

    def delete_layer( self ):
        # The standard Selection_layer ignores delete_layer messages, but they
        # can be overridden in derived classes.
        pass

    def add_points_to_selected( self, points, projection, to_layer = None,
                                to_index = None ):
        for child in self.children:
            child.inbox.send(
                request = "add_points",
                points = points,
                projection = projection,
                to_layer = to_layer,
                to_index = to_index,
            )

    def add_lines_to_selected( self, points, projection, to_layer = None,
                               to_index = None ):
        for child in self.children:
            child.inbox.send(
                request = "add_lines",
                points = points,
                projection = projection,
                to_layer = to_layer,
                to_index = to_index,
            )

    def get_savers_for_selected( self, response_box ):
        # Can be overridden in derived classes.
        pass

    def save_selected( self, response_box, filename = None, saver = None ):
        # Can be overridden in derived classes.
        pass

    def find_duplicates_in_selected( self, distance_tolerance,
                                     depth_tolerance, response_box ):
        # Can be overridden in derived classes.
        pass

    def property_updated( self, layer, property ):
        if property.name == "Depth unit":
            self.depth_unit = property.value

    def selection_updated( self ):
        selections = []
        total_indices = 0

        for child in self.children:
            child.inbox.send(
                request = "get_indices",
                response_box = self.inbox,
            )

            message = self.inbox.receive(
                request = "indices",
            )

            indices = message.get( "indices" )
            total_indices += len( indices )
            selections.append( ( child, indices ) )

        self.outbox.send(
            request = "selection_updated",
            selections = selections,
            raisable = None,
            lowerable = None,
            deletable = ( total_indices > 0 ),
            layer_deletable = None,
        )
