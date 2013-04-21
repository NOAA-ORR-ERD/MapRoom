from maproomlib.plugin.Selection_layer import Selection_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer


class Layer_selection_layer( Selection_layer ):
    """
    This variant of :class:`Selection_layer` is intended for selecting whole
    layers rather than specific objects within layers. It is only interested
    in itself and its own siblings. Rather than playing nicely with other
    selection layers, it maintains its own sense of what's selected and
    ignores selection-related messages that apply to layers it doesn't care
    about.

    A :class:`Layer_selection_layer` is useful for maintaining a list
    of always-selected root layers, for instance, without other selections
    affecting them.
    """
    PLUGIN_TYPE = "layer"

    # Layer types that don't show up in the list of layers, but are rendered
    # on the map.
    GHOST_LAYER_TYPES = set( (
        "Layer_selection_layer",
        "Selection_layer",
        "Flag_layer",
        "Polygon_set_layer",
        "Tile_set_layer",
        "Triangle_set_layer",
        "Selected_whole_layer",
        "Selected_point_set_layer",
        "Selected_line_set_layer",
    ) )

    def replace_selection( self, layer, object_indices, raisable = None,
                           lowerable = None, deletable = None,
                           layer_deletable = None, include_range = False,
                           record_undo = True ):
        # A Layer_selection_layer is only interested in selecting whole
        # layers, not individual objects within a layer.
        if object_indices:
            return

        if layer is None or not hasattr( layer, "inbox" ):
            return
        if self.ghost_layer( layer ):
            return

        layer.inbox.send(
            request = "make_selection",
            object_indices = object_indices,
            color = None,
            depth_unit = None,
            response_box = self.inbox,
        )

        message = self.inbox.receive( request = "selection" )
        selection = message.get( "layer" )

        self.replace_layers(
            selection,
            description = "Set %s" % \
                ( selection.name if selection else "Selection" ),
            record_undo = record_undo,
        )

    def add_selection( self, *args, **kwargs ):
        # When an add selection request comes in, replace the current
        # selection instead.
        self.replace_selection( *args, **kwargs )

    def clear_selection( self ):
        pass

    def move_selection( self, movement_vector, cumulative = False ):
        pass

    def delete_selection( self ):
        # When a standard selection deletion request comes in, ignore it.
        pass

    def triangulate_selected( self, q, a, transformer, response_box ):
        for child in self.children:
            child.inbox.send(
                request = "triangulate",
                q = q,
                a = a,
                transformer = transformer,
                response_box = response_box,
            )

    def contour_selected( self, levels, neighbor_count, grid_steps,
                          buffer_factor, response_box ):
        for child in self.children:
            child.inbox.send(
                request = "contour",
                levels = levels,
                neighbor_count = neighbor_count,
                grid_steps = grid_steps,
                buffer_factor = buffer_factor,
                response_box = response_box,
            )

    def delete_layer( self ):
        # When a special layer deletion request comes in, delete the layer.
        Selection_layer.delete_selection( self )

    def get_savers_for_selected( self, response_box ):
        for child in self.children:
            if hasattr( child, "supported_savers" ) and \
               len( child.supported_savers ) > 0:
                response_box.send(
                    request = "savers",
                    savers = child.supported_savers,
                )
                return

        response_box.send(
            exception = NotImplementedError(
                "Saving that layer type is not yet implemented.",
            ),
        )

    def save_selected( self, response_box, filename = None, saver = None ):
        for child in self.children:
            child.inbox.send(
                request = "save",
                filename = filename,
                saver = saver,
                response_box = response_box,
            )

    def find_duplicates_in_selected( self, distance_tolerance,
                                     depth_tolerance, response_box ):
        for child in self.children:
            child.inbox.send(
                request = "find_duplicates",
                distance_tolerance = distance_tolerance,
                depth_tolerance = depth_tolerance,
                response_box = response_box,
            )

    def selection_updated( self ):
        selections = []
        total_indices = 0

        if len( self.children ) == 0:
            raisable = False
            lowerable = False
            layer_deletable = False
        else:
            raisable = True
            lowerable = True
            layer_deletable = True

        for child in self.children:
            indices = ()
            selections.append( ( child, indices ) )

            if not hasattr( child, "wrapped_layer" ):
                raisable = False
                lowerable = False
                layer_deletable = False
                continue

            layer = child.wrapped_layer

            if not hasattr( layer, "parent" ) or layer.parent is None:
                raisable = False
                lowerable = False
                layer_deletable = False
                continue

            if hasattr( layer.parent, "parent" ) and layer.parent.parent:
                raisable = False
                lowerable = False
                layer_deletable = False
                continue

            siblings = [
                sibling for sibling in layer.parent.children
                if not self.ghost_layer( sibling )
            ]

            if len( siblings ) <= 1 or siblings[ -1 ] == layer:
                raisable = False
            if len( siblings ) <= 1 or siblings[ 0 ] == layer:
                lowerable = False

            if child.wrapped_layer.parent.parent is not None:
                layer_deletable = False

        self.outbox.send( 
            request = "selection_updated", 
            selections = selections,
            raisable = raisable,
            lowerable = lowerable,
            deletable = False,
            layer_deletable = layer_deletable,
        )

    @staticmethod
    def ghost_layer( layer ):
        """
        Return whether the given layer should be omitted from the list of
        layers.
        """
        # If the layer is of a type that's always a ghost layer, then we
        # have our answer.
        if layer.__class__.__name__ in Layer_selection_layer.GHOST_LAYER_TYPES:
            return True
        if layer.name is None:
            return True

        return False
