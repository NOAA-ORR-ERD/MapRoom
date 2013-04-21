import os
import time
import logging
import traceback
import pyproj
import maproomlib.utility as utility
from Selected_whole_layer import Selected_whole_layer


class Load_layer_error( Exception ):
    """
    An error occurring when attempting to load a file as a layer.

    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Save_layer_error( Exception ):
    """
    An error occurring when attempting to save a layer to a file.

    :param message: the textual message explaining the error
    """
    def __init__( self, message = None ):
        Exception.__init__( self, message )


class Composite_layer:
    """
    A layer composed of multiple child layers. This class suffers from bloated
    base class syndrome and could use some refactoring. You have been warned.

    :param command_stack: undo/redo stack
    :type command_stack: maproomlib.utility.Command_stack
    :param plugin_loader: used to load the appropriate plugin for a file
    :type plugin_loader: maproomlib.utility.Plugin_loader
    :param parent: parent layer containing this layer (if any)
    :type parent: object or NoneType
    :param name: user-facing name of the layer
    :type name: str
    :param child_subscribe_requests: requests for messages from child layers
           to forward
    :type child_subscribe_requests: list of strings

    .. function:: get_layers( response_box )

        When a message with ``request = "get_layers"`` is received within the
        :attr:`inbox`, a handler returns a list of all child layers. It's
        useful for doing one initial poll of the current layers before then
        relying on ``layer_added`` and similar messages to find out about
        layer changes.

        :param response_box: response message sent here
        :type response_box: Inbox

        The list of child layers is sent to this object's :attr:`outbox` as
        follows::

            response_box.send( request = "layers", layers, hidden_layers ) 

    .. function:: load_layer( filename, response_box = None )

        When a message with ``request = "load_layer"`` is received within the
        :attr:`inbox`, a handler attempts to load the given ``filename`` as a
        layer and add it to the list of layers.

        :param filename: full path to the file to load
        :type filename: str
        :param response_box: response message sent here (optional)
        :type response_box: Inbox or NoneType

        If the given ``filename`` could not be loaded, then a
        :class:`Load_layer_error` is sent as a message to the provided
        ``response_box``.

        If the load was successful, then the loaded layer is sent to the
        given ``response_box`` (if any). This ``layer`` is an instance of the
        appropriate layer plugin. The originally provided ``filename`` is
        sent as well::

            response_box.send( request = "layer", layer, filename )

        Additionally, on a successful load, the loaded layer is sent to this
        object's :attr:`outbox` as follows::

            outbox.send( request = "layer_added", layer )

    .. function:: add_layer( layer, description = None, hidden = False, record_undo = True )

        When a message with ``request = "add_layer"`` is received within the
        :attr:`inbox`, a handler adds the given layer to the list of layers.

        :param layer: layer to add
        :type layer: object
        :param description: description to use for undo (optional)
        :type description: str or NoneType
        :param hidden: whether to add the layer as a hidden child layer
                       (defaults to False)
        :type hidden: bool
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The added layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_added", layer )

    .. function:: remove_layer( layer, record_undo = True )

        When a message with ``request = "remove_layer"`` is received within the
        :attr:`inbox`, a handler removes the given layer from the list of
        layers.

        :param layer: layer to remove
        :type layer: object
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The removed layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_removed", layer, parent )

    .. function:: replace_layers( layer, record_undo = True )

        When a message with ``request = "replace_layers"`` is received within
        the :attr:`inbox`, a handler clears the list of layers and adds adds
        the given layer (if any) to the list of layers.

        :param layer: layer to add (optional)
        :type layer: object or NoneType
        :param layers_to_replace: layers to replace (defaults to all child
                                  layers)
        :type layers_to_replace: list or NoneType
        :param description: text to describe this operation in the undo stack
                            (defaults to something generic)
        :type description: str
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        Each removed layer is sent to this object's :attr:`outbox` in a
        separate message:

            outbox.send( request = "layer_removed", layer, parent )

        The added layer (if any) is sent to this object's :attr:`outbox` as
        follows::

            outbox.send( request = "layer_added", layer )

    .. function:: unreplace_layers( layer, removed_layers, record_undo = True )

        When a message with ``request = "unreplace_layers"`` is received within
        the :attr:`inbox`, a handler removes the given layer and restores the
        list of removed layers.

        :param layer: layer to remove
        :type layer: object or NoneType
        :param removed_layers: originally removed layers to restore
        :type removed_layers: list
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The given layer (if any) is sent to this object's :attr:`outbox` as
        follows::

            outbox.send( request = "layer_removed", layer, parent )

        Each added layer is sent to this object's :attr:`outbox` in a separate
        message:

            outbox.send( request = "layer_added", layer )

    .. function:: show_layer( layer, record_undo = True )

        When a message with ``request = "show_layer"`` is received within the
        :attr:`inbox`, a handler sets the layer as visible.

        :param layer: layer to make visible
        :type layer: object
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The shown layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_shown", layer )

    .. function:: hide_layer( layer, record_undo = True )

        When a message with ``request = "hide_layer"`` is received within the
        :attr:`inbox`, a handler sets the layer as hidden.

        :param layer: layer to make hidden
        :type layer: object
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The hidden layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_hidden", layer )

    .. function:: raise_layer( layer, record_undo = True )

        When a message with ``request = "raise_layer"`` is received within the
        :attr:`inbox`, a handler moves the layer to be before its previous
        sibling (if any). Note: It is assumed that the given layer is not
        already the first layer among its siblings.

        :param layer: layer to raise
        :type layer: object
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The raised layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_raised", layer )

    .. function:: lower_layer( layer, record_undo = True )

        When a message with ``request = "lower_layer"`` is received within the
        :attr:`inbox`, a handler moves the layer to be after its next sibling
        (if any). Note: It is assumed that the given layer is not already the
        last layer among its siblings.

        :param layer: layer to lower
        :type layer: object
        :param record_undo: whether to make this operation undoable (defaults
                            to True)
        :type record_undo: bool

        The lowered layer is sent to this object's :attr:`outbox` as follows::

            outbox.send( request = "layer_lowered", layer )

    .. function:: start_progress( **progress_details )

        When a message with ``request = "start_progress"`` is received within
        the :attr:`inbox` (usually from a loading layer), a handler forwards
        the message to this object's :attr:`outbox`. This allows subscribers
        to track load progress.

    .. function:: end_progress( **progress_details )

        When a message with ``request = "end_progress"`` is received within
        the :attr:`inbox` (usually from a loading layer), a handler forwards
        the message to this object's :attr:`outbox`. This allows subscribers
        to track load progress.

    .. function:: get_properties( response_box, indices ):

        When a message with ``request = "get_properties"`` is received within
        the :attr:`inbox`, a handler sends the property data for this layer
        to the given ``response_box``.

        :param response_box: response message sent here
        :type response_box: Inbox
        :param indices: ignored
        :type indices: object

        The response message is sent as follows::

            response_box.send( request = "properties", properties )

        ``properties`` is a tuple of the properties for this layer.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "composite"
    SELECTION_UPDATED_THROTTLE_DELTA = 0.2 # seconds

    def __init__( self, command_stack, plugin_loader, parent, name,
                  child_subscribe_requests = None, children = None,
                  saver = None, supported_savers = None, filename = None,
                  children_metadata = None ):
        self.parent = parent
        self.name = utility.Property( "Layer name", name )
        self.command_stack = command_stack
        self.plugin_loader = plugin_loader
        self.projection = None
        self.children = children or []
        self.hidden_children = set()
        self.saver = saver
        self.supported_savers = supported_savers or []
        self.filename = filename
        self.children_metadata = children_metadata
        self.logger = logging.getLogger( __name__ )
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.selection_updated_time = None
        self.child_subscribe_requests = [
            "start_progress", "end_progress", "selection_updated",
            "flags_updated",
            "layer_shown", "layer_hidden",
        ]
        if child_subscribe_requests:
            self.child_subscribe_requests.extend( child_subscribe_requests )

    def run( self, scheduler, **handlers ):
        if self.parent is not None:
            self.parent.outbox.subscribe(
                self.inbox,
                request = (
                    "replace_selection", "add_selection", "clear_selection",
                    "delete_selection", "move_selection",
                    "triangulate_selected", "contour_selected",
                    "delete_layer",
                    "add_points_to_selected", "add_lines_to_selected",
                    "get_savers_for_selected", "save_selected",
                    "find_duplicates_in_selected",
                    "save",
                    "layer_removed",
                    "property_updated",
                ),
            )

        for child in self.children:
            child.outbox.subscribe(
                self.inbox,
                request = self.child_subscribe_requests,
            )
            scheduler.add( child.run )

        while True:
            requests = [
                "get_layers", "get_indices", "load_layer",
                "add_layer", "create_layer", "remove_layer",
                "replace_layers", "unreplace_layers",
                "show_layer", "hide_layer",
                "raise_layer", "lower_layer",
                "make_selection", "replace_selection", "add_selection",
                "clear_selection", "move_selection", "delete_selection",
                "get_properties",
                "delete",
                "triangulate_selected", "contour_selected",
                "delete_layer", "merge_layers",
                "pan_mode", "add_points_mode", "add_lines_mode",
                "start_zoom_box", "end_zoom_box",
                "add_lines_to_selected", "add_points_to_selected",
                "add_points", "add_lines", "find_duplicates",
                "get_savers_for_selected", "save_selected",
                "find_duplicates_in_selected",
                "save",
                "set_property",
                "property_updated",
                "get_dimensions",
                "layer_removed",
                "contour",
            ]
            requests.extend( self.child_subscribe_requests )
            requests.extend( handlers.keys() )

            message = self.inbox.receive(
                request = requests,
            )

            request = message.pop( "request" )
            handler = handlers.get( request )

            if handler is not None:
                handler( **message )
            elif request == "selection_updated":
                # To prevent conflicting selection_updated messages issued by
                # children of this layer, ignore an empty selection_updated
                # message that comes in immediately after a non-empty one.
                now = time.time()

                if self.selection_updated_time is not None:
                    elapsed = now - self.selection_updated_time

                    if len( message[ "selections" ] ) == 0 and \
                       elapsed < self.SELECTION_UPDATED_THROTTLE_DELTA:
                        continue

                self.selection_updated_time = now
                message[ "request" ] = request
                self.outbox.send( **message )
            elif request in (
                "replace_selection", "add_selection", "clear_selection",
                "move_selection", "delete_selection",
                "triangulate_selected", "contour_selected",
                "delete_layer",
                "pan_mode", "add_points_mode", "add_lines_mode",
                "start_zoom_box", "end_zoom_box",
                "add_lines_to_selected", "add_points_to_selected",
                "add_points", "add_lines", "find_duplicates",
                "get_savers_for_selected", "save_selected",
                "find_duplicates_in_selected",
                "property_updated", "flags_updated",
            ) or request in self.child_subscribe_requests:
                message[ "request" ] = request
                self.outbox.send( **message )
            elif request == "make_selection":
                self.make_selection( scheduler, **message )
            elif request == "get_layers":
                self.get_layers( **message )
            elif request == "get_indices":
                self.get_indices( **message )
            elif request == "load_layer":
                self.load_layer( scheduler, **message )
            elif request == "add_layer":
                self.add_layer( **message )
            elif request == "create_layer":
                self.create_layer( scheduler, **message )
            elif request == "remove_layer":
                self.remove_layer( **message )
            elif request == "replace_layers":
                self.replace_layers( **message )
            elif request == "unreplace_layers":
                self.unreplace_layers( **message )
            elif request == "show_layer":
                self.show_layer( **message )
            elif request == "hide_layer":
                self.hide_layer( **message )
            elif request == "raise_layer":
                self.raise_layer( **message )
            elif request == "lower_layer":
                self.lower_layer( **message )
            elif request == "save":
                self.save( scheduler, **message )
            elif request == "get_properties":
                response_box = message.get( "response_box" )
                response_box.send(
                    request = "properties",
                    properties = ( self.name, ),
                )
            elif request == "set_property":
                self.set_property( **message )
            elif request == "property_updated":
                pass
            elif request == "delete":
                self.remove_layer( self )
            elif request == "merge_layers":
                self.merge_layers( scheduler, **message )
            elif request == "get_dimensions":
                self.get_dimensions( **message )
            elif request == "layer_removed":
                pass
            elif request == "contour":
                message.get( "response_box" ).send(
                    exception = NotImplementedError(
                        "The selected layer does not support contouring. Please select a point layer to contour.",
                    ),
                )

    def make_selection( self, scheduler, object_indices, color, depth_unit,
                        response_box ):
        # Create a selection layer that simply stands in for this layer.
        selection = Selected_whole_layer(
            self.command_stack,
            name = "Selection",
            wrapped_layer = self,
        )
        scheduler.add( selection.run )

        response_box.send(
            request = "selection",
            layer = selection,
        )

    def get_layers( self, response_box ):
        response_box.send(
            request = "layers",
            layers = list( self.children ),
            hidden_layers = set( self.hidden_children ),
        )

    def get_indices( self, response_box ):
        response_box.send(
            request = "indices",
            indices = (),
        )

    def load_layer( self, scheduler, filename, response_box = None,
                    record_undo = True ):
        self.outbox.send(
            request = "start_progress",
            id = "load " + filename,
            message = "Loading %s" % filename,
        )

        # Load a layer plugin appropriate to the given file.
        self.plugin_loader.inbox.send(
            request = "get_plugin",
            plugin_type = "file_loader",
            response_box = self.inbox,
            filename = filename,
            command_stack = self.command_stack,
            plugin_loader = self.plugin_loader,
            parent = self,
        )

        try:
            message = self.inbox.receive( request = "plugin" )
        except utility.Load_plugin_error, error:
            self.outbox.send(
                request = "end_progress",
                id = "load " + filename,
            )
            if response_box:
                response_box.send( exception = error )
            return
        except Exception, error:
            self.outbox.send(
                request = "end_progress",
                id = "load " + filename,
            )
            if response_box:
                response_box.send( exception = Load_layer_error(
                    "The file %s cannot be opened." % filename,
                ) )
            return

        layer = message.get( "plugin" )

        # Raster layers cannot (currently) be reprojected. So prevent a raster
        # layer from being loaded if another raster layer with a different
        # projection is already loaded.
        if layer.PLUGIN_TYPE == "raster_layer" and layer.projection:
            for child in self.children:
                if child.PLUGIN_TYPE != "raster_layer" or \
                   child.projection.srs is None or \
                   layer.projection.srs == child.projection.srs:
                    continue

                self.outbox.send(
                    request = "end_progress",
                    id = "load " + filename,
                )
                if response_box:
                    response_box.send( exception = Load_layer_error(
                        'The file %s has a projection incompatible with the layer "%s".\n\n' % (
                            os.path.basename( filename ),
                            child.name,
                        ) +
                        "Current projection: %s\n" % child.projection.srs +
                        "Incompatible projection: %s" % layer.projection.srs
                    ) )
                return

        scheduler.add( layer.run )
        layer.outbox.subscribe(
            self.inbox,
            request = self.child_subscribe_requests,
        )

        # Once the layer plugin has loaded, add it to the list of layers.
        self.children.append( layer )

        # Notify listeners that the layers have updated.
        if response_box:
            response_box.send( request = "layer", layer = layer,
                               filename = filename )
        self.outbox.send(
            request = "layer_added", layer = layer, parent = self,
        )
        self.outbox.send(
            request = "end_progress",
            id = "load " + filename,
            message = "Loaded %s" % filename,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Load %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "add_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "remove_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def add_layer( self, layer, insert_index = None, description = None,
                   hidden = False, record_undo = True ):
        layer.outbox.subscribe(
            self.inbox,
            request = self.child_subscribe_requests,
        )

        if insert_index is None:
            self.children.append( layer )
        else:
            self.children.insert( insert_index, layer )

        if hidden:
            self.hidden_children.add( layer )

        self.outbox.send(
            request = "layer_added", layer = layer,
            insert_index = insert_index, parent = self,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = description or "Add %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "add_layer",
                    layer = layer,
                    insert_index = insert_index,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "remove_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def create_layer( self, scheduler, plugin_name, name = None,
                      record_undo = True ):
        self.plugin_loader.inbox.send(
            request = "get_plugin",
            plugin_type = "vector_layer",
            plugin_name = plugin_name,
            response_box = self.inbox,
            filename = None,
            command_stack = self.command_stack,
            plugin_loader = self.plugin_loader,
            parent = self,
            name = name,
        )
        message = self.inbox.receive( request = "plugin" )
        layer = message.get( "plugin" )
        scheduler.add( layer.run )

        self.add_layer( layer, record_undo = record_undo )

    def remove_layer( self, layer, record_undo = True ):
        if layer == self:
            if self.parent is not None:
                self.parent.inbox.send(
                    request = "remove_layer",
                    layer = self,
                )
            return

        if layer not in self.children:
            return

        self.children.remove( layer )
        hidden = layer in self.hidden_children
        self.hidden_children.discard( layer )

        self.outbox.send(
            request = "layer_removed", layer = layer, parent = self,
        )
        layer.outbox.unsubscribe( self.inbox )

        self.outbox.send(
            request = "end_progress",
            id = layer.name,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Delete %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "remove_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "add_layer",
                    layer = layer,
                    hidden = hidden,
                    record_undo = False,
                ),
                cleanup = lambda: layer and layer.inbox and layer.inbox.send(
                    request = "cleaned_up_undo",
                )
            )

    def replace_layers( self, layer = None, layers_to_replace = None,
                        insert_index = None, description = None,
                        record_undo = True ):
        if len( self.children ) == 1 and self.children[ 0 ] == layer:
            return

        if layers_to_replace is None:
            layers_to_replace = list( self.children )

        replaced_index = None

        for child in layers_to_replace:
            self.outbox.send(
                request = "layer_removed", layer = child, parent = self,
            )
            child.outbox.unsubscribe( self.inbox )
            replaced_index = self.children.index( child )
            self.children.remove( child )

        if layer is not None:
            layer.outbox.subscribe(
                self.inbox,
                request = self.child_subscribe_requests,
            )

            if replaced_index is None:
                if insert_index is None:
                    self.children.append( layer )
                else:
                    self.children.insert( insert_index, layer )
                    replaced_index = insert_index
            else:
                self.children.insert( replaced_index, layer )

            self.outbox.send(
                request = "layer_added", layer = layer,
                insert_index = replaced_index, parent = self,
            )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = description or "Replace %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "replace_layers",
                    layer = layer,
                    layers_to_replace = layers_to_replace,
                    insert_index = insert_index,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "unreplace_layers",
                    layer = layer,
                    removed_layers = layers_to_replace,
                    insert_index = insert_index,
                    record_undo = False,
                ),
                cleanup = lambda: layer and layer.inbox and layer.inbox.send(
                    request = "cleaned_up_undo",
                )
            )

    def unreplace_layers( self, layer, removed_layers, insert_index = None,
                          record_undo = True ):
        unreplaced_index = None

        if layer is not None:
            layer.outbox.unsubscribe( self.inbox )
            if layer in self.children:
                unreplaced_index = self.children.index( layer )
                self.children.remove( layer )
                self.outbox.send(
                    request = "layer_removed", layer = layer, parent = self,
                )

        for child in removed_layers:
            child.outbox.subscribe(
                self.inbox,
                request = self.child_subscribe_requests,
            )

            if unreplaced_index is None:
                if insert_index is None:
                    self.children.append( child )
                else:
                    self.children.insert( insert_index, child )
                    unreplaced_index = insert_index
            else:
                self.children.insert( unreplaced_index, child )

            self.outbox.send(
                request = "layer_added", layer = child,
                insert_index = unreplaced_index, parent = self,
            )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Unreplace %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "unreplace_layers",
                    layer = layer,
                    removed_layers = removed_layers,
                    insert_index = insert_index,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "replace_layers",
                    layer = layer,
                    layers_to_replace = removed_layers,
                    insert_index = insert_index,
                    record_undo = False,
                ),
            )

    def show_layer( self, layer, record_undo = True ):
        # If the layer isn't one our hidden children, then forward on the
        # request to each of our children.
        if layer not in self.hidden_children:
            for child in self.children:
                child.inbox.send(
                    request = "show_layer",
                    layer = layer,
                    record_undo = record_undo,
                )
            return

        self.outbox.send(
            request = "layer_shown",
            layer = layer,
        )

        self.hidden_children.discard( layer )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Show %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "show_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "hide_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def hide_layer( self, layer, record_undo = True, description = None ):
        if layer in self.hidden_children:
            for child in self.children:
                child.inbox.send(
                    request = "hide_layer",
                    layer = layer,
                    record_undo = record_undo,
                )
            return

        self.outbox.send(
            request = "layer_hidden",
            layer = layer,
        )

        self.hidden_children.add( layer )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Hide %s Layer" % layer.name \
                    if description is None else description,
                redo = lambda: self.inbox.send(
                    request = "hide_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "show_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def raise_layer( self, layer, record_undo = True ):
        index = self.children.index( layer )
        self.children.remove( layer )
        self.children.insert( index + 1, layer )

        self.outbox.send(
            request = "layer_raised",
            layer = layer,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Raise %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "raise_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "lower_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def lower_layer( self, layer, record_undo = True ):
        index = self.children.index( layer )
        self.children.remove( layer )
        self.children.insert( index - 1, layer )

        self.outbox.send(
            request = "layer_lowered",
            layer = layer,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Lower %s Layer" % layer.name,
                redo = lambda: self.inbox.send(
                    request = "lower_layer",
                    layer = layer,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "raise_layer",
                    layer = layer,
                    record_undo = False,
                ),
            )

    def save( self, scheduler, saver, response_box, layer = None,
              filename = None ):
        """
        Use the given :arg:`saver` to save this layer to :arg:`filename`. If
        an error occurs, log it and send it to :arg:`response_box`. Otherwise,
        report success to :arg`response_box`. If a filename is not provided
        but is necessary, then send a "filename_needed" message to
        :arg:`response_box`.
        """
        if filename is None and hasattr( self, "filename" ):
            filename = self.filename
        if filename is None:
            response_box.send(
                request = "filename_needed",
            )
            return

        if saver is None:
            saver = self.saver
        if saver is None:
            response_box.send(
                request = "saver_needed",
            )
            return

        unique_id = "save %s" % filename
        self.outbox.send(
            request = "start_progress",
            id = unique_id,
            message = "Saving %s" % filename,
        )
        self.logger.debug( "Saving %s" % filename )

        # Let other tasks find out that we've started progress on saving.
        scheduler.switch()

        if hasattr( self, "flag_layer" ):
            self.flag_layer.inbox.send(
                request = "clear_selection",
            )

        try:
            filename = saver( self, filename )
            self.filename = filename
            self.saver = saver
        except Exception, error:
            self.logger.error( traceback.format_exc( error ) )
            self.outbox.send(
                request = "end_progress",
                id = unique_id,
            )

            if hasattr( error, "points" ) and error.points and \
               hasattr( self, "flag_layer" ):
                self.flag_layer.inbox.send(
                    request = "replace_selection",
                    layer = error.points_layer
                        if hasattr( error, "points_layer" )
                        else self.points_layer,
                    object_indices = error.points,
                )

            response_box.send(
                exception = Save_layer_error( str( error ) ),
            )

        self.outbox.send(
            request = "end_progress",
            id = unique_id,
            message = "Saved %s" % filename,
        )
        response_box.send( request = "saved" )

    def set_property( self, property, value, response_box = None,
                      record_undo = True ):
        orig_value = property.value

        try:
            property.update( value )
        except ValueError, error:
            if response_box:
                response_box.send( exception = error )
            return

        self.outbox.send(
            request = "property_updated",
            layer = self,
            property = property,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Set %s" % property.name.title(),
                redo = lambda: self.inbox.send(
                    request = "set_property",
                    property = property,
                    value = value,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "set_property",
                    property = property,
                    value = orig_value,
                    record_undo = False,
                ),
            )

    def merge_layers( self, scheduler, layers, response_box,
                      record_undo = True ):
        # Send a merge message to the first layer with the other layers as
        # parameters. This doesn't modify any of the layers.
        layers[ 0 ].inbox.send(
            request = "merge",
            layers = layers[ 1: ],
            response_box = self.inbox,
        )

        # Receive a new merged layer in response.
        try:
            message = self.inbox.receive(
                request = "merged",
                timeout = 1.0,
            )
        except ValueError, error:
            response_box.send(
                exception = error,
            )
            return

        merged = message.get( "layer" )
        scheduler.add( merged.run )

        # Delete the original layers and add the new merged layer.
        self.replace_layers(
            layer = merged,
            layers_to_replace = layers,
            record_undo = False,
        )

        response_box.send(
            request = "layer",
            layer = merged,
        )

        if record_undo is True:
            self.command_stack.inbox.send(
                request = "add",
                description = "Merge Layers",
                redo = lambda: self.inbox.send(
                    request = "replace_layers",
                    layer = merged,
                    layers_to_replace = layers,
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "unreplace_layers",
                    layer = merged,
                    removed_layers = layers,
                    record_undo = False,
                ),
            )

    def get_dimensions( self, response_box = None ):
        origin_xs = []
        origin_ys = []
        corner_xs = []
        corner_ys = []
        lat_long = pyproj.Proj( "+proj=latlong" )

        for child in self.children:
            if not hasattr( child, "origin" ) or not hasattr( child, "size" ):
                continue
            if child.origin is None or child.size is None:
                continue
            if child in self.hidden_children:
                continue

            origin = child.origin
            size = child.size

            # Make sure origin and size coordinates are in lat-long.
            if child.projection.srs != lat_long.srs:
                upper_right = (
                    origin[ 0 ] + size[ 0 ],
                    origin[ 1 ] + size[ 1 ],
                )

                origin = pyproj.transform(
                    child.projection,
                    lat_long,
                    origin[ 0 ], origin[ 1 ],
                )

                upper_right = pyproj.transform(
                    child.projection,
                    lat_long,
                    upper_right[ 0 ], upper_right[ 1 ],
                )

                size = (
                    upper_right[ 0 ] - origin[ 0 ],
                    upper_right[ 1 ] - origin[ 1 ],
                )

            origin_xs.append( origin[ 0 ] )
            origin_ys.append( origin[ 1 ] )
            corner_xs.append( origin[ 0 ] + size[ 0 ] )
            corner_ys.append( origin[ 1 ] + size[ 1 ] )

        if len( origin_xs ) > 0:
            origin = ( min( origin_xs ), min( origin_ys ) )
            size = (
                max( corner_xs ) - origin[ 0 ],
                max( corner_ys ) - origin[ 1 ],
            )
        else:
            origin = ( 0.0, 0.0 )
            size = ( 0.0, 0.0 )

        if response_box is None:
            return ( origin, size, lat_long )
        else:
            response_box.send(
                request = "dimensions",
                origin = origin,
                size = size,
                projection = lat_long,
            )
