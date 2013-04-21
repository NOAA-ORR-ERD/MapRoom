import maproomlib.ui as ui
import maproomlib.utility as utility
from maproomlib.plugin.Opengl_renderer.Tile_set_renderer import Tile_set_renderer
from maproomlib.plugin.Opengl_renderer.Point_set_renderer import Point_set_renderer
from maproomlib.plugin.Opengl_renderer.Line_set_renderer import Line_set_renderer
from maproomlib.plugin.Opengl_renderer.Polygon_set_renderer import Polygon_set_renderer
from maproomlib.plugin.Opengl_renderer.Label_set_renderer import Label_set_renderer
from maproomlib.plugin.Opengl_renderer.Triangle_set_renderer import Triangle_set_renderer
from maproomlib.plugin.Layer_selection_layer import Layer_selection_layer


RENDERER_MAP = dict(
    tile_set = Tile_set_renderer,
    point_set = Point_set_renderer,
    line_set = Line_set_renderer,
    polygon_set = Polygon_set_renderer,
    label_set = Label_set_renderer,
    triangle_set = Triangle_set_renderer,
)


class Composite_renderer:
    """
    A sub-renderer that the :class:`Opengl_renderer` delegates to for
    rendering composite layers.
    """
    def __init__( self, root_layer, layer, viewport, opengl_renderer,
                  transformer, picker ):
        self.root_layer = root_layer
        self.layer = layer
        self.viewport = viewport
        self.inbox = ui.Wx_inbox()
        self.outbox = utility.Outbox()
        self.transformer = transformer
        self.opengl_renderer = opengl_renderer
        self.picker = picker
        self.children = [] # index 0 is bottom; last element is on top
        self.hidden_children = set()
        self.selected_layers = []

        layer_selection_layers = [
            child for child in root_layer.children
            if isinstance( child, Layer_selection_layer )
        ]
        self.layer_selection_layer = layer_selection_layers[ 0 ] \
            if layer_selection_layers else None

    def run( self, scheduler ):
        self.layer.outbox.subscribe(
            self.inbox,
            request = (
                "layer_added", "layer_removed",
                "layer_hidden", "layer_shown",
                "layer_raised", "layer_lowered",
            ),
        )

        # If we don't have a transformer yet, then wait until someone
        # sends us one.
        if self.transformer is None:
            message = self.inbox.receive(
                request = "transformer",
            )
            self.transformer = message.get( "transformer" )

        self.poll_children( scheduler )

        if self.layer_selection_layer:
            self.layer_selection_layer.outbox.subscribe(
                self.inbox,
                request = "selection_updated",
            )

            # Force an initial update so we can find out what's selected.
            self.layer_selection_layer.inbox.send(
                request = "update_selection",
            )

        while True:
            message = self.inbox.receive(
                request = (
                    "update",
                    "layer_added", "layer_removed",
                    "layer_hidden", "layer_shown",
                    "layer_raised", "layer_lowered",
                    "start_progress", "end_progress",
                    "projection_changed",
                    "selection_updated",
                    "close",
                ),
            )

            request = message.get( "request" )
            layer = message.get( "layer" )

            if request == "update":
                self.inbox.discard( request = "update" )
                for renderer in self.children:
                    renderer.inbox.send( **message )
                continue
            elif request == "layer_added":
                self.make_child_renderers(
                    scheduler, layer, self.transformer,
                    insert_index = message.get( "insert_index" ),
                )
                self.opengl_renderer.Refresh( False )
                continue
            elif request in ( "start_progress", "end_progress" ):
                self.outbox.send( **message )
                continue
            elif request == "projection_changed":
                for renderer in self.children:
                    renderer.inbox.send( request = "projection_changed" )
                continue
            elif request == "selection_updated":
                self.selected_layers = [
                    selection.wrapped_layer
                    for ( selection, indices ) in message.get( "selections" )
                    if hasattr( selection, "wrapped_layer" )
                ]
                self.opengl_renderer.Refresh( False )
                continue
            elif request == "close":
                for renderer in self.children:
                    renderer.inbox.send( request = "close" )
                break

            handled = False

            # Iterate over a copy of self.children since it can be changed
            # within the loop.
            for renderer in list( self.children ):
                if renderer.layer == layer:
                    if request == "layer_removed":
                        self.children.remove( renderer )
                        self.hidden_children.discard( renderer )
                        renderer.inbox.send( request = "close" )
                    elif request == "layer_hidden":
                        self.hidden_children.add( renderer )
                    elif request == "layer_shown":
                        self.hidden_children.discard( renderer )
                    elif request == "layer_raised":
                        index = self.children.index( renderer )
                        if index == len( self.children ) - 1: break
                        self.children[ index ] = self.children[ index + 1 ]
                        self.children[ index + 1 ] = renderer
                    elif request == "layer_lowered":
                        index = self.children.index( renderer )
                        if index == 0: break
                        self.children[ index ] = self.children[ index - 1 ]
                        self.children[ index - 1 ] = renderer

                    handled = True

            if handled:
                self.opengl_renderer.Refresh( False )
            else:
                # If the message doesn't apply to one of this renderer's
                # children, then just forward on the whole message to each
                # child renderer and let them sort it out.
                for renderer in self.children:
                    renderer.inbox.send( **message )

        self.layer.outbox.unsubscribe( self.inbox )

        if self.layer_selection_layer:
            self.layer_selection_layer.outbox.unsubscribe( self.inbox )

    def make_child_renderers( self, scheduler, layer, transformer,
                              hidden = False, insert_index = None ):
        if layer in [ renderer.layer for renderer in self.children ]:
            return

        layer_types = layer.LAYER_TYPE
        if not hasattr( layer_types, "__iter__" ):
            layer_types = [ layer_types ]

        for layer_type in layer_types:
            if layer_type == "composite":
                Renderer = self.__class__
            else:
                Renderer = RENDERER_MAP.get( layer_type )

            if not Renderer: continue

            renderer = Renderer(
                self.root_layer,
                layer,
                self.viewport,
                self.opengl_renderer,
                transformer,
                self.picker,
            )
            if insert_index is not None:
                self.children.insert( insert_index, renderer )
            else:
                self.children.append( renderer )

            if hidden:
                self.hidden_children.add( renderer )

            renderer.outbox.subscribe(
                self.inbox,
                request = ( "start_progress", "end_progress" ),
            )
            scheduler.add( renderer.run )

    def poll_children( self, scheduler ):
        self.layer.inbox.send(
            request = "get_layers",
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "layers" )

        for child in message.get( "layers" ):
            hidden = child in message.get( "hidden_layers" )
            self.make_child_renderers(
                scheduler, child, self.transformer, hidden,
            )

    def render( self, pick_mode = False ):
        for renderer in self.children:
            # Don't render hidden children.
            if renderer in self.hidden_children:
                continue

            # Only selected children should be pickable.
            if pick_mode is True and \
               self.layer_selection_layer is not None and \
               self.layer not in self.selected_layers and \
               renderer.layer not in self.selected_layers and \
               not isinstance( renderer, Composite_renderer ):
                continue

            renderer.render( pick_mode )

    def delete( self ):
        for renderer in self.children:
            renderer.delete() 
