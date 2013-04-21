import maproomlib.utility as utility
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Composite_layer import Composite_layer
from maproomlib.plugin.Selected_whole_layer import Selected_whole_layer


class Label_set_layer:
    """
    A set of text labels. Each label is associated with a point in a
    :class:`Point_set_layer`.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "label_set"
    HIDE_WHEN_ZOOMED = False

    def __init__( self, name, points_layer, command_stack, labels = None ):
        self.name = name
        self.parent = None
        self.points_layer = points_layer
        self.command_stack = command_stack
        self.projection = points_layer.projection
        self.labels = labels
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()

    def run( self, scheduler ):
        self.points_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_updated",
                "depths_updated",
                "points_deleted",
                "points_undeleted",
                "points_added",
                "line_points_added",
                "property_updated",
                "size_changed"
            )
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "get_labels",
                    "get_points",
                    "make_selection",
                    "points_updated",
                    "depths_updated",
                    "points_deleted",
                    "points_undeleted",
                    "points_added",
                    "line_points_added",
                    "property_updated",
                    "get_properties",
                    "size_changed",
                    "triangulate",
                    "contour",
                ),
            )

            request = message.pop( "request" )

            if request == "get_labels":
                self.get_labels( **message )
            elif request == "get_points":
                self.get_points( **message )
            elif request == "make_selection":
                self.make_selection( scheduler, **message )
            elif request in ( "points_added", "line_points_added" ):
                self.points_added( **message )
            elif request == "property_updated":
                self.property_updated( **message )
            elif request == "get_properties":
                response_box = message.get( "response_box" )
                response_box.send(
                    request = "properties",
                    properties = (),
                )
            elif request == "triangulate":
                self.triangulate( **message )
            elif request == "contour":
                self.contour( **message )
            else:
                message[ "request" ] = request
                self.outbox.send( **message )

    def get_labels( self, origin, size, response_box ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = origin,
            size = size,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
        points = message.get( "points" )
        label_count = message.get( "count" )
        
        labels = []

        for index in range( len( points ) ):
            text = self.label_by_index( points, index )
            anchor_position = (
                points.x[ index ],
                points.y[ index ],
            )

            labels.append( ( text, anchor_position ) )
            
        response_box.send(
            request = "labels",
            labels = labels,
            label_count = label_count,
            projection = message.get( "projection" ),
            origin = message.get( "origin" ),
            size = message.get( "size" ),
            hide_when_zoomed = self.HIDE_WHEN_ZOOMED,
        )

    def get_points( self, origin, size, response_box ):
        self.points_layer.inbox.send(
            request = "get_points",
            origin = origin,
            size = size,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
            
        response_box.send( **message )

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

    def points_added( self, layer, points, projection, start_index, count,
                      from_index = None, to_layer = None, to_index = None,
                      selected_start_index = None,
                      selected_end_index = None, undo_recorded = None ):
        labels = []

        for index in range( start_index, start_index + count ):
            text = self.label_by_index( points, index )
            anchor_position = (
                points.x[ index ],
                points.y[ index ],
            )

            labels.append( ( text, anchor_position ) )
            
        self.outbox.send(
            request = "labels_added",
            labels = labels,
            projection = projection,
            start_index = start_index,
            count = count,
        )

    def property_updated( self, layer, property ):
        pass # Can be overridden in derived classes.

    def label_by_index( self, points, point_index ):
        if self.labels is None or \
           point_index < 0 or point_index >= len( self.labels ):
            return None

        return self.labels[ point_index ]

    def triangulate( self, transformer, response_box ):
        response_box.send(
            exception = NotImplementedError(
                "Label layers do not support triangulation.",
            ),
        )

    def contour( self, response_box, **kwargs ):
        response_box.send(
            exception = NotImplementedError(
                "Label layers do not support contouring. Please select a point layer to contour.",
            ),
        )
