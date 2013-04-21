import numpy as np
from maproomlib.plugin.Label_set_layer import Label_set_layer


class Depth_label_set_layer( Label_set_layer ):
    """
    A set of text depth labels. Each label is associated with a point in a
    :class:`Point_set_layer`, which is where its depth is defined.
    """
    PLUGIN_TYPE = "layer"
    HIDE_WHEN_ZOOMED = True

    def property_updated( self, layer, property ):
        if property.name != "Depth":
            return

        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
        points = message.get( "points" )
        labels = []

        for index in property.indices:
            text = self.label_by_index( points, index )
            anchor_position = points[ index ]
            labels.append( ( text, anchor_position, index ) )

        self.outbox.send(
            request = "labels_updated",
            labels = labels,
            projection = message.get( "projection" ),
        )
        
    def label_by_index( self, points, point_index ):
        if point_index < 0 or point_index >= len( points ):
            return None

        depth = points.z[ point_index ]
        if np.isnan( depth ):
            return None

        if depth == int( depth ):
            return str( int( depth ) )

        return str( depth )
