import operator
import pyproj
import logging
import numpy as np
import maproomlib.utility as utility


class Selected_point_set_layer:
    """
    A set of selected or flagged points.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "point_set"
    EXTRA_CAPACITY_FACTOR = 2.0    # extra capacity to reserve
    POINTS_DTYPE = np.dtype( [
        ( "x", np.float32 ),
        ( "y", np.float32 ),
        ( "z", np.float32 ),
        ( "color", np.uint32 ),
    ] )

    def __init__( self, command_stack, name, points_layer, point_size,
                  points, indices, point_count, color, depth_unit = None,
                  lines_layer = None ):
        self.command_stack = command_stack
        self.name = name
        self.origin = None
        self.size = None
        self.points_layer = points_layer
        self.projection = points_layer.projection
        self.point_size = point_size
        self.points = points
        self.add_index = point_count
        self.color = color
        self.depth_unit = depth_unit
        self.lines_layer = lines_layer
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )

        # Map from self.points_layer index -> our self.points index.
        self.point_map = dict(
            [ ( point_index, our_index ) for ( our_index, point_index ) in \
                enumerate( indices ) ]
        )

        self.update_dimensions()

    wrapped_layer = property( lambda self: self.points_layer )

    def run( self, scheduler ):
        self.points_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "property_updated",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "get_points",
                    "get_indices",
                    "move",
                    "delete",
                    "undelete",
                    "extend",
                    "unextend",
                    "reextend",
                    "add_points",
                    "add_lines",
                    "delete_lines",
                    "undelete_lines",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "cleaned_up_undo",
                    "set_property",
                    "get_properties",
                    "property_updated",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "get_points":
                self.get_points( **message )
            elif request == "get_indices":
                self.get_indices( **message )
            elif request == "move":
                self.move( **message )
            elif request == "delete":
                self.delete( **message )
            elif request == "undelete":
                self.undelete( **message )
            elif request == "extend":
                self.extend( **message )
            elif request == "unextend":
                self.unextend( **message )
            elif request == "reextend":
                self.reextend( **message )
            elif request == "add_points":
                self.add_points( **message )
            elif request == "add_lines":
                self.add_lines( **message )
            elif request == "delete_lines":
                self.delete_lines( **message )
            elif request == "undelete_lines":
                self.undelete_lines( **message )
            elif request == "points_updated":
                self.points_updated( **message )
            elif request == "points_deleted":
                self.points_deleted( **message )
            elif request == "points_undeleted":
                self.points_undeleted( **message )
            elif request == "line_points_added":
                self.line_points_added( **message )
            elif request == "cleaned_up_undo":
                if len( self.outbox.inboxes.values() ) > 1:
                    continue
                self.points_layer.outbox.unsubscribe( self.inbox )
                self.outbox.send(
                    request = "points_nuked",
                    points = self.points,
                    projection = self.projection,
                )
                return
            elif request == "set_property":
                message[ "request" ] = request
                self.points_layer.inbox.send( **message )
            elif request == "get_properties":
                message[ "request" ] = request
                response_box = message[ "response_box" ]
                message[ "response_box" ] = self.inbox
                self.points_layer.inbox.send( **message )
                message = self.inbox.receive( request = "properties" )

                if self.depth_unit is not None and \
                   len( message[ "properties" ] ) >= 2:
                    message[ "properties" ] = (
                        message[ "properties" ][ 0 ],
                        utility.Property(
                            "Depth unit",
                            self.depth_unit,
                            type = str,
                            mutable = False,
                        ),
                        message[ "properties" ][ 1 ],
                    )

                response_box.send( **message )
            elif request == "property_updated":
                message[ "request" ] = request
                message[ "layer" ] = self
                self.outbox.send( **message )
            elif request == "close":
                self.points_layer.outbox.unsubscribe( self.inbox )
                self.outbox.send(
                    request = "points_nuked",
                    points = self.points,
                    projection = self.projection,
                )
                return

    def get_points( self, response_box, origin = None, size = None ):
        response_box.send(
            request = "points",
            points = self.points,
            projection = self.projection,
            count = self.add_index,
            point_size = self.point_size,
            origin = origin,
            size = size,
        )

    def get_indices( self, response_box ):
        response_box.send(
            request = "indices",
            indices = self.point_map.keys(),
        )

    def move( self, movement_vector, indices = None, cumulative = False,
              record_undo = True ):
        self.points_layer.inbox.send(
            request = "move",
            movement_vector = movement_vector,
            indices = self.point_map.keys(),
            cumulative = cumulative,
            record_undo = record_undo,
        )

    def delete( self, indices = None, record_undo = True ):
        self.points_layer.inbox.send(
            request = "delete",
            indices = self.point_map.keys(),
            record_undo = record_undo,
        )

    def undelete( self, points, indices = None, record_undo = True ):
        self.points_layer.inbox.send(
            request = "undelete",
            points = points,
            indices = self.point_map.keys(),
            record_undo = record_undo,
        )

    def extend( self, layer, indices, response_box, include_range = False,
                record_undo = True, description = None ):
        if layer != self.points_layer:
            response_box.send(
                request = "selected",
                indices = None,
            )
            return

        # If requested, select additional points on a line between the
        # existing selected point and the newly selected point. This only
        # works if there is a single already selected point, a single point is
        # being added to the selection, and they're both on the same line
        # (series of line segments).
        if include_range and self.lines_layer and len( self.point_map ) == 1 and \
           len( indices ) == 1 and indices[ 0 ] not in self.point_map:
            self.lines_layer.inbox.send(
                request = "get_point_range",
                start_point_index = self.point_map.keys()[ 0 ],
                end_point_index = indices[ 0 ],
                response_box = self.inbox,
            )

            message = self.inbox.receive( request = "point_range" )
            indices = indices + message.get( "point_indices" )

        count = len( indices )
        start_index = self.add_index

        # Update our selected points with the new points, growing their array
        # if necessary.
        if start_index + count > len( self.points ):
            new_size = int(
                ( start_index + count ) * self.EXTRA_CAPACITY_FACTOR
            )
            self.logger.debug(
                "Growing selected points array from %d points capacity to %d." % (
                    len( self.points ), new_size,
                ),
            )
            self.points = np.resize(
                self.points, ( new_size, ),
            ).view( np.recarray )

        self.points_layer.inbox.send(
            request = "get_points",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "points" )
        all_points = message.get( "points" )

        unselected_indices = []
        selected_count = 0
        our_index = start_index
        orig_points = self.points.copy()
        orig_point_map = self.point_map.copy()

        for point_index in indices:
            existing_index = self.point_map.get( point_index )

            # If the point is already selected, then unselect it.
            if existing_index is not None:
                unselected_indices.append( existing_index )
                del self.point_map[ point_index ]
                self.points.x[ existing_index ].fill( np.nan )
                self.points.y[ existing_index ].fill( np.nan )

                # If the first points are unselected, then adjust the
                # start_index to be after it.
                if selected_count == 0:
                    start_index = our_index + 1

            # Otherwise, select the point.
            else:
                self.point_map[ point_index ] = our_index
                self.points[ our_index ] = all_points[ point_index ]
                self.points.color[ our_index ] = self.color
                self.add_index += 1
                our_index += 1
                selected_count += 1

        response_box.send(
            request = "selected",
            indices = indices,
        )

        if unselected_indices:
            self.outbox.send(
                request = "points_deleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = unselected_indices,
            )

        if selected_count > 0:
            self.outbox.send(
                request = "points_added",
                layer = self,
                points = self.points,
                projection = self.projection,
                start_index = start_index,
                count = selected_count,
            )

        if record_undo is True:
            s = "s" if count > 1 else ""

            points = self.points[
                start_index : start_index + count
            ].copy()

            self.command_stack.inbox.send(
                request = "add",
                description = description or "Edit Point%s Selection" % s,
                redo = lambda: self.inbox.send(
                    request = "reextend",
                    points = points,
                    start_index = start_index,
                    count = selected_count,
                    point_map = self.point_map.copy(),
                    unselected_indices = unselected_indices,
                ),
                undo = lambda: self.inbox.send(
                    request = "unextend",
                    start_index = start_index,
                    count = selected_count,
                    point_map = orig_point_map,
                    points = orig_points,
                    unselected_indices = unselected_indices,
                ),
            )

    def unextend( self, start_index, count, point_map, points,
                  unselected_indices ):
        self.point_map = point_map

        # Add deleted points.
        if len( unselected_indices ) > 0:
            for index in unselected_indices:
                self.points.x[ index ] = points.x[ index ]
                self.points.y[ index ] = points.y[ index ]

                self.outbox.send(
                    request = "points_added",
                    layer = self,
                    points = self.points,
                    projection = self.projection,
                    start_index = index,
                    count = 1,
                )

        # Delete added points.
        if count > 0:
            self.points.x[ start_index : start_index + count ].fill( np.nan )
            self.points.y[ start_index : start_index + count ].fill( np.nan )

            self.outbox.send(
                request = "points_deleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = range( start_index, start_index + count ),
            )

    def reextend( self, points, start_index, count, point_map,
                  unselected_indices ):
        self.point_map = point_map

        # Re-delete deleted points.
        if len( unselected_indices ) > 0:
            for index in unselected_indices:
                self.points.x[ index ].fill( np.nan )
                self.points.y[ index ].fill( np.nan )

            self.outbox.send(
                request = "points_deleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = unselected_indices,
            )

        # Re-add added points.
        if count > 0:
            self.points[ start_index : start_index + count ] = points

            self.outbox.send(
                request = "points_undeleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = range( start_index, start_index + count ),
            )

    def add_points( self, points, projection, to_layer = None, to_index = None,
                   record_undo = True ):
        if len( self.point_map ) == 0:
            from_point_index = None
        else:
            # Determine the index of the mostly recently added point (the point
            # with the maximum point_index).
            ( from_point_index, from_our_index ) = sorted(
                self.point_map.items(),
                key = operator.itemgetter( 0 ),
            )[ -1 ]

        self.points_layer.inbox.send(
            request = "add_points",
            points = points.copy(),
            projection = projection,
            from_index = from_point_index,
            to_layer = to_layer,
            to_index = to_index,
            record_undo = record_undo,
        )

    def add_lines( self, points, projection, to_layer = None, to_index = None,
                   record_undo = True ):
        # Add a line from each selected point.
        for ( point_index, our_index ) in self.point_map.iteritems():
            self.points_layer.inbox.send(
                request = "add_lines",
                points = points.copy(),
                projection = projection,
                from_index = point_index,
                to_layer = to_layer,
                to_index = to_index,
                record_undo = record_undo,
            )

    def delete_lines( self, layer, selected_start_index, start_index, count,
                      selected_end_index, record_undo = True ):
        if count > 0:
            self.delete(
                range( start_index, start_index + count ),
                record_undo,
            )
        else:
            self.outbox.send(
                request = "line_points_deleted",
                layer = self,
                selected_start_index = selected_start_index,
                start_index = start_index,
                count = count,
                selected_end_index = selected_end_index,
            )

    def undelete_lines( self, layer, points, selected_start_index,
                        start_index, count, selected_end_index,
                        record_undo = True ):
        if count > 0:
            self.undelete(
                points,
                range( start_index, start_index + count ),
                record_undo,
            )
        else:
            self.outbox.send(
                request = "line_points_added",
                layer = self,
                points = self.points,
                projection = self.projection,
                selected_start_index = selected_start_index,
                start_index = start_index,
                count = count,
                selected_end_index = selected_end_index,
            )

    def points_updated( self, layer, points, projection, indices,
                        undo_recorded = None ):
        updated_indices = []

        for point_index in indices:
            our_index = self.point_map.get( point_index )

            if our_index is not None:
                self.points[ our_index ] = points[ point_index ]
                self.points.color[ our_index ] = self.color
                updated_indices.append( our_index )

        if len( updated_indices ) > 0:
            self.outbox.send(
                request = "points_updated",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = updated_indices,
            )

    def points_deleted( self, layer, points, projection, indices,
                        undo_recorded = None ):
        deleted_indices = []

        for point_index in indices:
            our_index = self.point_map.get( point_index )

            if our_index is not None:
                self.points.x[ our_index ].fill( np.nan )
                self.points.y[ our_index ].fill( np.nan )
                deleted_indices.append( our_index )

        if len( deleted_indices ) > 0:
            self.outbox.send(
                request = "points_deleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = deleted_indices,
            )

    def points_undeleted( self, layer, points, projection, indices ):
        undeleted_indices = []

        for point_index in indices:
            our_index = self.point_map.get( point_index )

            if our_index is not None:
                self.points[ our_index ] = points[ point_index ]
                self.points.color[ our_index ] = self.color
                undeleted_indices.append( our_index )

        if len( undeleted_indices ) > 0:
            self.outbox.send(
                request = "points_undeleted",
                layer = self,
                points = self.points,
                projection = self.projection,
                indices = undeleted_indices,
            )

    def update_dimensions( self ):
        if self.add_index == 0:
            self.origin = None
            self.size = None
        elif self.add_index == 1:
            self.origin = ( self.points.x[ 0 ], self.points.y[ 0 ] )
            self.size = None
        else:
            self.origin = (
                np.nanmin( self.points.x[ : self.add_index ] ),
                np.nanmin( self.points.y[ : self.add_index ] ),
            )
            self.size = (
                np.nanmax( self.points.x[ : self.add_index ] ) - self.origin[ 0 ],
                np.nanmax( self.points.y[ : self.add_index ] ) - self.origin[ 1 ],
            )

        self.outbox.send(
            request = "size_changed",
            origin = self.origin,
            size = self.size,
        )

    @staticmethod
    def make_points( count, exact = False, color = None ):
        """
        Make a points array out of the given indices, plus some extra capacity
        for future additions.
        """
        if not exact:
            count = int( count * Selected_point_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                (
                    np.nan, np.nan, np.nan, color or 0,
                ),
            ], dtype = Selected_point_set_layer.POINTS_DTYPE ),
            count,
        ).view( np.recarray )
