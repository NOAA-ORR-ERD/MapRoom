import numpy as np
import logging
import maproomlib.utility as utility
from maproomlib.plugin.Point_set_layer import Point_set_layer


class Selected_line_set_layer:
    """
    A set of selected lines.
    """
    PLUGIN_TYPE = "layer"
    LAYER_TYPE = "line_set"
    EXTRA_CAPACITY_FACTOR = 2.0   # extra capacity to reserve
    LINES_DTYPE = np.dtype( [
        ( "point1", np.uint32 ),
        ( "point2", np.uint32 ),
        ( "type", np.uint32 ),
        ( "color", np.uint32 ),
    ] )

    def __init__( self, command_stack, name, lines_layer, line_width,
                  lines, indices, line_count, color ):
        self.command_stack = command_stack
        self.name = name
        self.lines_layer = lines_layer
        self.projection = lines_layer.projection
        self.line_width = line_width
        self.lines = lines
        self.add_index = line_count
        self.color = color
        self.children = []
        self.inbox = utility.Inbox()
        self.outbox = utility.Outbox()
        self.logger = logging.getLogger( __name__ )

        # Map from self.lines_layer index -> our self.lines index.
        self.line_map = dict(
            [ ( line_index, our_index ) for ( our_index, line_index ) in \
                enumerate( indices ) ]
        )

        # Same as Line_set_layer's point_map, but for our own indices.
        self.point_map = {}
        self.generate_point_map()

    wrapped_layer = property( lambda self: self.lines_layer )

    def generate_point_map( self ):
        self.point_map = {}

        for line_index in range( self.add_index ):
            self.point_map.setdefault(
                self.lines.point1[ line_index ], list(),
            ).append( line_index * 2 )
            self.point_map.setdefault(
                self.lines.point2[ line_index ], list(),
            ).append( line_index * 2 + 1 )

    def run( self, scheduler ):
        self.lines_layer.outbox.subscribe(
            self.inbox,
            request = (
                "points_updated",
                "points_deleted",
                "points_undeleted",
                "points_added",
                "lines_deleted",
                "lines_undeleted",
                "property_updated",
            ),
        )

        while True:
            message = self.inbox.receive(
                request = (
                    "get_lines",
                    "get_point_map",
                    "get_indices",
                    "delete",
                    "undelete",
                    "extend",
                    "unextend",
                    "reextend",
                    "points_updated",
                    "points_deleted",
                    "points_undeleted",
                    "points_added",
                    "lines_deleted",
                    "lines_undeleted",
                    "move",
                    "cleaned_up_undo",
                    "set_property",
                    "get_properties",
                    "property_updated",
                    "close",
                ),
            )
            request = message.pop( "request" )

            if request == "get_lines":
                self.get_lines( **message )
            elif request == "get_point_map":
                self.get_point_map( **message )
            elif request == "get_indices":
                self.get_indices( **message )
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
            elif request in \
                ( "points_updated", "points_deleted", "points_undeleted",
                  "points_added" ):
                message[ "request" ] = request
                message[ "point_map" ] = self.point_map
                self.outbox.send( **message )
            elif request == "lines_deleted":
                self.lines_deleted( **message )
            elif request == "lines_undeleted":
                self.lines_undeleted( **message )
            elif request == "move":
                pass
            elif request == "cleaned_up_undo":
                if len( self.outbox.inboxes.values() ) > 1:
                    continue
                self.lines_layer.outbox.unsubscribe( self.inbox )
                return
            elif request == "set_property":
                message[ "request" ] = request
                self.lines_layer.inbox.send( **message )
            elif request == "get_properties":
                message[ "request" ] = request
                self.lines_layer.inbox.send( **message )
            elif request == "property_updated":
                message[ "request" ] = request
                message[ "layer" ] = self
                self.outbox.send( **message )
            elif request == "close":
                self.lines_layer.outbox.unsubscribe( self.inbox )
                return

    def get_lines( self, origin, size, response_box ):
        self.lines_layer.inbox.send(
            request = "get_lines",
            origin = origin,
            size = size,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "lines" )

        response_box.send(
            request = "lines",
            points = message.get( "points" ),
            point_count = message.get( "count" ),
            lines = self.lines,
            line_count = self.add_index,
            projection = message.get( "projection" ),
            line_width = self.line_width,
            origin = message.get( "origin" ),
            size = message.get( "size" ),
            point_map = self.point_map,
        )

    def get_point_map( self, response_box ):
        response_box.send(
            request = "point_map",
            point_map = self.point_map,
        )

    def get_indices( self, response_box ):
        response_box.send(
            request = "indices",
            indices = self.line_map.keys(),
        )

    def delete( self, record_undo = True ):
        self.lines_layer.inbox.send(
            request = "delete",
            indices = self.line_map.keys(),
            record_undo = record_undo,
        )

    def undelete( self, lines, indices, record_undo = True ):
        self.lines_layer.inbox.send(
            request = "undelete",
            lines = lines,
            indices = self.line_map.keys(),
            record_undo = record_undo,
        )

    def extend( self, layer, indices, response_box, include_range = False,
                record_undo = True, description = None ):
        if layer != self.lines_layer:
            response_box.send(
                request = "selected",
                indices = None,
            )
            return

        count = len( indices )
        start_index = self.add_index

        # Update our selected lines with the new lines, growing their array
        # if necessary.
        if start_index + count > len( self.lines ):
            new_size = int(
                ( start_index + count ) * self.EXTRA_CAPACITY_FACTOR
            )
            self.logger.debug(
                "Growing selected lines array from %d lines capacity to %d." % (
                    len( self.lines ), new_size,
                ),
            )
            self.lines = np.resize(
                self.lines, ( new_size, ),
            ).view( np.recarray )

        self.lines_layer.inbox.send(
            request = "get_lines",
            origin = None,
            size = None,
            response_box = self.inbox,
        )
        message = self.inbox.receive( request = "lines" )

        all_lines = message.get( "lines" )
        unselected_indices = []
        selected_count = 0
        our_index = start_index
        orig_points = message.get( "points" ).copy()
        orig_lines = self.lines.copy()
        orig_line_map = self.line_map.copy()
        orig_point_map = self.point_map.copy()

        for line_index in indices:
            existing_index = self.line_map.get( line_index )

            # If the line is already selected, then unselect it.
            if existing_index is not None:
                unselected_indices.append( existing_index )
                del self.line_map[ line_index ]
                self.point_map[ self.lines.point1[ existing_index ] ].remove(
                    existing_index * 2,
                )
                self.point_map[ self.lines.point2[ existing_index ] ].remove(
                    existing_index * 2 + 1,
                )
                self.lines.point1[ existing_index ] = 0
                self.lines.point2[ existing_index ] = 0

                # If the first lines are unselected, then adjust the
                # start_index to be after it.
                if selected_count == 0:
                    start_index = our_index + 1
            else:
                self.line_map[ line_index ] = our_index
                self.lines[ our_index ] = all_lines[ line_index ]
                self.point_map.setdefault(
                    self.lines.point1[ our_index ], list(),
                ).append( our_index * 2 )
                self.point_map.setdefault(
                    self.lines.point2[ our_index ], list(),
                ).append( our_index * 2 + 1 )
                self.add_index += 1
                our_index += 1
                selected_count += 1

        self.lines.color[ start_index : self.add_index ] = self.color

        response_box.send(
            request = "selected",
            indices = indices,
        )

        if unselected_indices:
            self.outbox.send(
                request = "lines_deleted",
                layer = self,
                indices = unselected_indices,
            )

        if count - len( unselected_indices ) > 0:
            self.outbox.send(
                request = "lines_added",
                layer = self,
                points = message.get( "points" ),
                lines = self.lines,
                projection = message.get( "projection" ),
                indices = range( start_index, start_index + count ),
            )

        if record_undo is True:
            s = "s" if count > 1 else ""

            lines = self.lines[
                start_index : start_index + count
            ].copy()

            self.command_stack.inbox.send(
                request = "add",
                description = description or "Edit Line%s Selection" % s,
                redo = lambda: self.inbox.send(
                    request = "reextend",
                    lines = lines,
                    start_index = start_index,
                    count = selected_count,
                    line_map = self.line_map.copy(),
                    point_map = self.point_map.copy(),
                    points = message.get( "points" ).copy(),
                    unselected_indices = unselected_indices,
                ),
                undo = lambda: self.inbox.send(
                    request = "unextend",
                    start_index = start_index,
                    count = selected_count,
                    line_map = orig_line_map,
                    point_map = orig_point_map,
                    points = orig_points,
                    lines = orig_lines,
                    unselected_indices = unselected_indices,
                ),
            )

    def unextend( self, start_index, count, line_map, point_map, points,
                  lines, unselected_indices ):
        self.line_map = line_map
        self.point_map = point_map

        # Add deleted lines.
        if len( unselected_indices ) > 0:
            for index in unselected_indices:
                self.lines.point1[ index ] = lines.point1[ index ]
                self.lines.point2[ index ] = lines.point2[ index ]

            self.outbox.send(
                request = "lines_added",
                layer = self,
                points = points,
                lines = lines,
                projection = self.projection,
                indices = unselected_indices,
            )

        # Delete added lines.
        if count > 0:
            self.lines.point1[ start_index : start_index + count ].fill( 0 )
            self.lines.point2[ start_index : start_index + count ].fill( 0 )

            self.outbox.send(
                request = "lines_deleted",
                layer = self,
                indices = range( start_index, start_index + count ),
            )

    def reextend( self, lines, start_index, count, line_map, point_map,
                  points, unselected_indices ):
        self.line_map = line_map
        self.point_map = point_map

        # Re-delete deleted points.
        if len( unselected_indices ) > 0:
            for index in unselected_indices:
                self.lines.point1[ index ] = 0
                self.lines.point2[ index ] = 0

            self.outbox.send(
                request = "lines_deleted",
                layer = self,
                indices = unselected_indices,
            )

        # Re-add added points.
        if count > 0:
            self.lines[ start_index : start_index + count ] = lines

            self.outbox.send(
                request = "lines_undeleted",
                layer = self,
                points = points,
                lines = self.lines,
                projection = self.projection,
                indices = range( start_index, start_index + count ),
            )

    def add_lines( self, lines, description = None, record_undo = True,
                   undo_recorded = None ):
        count = len( lines )
        start_index = self.add_index

        if start_index + count > len( self.lines ):
            new_size = int( len( self.lines ) * self.EXTRA_CAPACITY_FACTOR )
            self.logger.debug(
                "Growing lines array from %d lines capacity to %d." % (
                    len( self.lines ), new_size,
                ),
            )
            self.lines = np.resize(
                self.lines, ( new_size, ),
            ).view( np.recarray )

        self.lines[ start_index: start_index + count ] = lines
        self.add_index += count

        for line_index in range( count ):
            doubled_index = ( start_index + line_index ) * 2

            self.point_map.setdefault(
                lines.point1[ line_index ], list(),
            ).append( doubled_index )

            self.point_map.setdefault(
                lines.point2[ line_index ], list(),
            ).append( doubled_index + 1 )

        self.outbox.send(
            request = "lines_added",
            layer = self,
            points = self.points_layer.points,
            lines = self.lines,
            projection = self.projection,
            indices = range( start_index, start_index + count ),
        )

        if record_undo is True:
            s = "s" if count > 1 else ""

            self.command_stack.inbox.send(
                request = "add",
                description = description or "Add Line%s" % s,
                redo = lambda: self.inbox.send(
                    request = "undelete",
                    lines = lines,
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
                undo = lambda: self.inbox.send(
                    request = "delete",
                    indices = range( start_index, start_index + count ),
                    record_undo = False,
                ),
            )

    def lines_deleted( self, layer, indices, undo_recorded = None ):
        deleted_indices = []

        for line_index in indices:
            our_index = self.line_map.get( line_index )

            if our_index is not None:
                self.point_map[ self.lines.point1[ our_index ] ].remove(
                    our_index * 2,
                )
                self.point_map[ self.lines.point2[ our_index ] ].remove(
                    our_index * 2 + 1,
                )
                self.lines.point1[ our_index ] = 0
                self.lines.point2[ our_index ] = 0
                deleted_indices.append( our_index )

        if len( deleted_indices ) > 0:
            self.outbox.send(
                request = "lines_deleted",
                layer = self,
                indices = deleted_indices,
            )

    def lines_undeleted( self, layer, lines, points, projection, indices ):
        undeleted_indices = []

        for line_index in indices:
            our_index = self.line_map.get( line_index )

            if our_index is not None:
                self.lines[ our_index ] = lines[ line_index ]
                self.lines.color[ our_index ] = self.color

                self.point_map.setdefault(
                    lines.point1[ line_index ], list(),
                ).append( our_index * 2 )
                self.point_map.setdefault(
                    lines.point2[ line_index ], list(),
                ).append( our_index * 2 + 1 )

                undeleted_indices.append( our_index )

        if len( undeleted_indices ) > 0:
            self.lines_layer.inbox.send(
                request = "get_lines",
                origin = None,
                size = None,
                response_box = self.inbox,
            )
            message = self.inbox.receive( request = "lines" )

            self.outbox.send(
                request = "lines_undeleted",
                layer = self,
                points = message.get( "points" ),
                lines = self.lines,
                projection = projection,
                indices = undeleted_indices,
            )

    @staticmethod
    def make_lines( count, exact = False, color = None ):
        """
        Make a default line segments array with the given number of rows, plus
        some extra capacity for future additions.
        """
        if not exact:
            count = int( count * Selected_line_set_layer.EXTRA_CAPACITY_FACTOR )

        return np.repeat(
            np.array( [
                (
                    0, 0, 0, color or 0,
                ),
            ], dtype = Selected_line_set_layer.LINES_DTYPE ),
            count,
        ).view( np.recarray )
