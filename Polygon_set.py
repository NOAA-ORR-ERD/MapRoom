"""
def make_selection( self, scheduler, object_indices, color, depth_unit,
                    response_box ):
    # If there are any points selected, bail. A polygon can only be
    # selected if nothing else is selected.
    if self.selection_layer and len( self.selection_layer.children ) > 0  or not object_indices:
        response_box.send(
            request = "selection",
            layer = None,
        )
        return
        
    self.points_layer.inbox.send(
        request = "get_points",
        origin = None,
        size = None,
        response_box = self.inbox,
    )
    message = self.inbox.receive( request = "points" )
    points = message.get( "points" )
        
    # Ignore all polygons other than first one.
    polygon_index = object_indices[ 0 ]
    point_count = self.polygons.count[ polygon_index ]
        
    point_indices = np.empty(
        ( point_count * self.EXTRA_CAPACITY_FACTOR, ),
        np.uint32,
    )
        
    point_index = self.polygons.start[ polygon_index ]
        
    for selected_index in xrange( 0, point_count ):
        point_indices[ selected_index ] = point_index
        point_index = self.polygon_points.next[ point_index ]
        
    self.points_layer.inbox.send(
        request = "change_shown",
        shown_indices = point_indices,
        shown_count = point_count,
    )
        
    # This doesn't really make a selection. It just changes the points
    # shown in the points_layer.
    response_box.send(
        request = "selection",
        layer = None,
    )
    
def points_added_to_new_polygon( self, layer, points, projection,
                                    start_index, count, from_index = None,
                                    to_layer = None, to_index = None,
                                    undo_recorded = None ):
    if start_index + count > len( self.polygon_points ):
        new_size = int( len( self.polygon_points ) * self.EXTRA_CAPACITY_FACTOR )
        self.logger.debug(
            "Growing polygon points array from %d points capacity to %d." % (
                len( self.polygon_points ), new_size,
            ),
        )
        self.polygon_points = np.resize(
            self.polygon_points, ( new_size, ),
        ).view( np.recarray )

    if self.polygon_add_index >= len( self.polygons ):
        new_size = int( len( self.polygons ) * self.EXTRA_CAPACITY_FACTOR )
        self.logger.debug(
            "Growing polygons array from %d polygons capacity to %d." % (
                len( self.polygons ), new_size,
            ),
        )
        self.polygons = np.resize(
            self.polygons, ( new_size, ),
        ).view( np.recarray )
        
    # This is essentially a one-point "polygon", so its next point is
    # itself.
    polygon_index = self.polygon_add_index
    self.polygon_points.next[ start_index ] = start_index
    self.polygon_points.polygon[ start_index ] = polygon_index
    self.polygons.start[ polygon_index ] = start_index
    self.polygons.count[ polygon_index ] = 1
    self.polygons.color[ polygon_index ] = self.default_polygon_color
        
    if polygon_index == 0:
        self.polygons.group[ polygon_index ] = 0
    else:
        # Just take the previous group id and increment it to make our
        # group id.
        self.polygons.group[ polygon_index ] = self.polygons.group[ polygon_index - 1 ] + 1

    self.polygon_add_index += 1
        
    self.outbox.send(
        request = "polygons_updated",
        points = points,
        polygon_points = self.polygon_points,
        polygons = self.polygons,
        polygon_count = self.polygon_add_index,
        updated_points = [ start_index ],
        projection = projection,
    )
    
def points_added_to_existing_polygon( self, layer, points, projection,
                                        start_index, count,
                                        from_index = None, to_layer = None,
                                        to_index = None, undo_recorded = None ):
    if start_index + count > len( self.polygon_points ):
        new_size = int( len( self.polygon_points ) * self.EXTRA_CAPACITY_FACTOR )
        self.logger.debug(
            "Growing polygon points array from %d points capacity to %d." % (
                len( self.polygon_points ), new_size,
            ),
        )
        self.polygon_points = np.resize(
            self.polygon_points, ( new_size, ),
        ).view( np.recarray )
        
    # Insert the point at the correct location in the polygon.
    next_point_index = self.polygon_points.next[ from_index ]
    polygon_index = self.polygon_points.polygon[ from_index ]
    self.polygon_points.next[ from_index ] = start_index
    self.polygon_points.next[ start_index ] = next_point_index
    self.polygon_points.polygon[ start_index ] = polygon_index
        
    self.polygons.count[ polygon_index ] += count
        
    self.outbox.send(
        request = "polygons_updated",
        points = points,
        polygon_points = self.polygon_points,
        polygons = self.polygons,
        polygon_count = self.polygon_add_index,
        updated_points = [ from_index ],
        projection = projection,
    )
    
def points_updated( self, layer, points, projection, indices,
                    undo_recorded = None ):
    start_indices = [
        self.polygons.start[
            self.polygon_points.polygon[ point_index ]
        ] for point_index in indices
    ]
        
    self.outbox.send(
        request = "polygons_updated",
        points = points,
        polygon_points = self.polygon_points,
        polygons = self.polygons,
        polygon_count = self.polygon_add_index,
        updated_points = start_indices,
        projection = projection,
    )
"""
