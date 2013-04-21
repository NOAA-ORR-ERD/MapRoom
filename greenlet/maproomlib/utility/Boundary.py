import numpy as np


class Find_boundaries_error( Exception ):
    def __init__( self, message, points = None ):
        Exception.__init__( self, message )
        self.points = points


def find_boundaries( points, point_count, lines, line_count ):
    # Build up a map of adjacency lists: point index -> list of indices
    # of adjacent points connected by lines.
    adjacency_map = {}
    non_boundary_points = set( xrange( 0, point_count ) )

    for line_index in xrange( line_count ):
        point1 = lines.point1[ line_index ]
        point2 = lines.point2[ line_index ]

        if point1 == point2:
            continue
        if np.isnan( points.x[ point1 ] ) or \
           np.isnan( points.x[ point2 ] ):
            continue

        adjacent1 = adjacency_map.setdefault( point1, [] )
        adjacent2 = adjacency_map.setdefault( point2, [] )

        if len( adjacent1 ) >= 2:
            raise Find_boundaries_error(
                "Branching boundaries are not supported.",
                points = ( point1, point2 ) + tuple( adjacent1 ),
            )
        if len( adjacent2 ) >= 2:
            raise Find_boundaries_error(
                "Branching boundaries are not supported.",
                points = ( point1, point2 ) + tuple( adjacent2 ),
            )

        if point2 not in adjacent1:
            adjacent1.append( point2 )
        if point1 not in adjacent2:
            adjacent2.append( point1 )
        non_boundary_points.discard( point1 )
        non_boundary_points.discard( point2 )

    # Walk the adjacency map to create a list of line boundaries.
    boundaries = [] # ( boundary point index list, boundary area )

    while len( adjacency_map ) > 0:
        boundary = []
        area = 0.0
        previous_point = None

        # Start from an arbitrary point.
        ( point, adjacent ) = adjacency_map.iteritems().next()
        boundary.append( point )
        del( adjacency_map[ point ] )

        while True:
            # If the first adjacent point is not the previous point, add it
            # to the boundary. Otherwise, try adding the second adjacent
            # point. If there isn't one, then the boundary isn't closed.
            if len( adjacent ) == 1:
                raise Find_boundaries_error(
                    "Only closed boundaries are supported.",
                    points = ( boundary[ -2 ], boundary[ -1 ], )
                             if len( boundary ) >= 2
                             else ( boundary[ 0 ], adjacent[ 0 ] ),
                )
            elif adjacent[ 0 ] != previous_point:
                adjacent_point = adjacent[ 0 ]
            elif adjacent[ 1 ] != previous_point:
                adjacent_point = adjacent[ 1 ]
            elif adjacent[ 0 ] == adjacent[ 1 ]:
                adjacent_point = adjacent[ 0 ]
            else:
                raise Find_boundaries_error(
                    "Two points are connected by multiple line segments.",
                    points = ( previous_point, ) + tuple( adjacent ),
                )
            
            previous_point = boundary[ -1 ]

            if adjacent_point != boundary[ 0 ]:
                # Delete the map as we walk through it so we know when we're
                # done.
                boundary.append( adjacent_point )
                adjacent = adjacency_map.pop( adjacent_point )

            # See http://alienryderflex.com/polygon_area/
            area += \
                ( points.x[ previous_point ] + points.x[ adjacent_point ] ) * \
                ( points.y[ previous_point ] - points.y[ adjacent_point ] )

            # If the adjacent point is the first point in the boundary,
            # the boundary is now closed and we're done with it.
            if adjacent_point == boundary[ 0 ]:
                break

        boundaries.append( ( boundary, 0.5 * area ) )

    if len( boundaries ) < 1:
        raise Find_boundaries_error(
            "At least one closed boundary is required."
        )

    # Find the outer boundary that contains all the other boundaries.
    # Determine this by simply selecting the boundary with the biggest
    # interior area.
    outer_boundary = None
    outer_boundary_index = None
    outer_boundary_area = None

    for ( index, ( boundary, area ) ) in enumerate( boundaries ):
        if outer_boundary_area is None or \
           abs( area ) > abs( outer_boundary_area ):
            outer_boundary = boundary
            outer_boundary_index = index
            outer_boundary_area = area

    # Make the outer boundary first in the list of boundaries if it's not
    # there already.
    if outer_boundary_index != 0:
        del( boundaries[ outer_boundary_index ] )
        boundaries.insert( 0, ( outer_boundary, outer_boundary_area ) )

    return ( boundaries, non_boundary_points )
