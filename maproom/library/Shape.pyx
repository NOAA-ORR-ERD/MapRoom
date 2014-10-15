import cython
import numpy as np
cimport numpy as np

import logging
progress_log = logging.getLogger("progress")

@cython.boundscheck( False )
@cython.cdivision( True )
def points_outside_polygon(
    np.ndarray[ np.float64_t ] points_x not None,
    np.ndarray[ np.float64_t ] points_y not None,
    np.uint32_t point_count,
    np.ndarray[ np.uint32_t ] polygon not None,
):
    """
    Return the indices of the points that occur outside the boundary of the
    given polygon.

    :param points_x: x coordinates of the points
    :type points_x: scalar numpy array of type float64 elements
    :param points_y: y coordinates of the points
    :type points_y: scalar numpy array of type float64 elements
    :param point_count: number of points to check, starting from point 0
    :type point_count: uint32
    :param polygon: indices of polygon boundary points in the points_x and
                    points_y arrays
    :type polygon: scalar numpy array of type uint32 elements
    :return: indices of points occuring outside of the polygon
    :rtype: scalar numpy array of type uint32 elements
    """
    cdef np.uint32_t point_index, polygon_index
    cdef np.uint32_t polygon_point_index, next_polygon_point_index
    cdef np.uint32_t polygon_point_count = polygon.shape[ 0 ]
    cdef np.float64_t x, y, b1x, b1y, b2x, b2y
    cdef bint outside
    cdef np.ndarray[ np.uint32_t ] outside_indices = \
        np.ndarray( ( point_count, ), np.uint32 )
    cdef np.uint32_t outside_add_index = 0
    cdef int count = 0
    cdef int points_per_tick = 700

    progress_log.info("TICKS=%d" % ((point_count/points_per_tick) + 1))
    for point_index in range( point_count ):
        if count >= points_per_tick:
            count = 0
            progress_log.info("Boundary check: %d points, %d outside boundary" % (point_index, outside_add_index))
        count += 1
        
        # See http://alienryderflex.com/polygon/
        x = points_x[ point_index ]
        if x != x: # NaN test
            continue

        y = points_y[ point_index ]
        outside = True

        for polygon_index in range( polygon_point_count ):
            polygon_point_index = polygon[ polygon_index ]

            if point_index == polygon_point_index:
                outside = False
                break

            next_polygon_point_index = polygon[
                ( polygon_index + 1 ) % polygon_point_count
            ]

            b1x = points_x[ polygon_point_index ]
            b1y = points_y[ polygon_point_index ]
            b2x = points_x[ next_polygon_point_index ]
            b2y = points_y[ next_polygon_point_index ]

            if ( b1y < y and b2y >= y ) or ( b2y < y and b1y >= y ):
                if b1x + ( y - b1y ) / ( b2y - b1y ) * ( b2x - b1x ) < x:
                    outside = not outside

        if outside:
            outside_indices[ outside_add_index ] = point_index
            outside_add_index += 1

    #outside_indices.resize( ( outside_add_index, ) )
    ## getting a exception on the resize
    outside_indices = np.resize(outside_indices, ( outside_add_index, ))
    return outside_indices


@cython.boundscheck( False )
@cython.cdivision( True )
def points_inside_polygon(
    np.ndarray[ np.double_t ] points_x not None,
    np.ndarray[ np.double_t ] points_y not None,
    np.uint32_t point_count,
    np.ndarray[ np.double_t ] polygon_x not None,
    np.ndarray[ np.double_t ] polygon_y not None,
):
    """
    Return whether any of the given candidate points occur inside the boundary
    of the given polygon.

    :param points_x: x coordinates of the points
    :type points_x: scalar numpy array of type double elements
    :param points_y: y coordinates of the points
    :type points_y: scalar numpy array of type double elements
    :param point_count: number of points to check, starting from point 0
    :type point_count: uint32
    :param points_x: x coordinates of the polygon points, with the first and
                     last point the same
    :type points_x: scalar numpy array of type double elements
    :param points_y: y coordinates of the polygon points, with the first and
                     last point the same
    :type points_y: scalar numpy array of type double elements
    :return: True if the points fall within the given boundary polygon
    :rtype: bool
    """
    cdef np.uint32_t point_index, polygon_point_index
    cdef np.uint32_t polygon_point_count = polygon_x.shape[ 0 ]
    cdef np.double_t x, y, b1x, b1y, b2x, b2y
    cdef bint inside = False

    for point_index in range( point_count ):
        # See http://alienryderflex.com/polygon/
        x = points_x[ point_index ]

        if x != x: # NaN test
            continue

        y = points_y[ point_index ]

        for polygon_point_index in range( 1, polygon_point_count ):
            b1x = polygon_x[ polygon_point_index - 1 ]
            b1y = polygon_y[ polygon_point_index - 1 ]
            b2x = polygon_x[ polygon_point_index ]
            b2y = polygon_y[ polygon_point_index ]

            if ( b1y < y and b2y >= y ) or ( b2y < y and b1y >= y ):
                if b1x + ( y - b1y ) / ( b2y - b1y ) * ( b2x - b1x ) < x:
                    inside = not inside

        if inside:
            return True

    return False


@cython.boundscheck( False )
@cython.cdivision( True )
def point_in_polygon(
    np.ndarray[ np.float64_t ] points_x not None,
    np.ndarray[ np.float64_t ] points_y not None,
    np.uint32_t point_count,
    np.ndarray[ np.uint32_t ] polygon not None,
    np.float64_t x,
    np.float64_t y,
):
    """
    Return whether the given candidate point occurs inside the boundary of
    the given polygon.

    :param points_x: x coordinates of the points
    :type points_x: scalar numpy array of type float64 elements
    :param points_y: y coordinates of the points
    :type points_y: scalar numpy array of type float64 elements
    :param point_count: number of points in points_x and points_y
    :type point_count: uint32
    :param polygon: indices of polygon boundary points in the points_x and
                    points_y arrays
    :type polygon: scalar numpy array of type uint32 elements
    :param x: x coordinate of point to test
    :type x: float64
    :param y: y coordinate of point to test
    :type y: float64
    :return: True if the point falls within the given boundary polygon
    :rtype: bool
    """
    cdef np.uint32_t polygon_index
    cdef np.uint32_t polygon_point_index, next_polygon_point_index
    cdef np.uint32_t polygon_point_count = polygon.shape[ 0 ]
    cdef np.float64_t b1x, b1y, b2x, b2y
    cdef bint inside

    if x != x: # NaN test
        return False

    inside = False

    # See http://alienryderflex.com/polygon/
    for polygon_index in range( polygon_point_count ):
        polygon_point_index = polygon[ polygon_index ]

        next_polygon_point_index = polygon[
            ( polygon_index + 1 ) % polygon_point_count
        ]

        b1x = points_x[ polygon_point_index ]
        b1y = points_y[ polygon_point_index ]
        b2x = points_x[ next_polygon_point_index ]
        b2y = points_y[ next_polygon_point_index ]

        if ( b1y < y and b2y >= y ) or ( b2y < y and b1y >= y ):
            if b1x + ( y - b1y ) / ( b2y - b1y ) * ( b2x - b1x ) < x:
                inside = not inside

    return inside
