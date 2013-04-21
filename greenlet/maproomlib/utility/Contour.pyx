import cython
from operator import itemgetter
import numpy as np
cimport numpy as np
import maproomlib.utility.cntr as cntr
from maproomlib.utility.Shape import points_inside_polygon
import scipy.linalg as linalg
from scipy.spatial import ckdtree


cdef np.float32_t FACTOR_FACTOR = <np.float32_t>0.1


@cython.boundscheck( False )
@cython.cdivision( True )
def inverse_covariance(
    np.ndarray[ np.float32_t, ndim = 2 ] points not None,
):
    cdef np.ndarray[ np.float32_t, ndim = 2 ] infless_points = \
        points[ :, np.logical_not( np.isinf( points[ 0 ] ) ) ]
    cdef np.uint32_t point_count = infless_points.shape[ 1 ]
    cdef np.uint32_t coords_count = infless_points.shape[ 0 ]

    # Scotts factor
    cdef np.double_t factor = \
        ( point_count ** ( -1.0 / ( coords_count + 4 ) ) ) * \
        FACTOR_FACTOR

    cdef np.ndarray[ np.double_t, ndim = 2 ] covariance = \
        np.cov( infless_points, rowvar = 1 ) * factor * factor

    return linalg.inv( covariance )


@cython.boundscheck( False )
@cython.cdivision( True )
def evaluate_kde(
    np.ndarray[ np.float32_t, ndim = 2 ] points not None,
    np.ndarray[ np.double_t, ndim = 2 ] grid_coords not None,
    tree,
    np.uint32_t neighbor_count,
    np.ndarray[ np.double_t, ndim = 2 ] inv_covariance,
):
    cdef np.uint32_t point_count = points.shape[ 1 ]
    cdef np.uint32_t bucket_count = grid_coords.shape[ 1 ]
    cdef np.ndarray[ np.double_t ] result = \
        np.zeros( ( bucket_count, ), grid_coords.dtype )
    cdef np.uint32_t i
    cdef np.ndarray[ np.double_t, ndim = 2 ] diff, tdiff
    cdef np.ndarray[ np.float32_t, ndim = 2 ] points_transposed = points.T
    cdef np.ndarray[ np.double_t ] energy

    tree_query = tree.query
    newaxis = np.newaxis
    dot = np.dot
    average = np.average
    sum = np.sum
    exp = np.exp

    for i in range( bucket_count ):
        ( distances, indices ) = tree_query(
            grid_coords[ :, i: i + 1 ].T, # Coordinate of a single grid bucket
            neighbor_count,            # Number of nearest neighbors to return
        )

        # cKDTree uses the point count to indicate a missing neighbor, so
        # filter out those values from the results.
        indices = indices[ 0 ]
        indices = indices[ indices != point_count ]

        diff = points_transposed[ indices ].T - grid_coords[ :, i, newaxis ]
        tdiff = dot(
            inv_covariance * ( average( distances ) ** 2 ),
            diff,
        )
        energy = sum( diff * tdiff, axis = 0 ) * 0.5

        result[ i ] = sum( exp( -energy ), axis = 0 )

    return result


@cython.boundscheck( False )
@cython.cdivision( True )
def contour_points(
    np.ndarray[ np.float32_t, ndim = 2 ] points not None,
    np.ndarray[ np.float32_t ] levels not None,
    np.uint32_t neighbor_count,
    np.uint32_t grid_steps,
    np.double_t buffer_factor,
):
    # Regular grid to evaluate kde upon (grid_steps in each dimension).
    cdef np.double_t points_x_min = np.nanmin( points[ :, 0 ] )
    cdef np.double_t points_x_max = np.nanmax( points[ :, 0 ] )
    cdef np.double_t points_y_min = np.nanmin( points[ :, 1 ] )
    cdef np.double_t points_y_max = np.nanmax( points[ :, 1 ] )

    # Buffer the extent out a bit.
    cdef np.double_t points_x_delta = points_x_max - points_x_min
    points_x_min -= points_x_delta * buffer_factor
    points_x_max += points_x_delta * buffer_factor

    cdef np.double_t points_y_delta = points_y_max - points_y_min
    points_y_min -= points_y_delta * buffer_factor
    points_y_max += points_y_delta * buffer_factor

    cdef np.ndarray[ np.double_t ] x_flat = \
        np.r_[ points_x_min: points_x_max: complex( 0, grid_steps ) ]
    cdef np.ndarray[ np.double_t ] y_flat = \
        np.r_[ points_y_min: points_y_max: complex( 0, grid_steps ) ]

    cdef np.ndarray[ np.double_t, ndim = 2 ] grid_x, grid_y
    ( grid_x, grid_y ) = np.meshgrid( x_flat, y_flat ) 

    cdef np.ndarray[ np.double_t, ndim = 2 ] grid_coords = \
        np.append(
            grid_x.reshape( -1, 1 ),
            grid_y.reshape( -1, 1 ),
            axis = 1,
        )

    # cKDTree handles infs but breaks on NaNs.
    points[ np.isnan( points ) ] = np.inf

    # Put the points into a cKDTree so that they are spatially indexed.
    tree = ckdtree.cKDTree( points )
    cdef np.ndarray[ np.float32_t, ndim = 2 ] points_transposed = points.T
    cdef np.ndarray[ np.double_t, ndim = 2 ] inv_conv = \
        inverse_covariance( points_transposed )

    # Make a grid by evaluating the kde, using the tree to speed things up by
    # looking at the nearest points to each kde bucket.
    cdef np.ndarray[ np.double_t ] flat_grid = \
        evaluate_kde(
            points_transposed,
            grid_coords.T,
            tree,
            neighbor_count,
            inv_conv,
        )

    cdef np.ndarray[ np.double_t, ndim = 2 ] grid = \
        flat_grid.reshape( grid_steps, grid_steps ) 

    cdef np.double_t grid_min = grid.min()
    cdef np.double_t grid_max = grid.max()
    cdef np.double_t grid_delta = grid_max - grid_min

    # Normalize the levels to the actual grid data.
    levels = levels * grid_delta + grid_min

    # Finally, contour the grid.
    cdef np.int32_t level_index
    cdef np.double_t area
    cdef np.uint32_t point_index, polygon_index
    cdef np.uint32_t polygon_point_count
    cdef np.ndarray[ np.double_t, ndim = 2 ] polygon
    contours = []

    for level_index in range( levels.shape[ 0 ] ):
        contour = cntr.Cntr( grid_x, grid_y, grid, None )
        polygons = contour.trace( levels[ level_index ] )
        non_holes = []
        groups = []

        for polygon_index in range( <np.uint32_t>len( polygons ) ):
            # See http://alienryderflex.com/polygon_area/
            area = 0.0
            polygon = polygons[ polygon_index ]
            polygon_point_count = <np.uint32_t>len( polygon )

            for point_index in range( polygon_point_count ):
                if point_index == 0: continue
                area += \
                    ( polygon[ point_index - 1, 0 ] + polygon[ point_index, 0 ] ) * \
                    ( polygon[ point_index - 1, 1 ] - polygon[ point_index, 1 ] )

            area = area * 0.5

            # If it's a negative area, so it's wound CCW and is an outer
            # boundary.
            if area < 0.0:
                non_holes.append( polygon_index )
                groups.append( polygon_index )
                continue

            # Otherwise, it's a positive area, and so it's wound CW and is a
            # hole. If any point of the hole falls within the interior of a
            # previous non-hole polygon, then consider the hole to be
            # contained within the non-hole.
            for non_hole_index in non_holes:
                if points_inside_polygon(
                    polygon[ :, 0 ],
                    polygon[ :, 1 ],
                    polygon_point_count,
                    polygons[ non_hole_index ][ :, 0 ],
                    polygons[ non_hole_index ][ :, 1 ],
                ):
                    groups.append( non_hole_index )
                    break

        # Group polygons together by group id, thereby making each non-hole
        # polygon immediately followed by its holes.
        groups_and_polygons = sorted(
            zip( groups, polygons ),
            key = itemgetter( 0 ),
        )

        contours.append( groups_and_polygons )

    return contours
