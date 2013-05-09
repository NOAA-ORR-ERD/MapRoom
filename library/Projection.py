import numpy as np
import pyproj
from Cache import Cache


class Projection:
    """
    A class used for transforming points from one projection to another. This
    is typically accomplished by wrapping a :class:`maproomlib.utility.Inbox`
    or :class:`maproomlib.utility.Outbox` and manipulating messages as they are
    sent. But you can also just call :meth:`transform()` directly.

    :param projection: the destination projection which points will be
                       transformed into
    :type projection: pyproj.Proj
    """
    MAX_CACHE_ENTRIES = 20

    def __init__( self, projection ):
        self.proj = pyproj.Proj( projection )
        self.srs = self.proj.srs

        # ( id of original array, original projection ) -> transformed points
        self.cache = Cache( self.MAX_CACHE_ENTRIES )
        
    def projection( self, x, y, inverse = False ):
        if self.is_identity():
            return (x, y)
        
        return self.proj( x, y, inverse=inverse )

    def is_identity( self ):
        identity = self.srs.find( "+proj=latlong" ) != -1
        return identity

    def transform( self, point, source_projection ):
        """
        Transform the point from the given :arg:`source_projection` to the
        destination :arg:`projection` passed to :meth:`__init__()`. Note that
        transforming a point directly like this does not use the cache.

        :param point: point to transform
        :type point: ( x, y ) tuple
        :param source_projection: geographic projection of the point
        :type source_projection: pyproj.Proj
        :return: the transformed point
        :rtype: ( x, y ) tuple
        """
        return pyproj.transform(
            source_projection, self.proj,
            point[ 0 ], point[ 1 ],
        )

    def reverse_transform( self, point, destination_projection ):
        """
        Transform the point from the :arg:`projection` passed to 
        :meth:`__init__()` to the given :arg:`destination_projection`. Note
        that transforming a point directly like this does not use the cache.

        :param point: point to transform
        :type point: ( x, y ) tuple
        :param destination_projection: geographic projection to transform into
        :type destination_projection: pyproj.Proj
        :return: the transformed point
        :rtype: ( x, y ) tuple
        """
        return pyproj.transform(
            self.proj, destination_projection,
            point[ 0 ], point[ 1 ],
        )

    def transform_many( self, source_points, destination_points,
                        source_projection, set_cache = True ):
        """
        Transform several :arg:`source_points` from the given
        :arg:`source_projection` to the destination :arg:`projection` passed
        to :meth:`__init__()` and optionally set them into the cache.

        :param source_points: points in :arg:`source_projection` to transform
        :type source_points: plugin.Point_set_layer.POINTS_DTYPE
        :param destination_points: array where the transformed points will be
                                   stored
        :type destination_points: plugin.Point_set_layer.POINTS_DTYPE
        :param source_projection: projection of the :arg:`source_points`
        :type source_projection: pyproj.Proj
        :param set_cache: True to set the points into the cache (defaults to
                          True)
        :type set_cache: bool
        """
        ( destination_points.x, destination_points.y ) = pyproj.transform(
            source_projection, self.proj,
            source_points.x, source_points.y,
        )

        destination_points.x[ np.isinf( destination_points.x ) ] = np.nan
        destination_points.y[ np.isinf( destination_points.y ) ] = np.nan

        if set_cache and len( destination_points ) > 1:
            key = self.cache_key( source_points, source_projection )
            self.cache.set( key, destination_points )

    def cache_lookup( self, source_points, source_projection ):
        """
        Lookup the given :arg:`source_points` in the cache and return the 
        transformed points.

        :param source_points: points in :arg:`source_projection` to transform
        :type source_points: plugin.Point_set_layer.POINTS_DTYPE
        :param source_projection: projection of the :arg:`source_points`
        :type source_projection: pyproj.Proj
        :return: transformed points, or None if not found in the cache
        :rtype: plugin.Point_set_layer.POINTS_DTYPE
        """
        key = self.cache_key( source_points, source_projection )
        return self.cache.get( key, None )

    def cache_remove( self, source_points, source_projection ):
        """
        Remove the given :arg:`source_points` from the cache.

        :param source_points: points in :arg:`source_projection` to remove
        :type source_points: plugin.Point_set_layer.POINTS_DTYPE
        :param source_projection: projection of the :arg:`source_points`
        :type source_projection: pyproj.Proj
        """
        key = self.cache_key( source_points, source_projection )
        self.cache.remove( key )

    def cache_key( self, source_points, source_projection ):
        """
        Return a cache key based on the given points array, points array
        shape, and projection.

        :param source_points: points in :arg:`source_projection`
        :type source_points: plugin.Point_set_layer.POINTS_DTYPE
        :param source_projection: projection of the :arg:`source_points`
        :type source_projection: pyproj.Proj
        :return: cache key used to lookup the points
        :rtype: hashable object
        """
        return (
            id( source_points ),
            source_projection.srs,
        )

    def change_projection( self, projection ):
        """
        Replace this transformer's destination projection, thereby
        invalidating its entire cache.

        :param projection: new destination projection for all transformations
        :type projection: pyproj.Proj
        """
        self.cache.clear()
        self.proj = pyproj.Proj( projection )
        self.srs = self.proj.srs

import unittest

class ProjectionTests(unittest.TestCase):
    def testIdentity(self):
        proj = Projection( "+proj=latlong" )
        orig_point = (-62.242000,  12.775000)
        new_point = proj.projection(orig_point[0], orig_point[1])
        
        self.assertEqual(orig_point, new_point)

if __name__ == "__main__":
    unittest.main()
            
