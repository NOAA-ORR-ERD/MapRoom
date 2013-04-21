import numpy as np
import pyproj
from Cache import Cache


class Transformer:
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
        self.projection = projection

        # ( id of original array, original projection ) -> transformed points
        self.cache = Cache( self.MAX_CACHE_ENTRIES )

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
            source_projection, self.projection,
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
            self.projection, destination_projection,
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
            source_projection, self.projection,
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
        self.projection = projection

    def __call__( self, box ):
        """
        Return a wrapper around a given inbox or outbox. The wrapper can be
        used in place of the original box in order to transform its points.
        The wrapper is created with the cache of this :class:`Transformer`,
        which means that multiple wrappers created from a particular
        :class:`Transformer` instance all share the same cache.

        :param box: the inbox or outbox to wrap
        :type box: utility.Inbox or utility.Outbox
        :return: the wrapped box
        :rtype: Box_transformer
        """
        return Box_transformer( box, self )


class Box_transformer:
    """
    A mostly transparent wrapper for a :class:`maproomlib.utility.Inbox` or
    :class:`maproomlib.utility.Outbox` that transforms points in messages sent
    with this :class:`Box_transformer`.

    A :class:`Box_transformer` is most useful when you'd like to subscribe to
    an outbox that broadcasts messages containing points in a projection other
    than the desired projection.

    This class is more than meets the eye. Besides providing transformation
    services, it also caches point arrays so that re-transformation is only
    performed when necessary.

    Note that the caching is keyed off of the original points array and its
    original projection, so if a new points array and/or source projection is
    received, then this class won't look in the cache.

    :param box: wrapped box
    :type box: Inbox or Outbox
    :param transformer: what actually performs the transformations
    :type transformer: Transformer
    """
    def __init__( self, box, transformer ):
        self.box = box
        self.transformer = transformer

    def send( self, **message ):
        """
        Send the given message. If the message contains a ``points`` key and
        an associated ``projection`` key, then its value is transformed
        accordingly, and its ``projection`` value is replaced with the new
        projection.

        The ``points`` value is expected to be a numpy recarray of dtype
        attr:`plugin.Point_set_layer.POINTS_DTYPE`. Also note that the
        transformation will be performed on a copy of the array to avoid
        modifying the original.

        If the message contains a ``request`` key with the value
        "points_updated" or "points_undeleted", then the newly updated points
        are transformed to the given ``projection`` and then the cache is
        updated with those new points. Similar, a ``request`` of
        "points_deleted" causes the deleted points to be set to NaN in the
        cache.

        :param message: original message to send
        :type message: dict
        """
        # Hack to more aggressively invalidate cache entries when the layer
        # containing the cached points is forcibly closed.
        if message.get( "request" ) == "points_nuked":
            self.transformer.cache_remove(
                message.get( "points" ),
                message.get( "projection" ),
            )
            self.box.send( **message )
            return

        orig_points = message.get( "points" )
        orig_projection = message.get( "projection" )
        if orig_points is None or orig_projection is None:
            self.box.send( **message )
            return

        points = self.transformer.cache_lookup( orig_points, orig_projection )

        # If the requested projection has cached points, then use those.
        if points is not None and len( points ) == len( orig_points ):
            # If the points have been updated, then transform just the updated
            # points and replace the relevant points stored in the cache.
            if message.get( "request" ) in \
               ( "points_added", "line_points_added" ):
                start_index = message.get( "start_index" )
                count = message.get( "count" )
                orig_points = message.get( "points" )[
                    start_index : start_index + count
                ]
                updated_points = points[ start_index : start_index + count ]
                self.transformer.transform_many(
                    source_points = orig_points,
                    destination_points = updated_points,
                    source_projection = orig_projection,
                    # The points are being updated directly in the cache, so
                    # there's no need to explicitly do a cache set.
                    set_cache = False,
                )
                updated_points.z = orig_points.z
                updated_points.color = orig_points.color
            elif message.get( "request" ) == "points_deleted":
                indices = message.get( "indices" )

                for index in indices:
                    updated_points = points[ index : index + 1 ]
                    updated_points.x.fill( np.nan )
                    updated_points.y.fill( np.nan )
            elif message.get( "request" ) in \
                    ( "points_updated", "points_undeleted" ):
                indices = message.get( "indices" )
                orig_points = message.get( "points" )
                updated_points = points

                for index in indices:
                    self.transformer.transform_many(
                        source_points = \
                            orig_points[ index : index + 1 ],
                        destination_points = \
                            updated_points[ index : index + 1 ],
                        source_projection = orig_projection,
                        set_cache = False,
                    )

            message[ "points" ] = points.copy()
            message[ "projection" ] = self.transformer.projection

            self.box.send( **message )
            return

        # Otherwise, transform the points and save them off for next time.
        points = orig_points.copy()

        self.transformer.transform_many(
            source_points = orig_points,
            destination_points = points,
            source_projection = orig_projection,
            set_cache = True,
        )

        message[ "points" ] = points
        message[ "projection" ] = self.transformer.projection
        self.box.send( **message )

    def __getattr__( self, name ):
        if hasattr( self.box, name ):
            return getattr( self.box, name )
        else:
            raise AttributeError, name
