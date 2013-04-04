import time
import heapq


class Cache_item:
    """
    An internal wrapper for a cached data item. This wrapper keeps track of
    the most recent use time of its data.

    :param key: key of item stored in the cache
    :type key: hashable
    :param value: value of item stored in the cache
    :type value: object
    :param use_time: a counter indicating when this item was initially used
    :type use_time: int
    """
    def __init__( self, key, value, use_time ):
        self.key = key
        self.value = value
        self.use_time = use_time
        self.heap_refcount = 0

    def __lt__( self, other ): # pragma: no cover
        return self.use_time < other.use_time

    def __le__( self, other ): # pragma: no cover
        return self.use_time <= other.use_time

    def __gt__( self, other ): # pragma: no cover
        return self.use_time > other.use_time

    def __ge__( self, other ): # pragma: no cover
        return self.use_time >= other.use_time


class Cache:
    """
    A lossy memory cache that stores a maximum number of items, discarding
    least recently used items as necessary.

    :param max_item_count: maximum number of items to keep in the cache
    :type max_item_count: int (1 or greater)
    :param backing_cache: another cache that this cache will use itself
                          (optional)
    :type backing_cache: Cache or similar
    :raises ValueError: when max_item_count is zero or negative

    Once the cache is full, then adding any additional item causes the least
    recently used item to be discarded.

    If an optional :attr:`backing_cache` is provided, then all cache items
    set into this cache will be set into the :attr:`backing_cache` as well.
    And when a cache miss occurs, this cache will look for the key in the
    :attr:`backing_cache`. This allows things like a disk cache to act as a
    backing store for a memory cache.
    """
    def __init__( self, max_item_count, backing_cache = None ):
        if max_item_count < 1:
            raise ValueError( "max_item_count must be 1 or greater" )

        self.max_item_count = max_item_count
        self.backing_cache = backing_cache
        self.item_map = {}  # key -> Cache_item object
        self.item_heap = [] # heap of Cache_items, ordered by use_time
        self.use_time = 0   # a simple counter rather than a timestamp

    def get( self, key, default_value = None ):
        """
        Lookup the given key in the cache. If found, return its value.
        Otherwise, return the value of the key from the backing cache (if
        any) and set it into this cache. Short of that, just return the
        :attr:`default_value`.

        :param key: key to lookup in the cache
        :type key: hashable
        :param default_value: default to return if key is not found
        :type default_value: object
        :return: corresponding value
        :rtype: object

        This method also performs bookkeeping to update the use time of the
        key/value pair.
        """
        item = self.item_map.get( key )
        if item is None:
            if self.backing_cache:
                value = self.backing_cache.get( key, default_value )
                if value is not default_value:
                    self.set( key, value, set_backing_cache = False )
                return value
            return default_value

        item.use_time = self.use_time
        self.use_time += 1
        heapq.heappush( self.item_heap, item )
        item.heap_refcount += 1

        self.compact()

        return item.value

    def get_only( self, key, default_value = None ):
        # Perform a get without updating the use time or checking the backing
        # cache. Useful in unit tests.
        item = self.item_map.get( key )
        if item is None:
            return default_value

        return item.value

    def set( self, key, value, set_backing_cache = True ):
        """
        Set the given key into the cache with the corresponding value. If
        there is a backing cache, set it there as well.

        :param key: key to set into the cache
        :type key: hashable
        :param value: corresponding value to set
        :type value: object
        :param set_backing_cache: whether to set the backing cache as well
                                  (defaults to True)
        :type set_backing_cache: bool

        This method also performs bookkeeping to update the use time of the
        key/value pair.
        """
        item = self.item_map.get( key )

        # Also set the key/value pair into the backing cache, if any.
        if set_backing_cache and self.backing_cache:
            self.backing_cache.set( key, value )

        # If the item already exists in the cache, just update it and return.
        if item is not None:
            item.value = value
            item.use_time = self.use_time
            self.use_time += 1
            heapq.heappush( self.item_heap, item )
            item.heap_refcount += 1
            self.compact()
            return

        # Otherwise, add a new item.
        item = Cache_item( key, value, self.use_time )
        self.use_time += 1
        self.item_map[ key ] = item

        # If necessary to maintain a maximum cache size, discard the least
        # recently used item.
        if len( self.item_map ) > self.max_item_count:
            if len( self.item_heap ) > 0:
                discarded_item = heapq.heapreplace( self.item_heap, item )

            # Since the item heap can have duplicates, keep looking until
            # we find an item with a heap refcount of 1 (a non-duplicate).
            # Then remove that item from the item map.
            while len( self.item_heap ) > 0:
                if discarded_item.heap_refcount == 1 and \
                   discarded_item.key in self.item_map:
                    del self.item_map[ discarded_item.key ]
                    break

                discarded_item = heapq.heappop( self.item_heap )
        else:
            heapq.heappush( self.item_heap, item )
            item.heap_refcount += 1

        self.compact()

    def remove( self, key ):
        try:
            del self.item_map[ key ]
        except KeyError:
            pass

    def compact( self ):
        # Periodically compact the heap by removing duplicate keys.
        if len( self.item_heap ) < self.max_item_count * 4:
            return

        new_heap = []

        for item in self.item_heap:
            if item.heap_refcount == 1:
                heapq.heappush( new_heap, item )
            else:
                item.heap_refcount -= 1

        self.item_heap = new_heap

    def clear( self ):
        self.item_map = {}
        self.item_heap = []
