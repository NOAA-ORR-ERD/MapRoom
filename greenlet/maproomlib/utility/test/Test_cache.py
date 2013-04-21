import nose.tools
from maproomlib.utility.Cache import Cache


class Test_cache:
    def setUp( self ):
        self.max_item_count = 3
        self.backing_cache = None
        self.cache = Cache( self.max_item_count )

    def test_get_missing( self ):
        assert self.cache.get( "key1" ) is None

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) is None

    def test_get_missing_with_default( self ):
        assert self.cache.get( "key1", 5 ) == 5

        if self.backing_cache:
            assert self.backing_cache.get( "key1", 5 ) == 5

    def test_set_and_get( self ):
        self.cache.set( "key1", "value1" )

        assert self.cache.get( "key1" ) == "value1"

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) == "value1"

    def test_remove_and_get( self ):
        self.cache.remove( "key1" )
        assert self.cache.get_only( "key1" ) is None

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) is None

    def test_set_remove_and_get( self ):
        self.cache.set( "key1", "value1" )

        self.cache.remove( "key1" )
        assert self.cache.get_only( "key1" ) is None

        # remove() doesn't touch the backing cache.
        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) == "value1"

    def test_set_and_get_with_default( self ):
        self.cache.set( "key1", "value1" )

        assert self.cache.get( "key1", 5 ) == "value1"

        if self.backing_cache:
            assert self.backing_cache.get( "key1", 5 ) == "value1"

    def test_update_and_get( self ):
        self.cache.set( "key1", "value1" )
        self.cache.set( "key1", "value1_updated" )

        assert self.cache.get( "key1" ) == "value1_updated"

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) == "value1_updated"

    def test_clear( self ):
        self.cache.clear()


class Test_cache_with_backing_cache( Test_cache ):
    def setUp( self ):
        self.max_item_count = 3
        self.backing_cache = Cache( self.max_item_count )
        self.cache = Cache( self.max_item_count, self.backing_cache )


class Test_cache_with_some_items:
    def setUp( self ):
        self.max_item_count = 3
        self.backing_cache = None
        self.cache = Cache( self.max_item_count )
        self.cache.set( "key1", "value1" )
        self.cache.set( "key2", "value2" )

    def test_get( self ):
        assert self.cache.get( "key1" ) == "value1"
        assert self.cache.get( "key2" ) == "value2"

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) == "value1"
            assert self.backing_cache.get( "key2" ) == "value2"

    def test_get_missing( self ):
        assert self.cache.get( "key3" ) is None

        if self.backing_cache:
            assert self.backing_cache.get( "key3" ) is None

    def test_get_missing_with_default( self ):
        assert self.cache.get( "key3", 5 ) is 5

        if self.backing_cache:
            assert self.cache.get( "key3", 5 ) is 5

    def test_set_and_get( self ):
        self.cache.set( "key3", "value3" )

        assert self.cache.get( "key3" ) == "value3"

        if self.backing_cache:
            assert self.cache.get( "key3" ) == "value3"

    def test_update_and_get( self ):
        self.cache.set( "key1", "value1_updated" )

        assert self.cache.get( "key1" ) == "value1_updated"
        assert self.cache.get( "key2" ) == "value2"

        if self.backing_cache:
            assert self.backing_cache.get( "key1" ) == "value1_updated"
            assert self.backing_cache.get( "key2" ) == "value2"

    def test_clear( self ):
        self.cache.clear()

        if self.backing_cache:
            assert self.cache.get_only( "key1" ) is None
            assert self.cache.get_only( "key2" ) is None

            # Clearing the main cache should not clear the backing cache.
            assert self.backing_cache.get( "key1" ) == "value1"
            assert self.backing_cache.get( "key2" ) == "value2"

            # Ensure that the main cache falls back to the backing cache.
            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == "value2"

            # When get() falls back to the backing cache, it does a set() on
            # the main cache with the retrieved values.
            assert self.cache.get_only( "key1" ) == "value1"
            assert self.cache.get_only( "key2" ) == "value2"
        else:
            assert self.cache.get( "key1" ) is None
            assert self.cache.get( "key2" ) is None


class Test_cache_with_some_items_and_backing_cache( Test_cache_with_some_items ):
    def setUp( self ):
        self.max_item_count = 3
        self.backing_cache = Cache( self.max_item_count )
        self.cache = Cache( self.max_item_count, self.backing_cache )
        self.cache.set( "key1", "value1" )
        self.cache.set( "key2", "value2" )


class Test_cache_full_of_items:
    def setUp( self ):
        self.max_item_count = 3
        self.backing_cache = None
        self.cache = Cache( self.max_item_count )
        self.cache.set( "key1", "value1" )
        self.cache.set( "key2", "value2" )
        self.cache.set( "key3", "value3" )

    def test_get_only( self ):
        assert self.cache.get_only( "key1" ) == "value1"
        assert self.cache.get_only( "key2" ) == "value2"
        assert self.cache.get_only( "key3" ) == "value3"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"

    def test_get_only_missing( self ):
        assert self.cache.get_only( "key4" ) is None

        if self.backing_cache:
            assert self.backing_cache.get_only( "key4" ) is None

    def test_get_only_missing_with_default( self ):
        assert self.cache.get_only( "key4", 5 ) is 5

        if self.backing_cache:
            assert self.backing_cache.get_only( "key4", 5 ) is 5

    def assert_backing_items( self ):
        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"

            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"
            assert self.cache.get( "key4" ) == "value4"

    def test_set_and_get_only( self ):
        self.cache.set( "key4", "value4" )

        # key1 is LRU, so it should be discarded from cache.
        assert self.cache.get_only( "key1" ) is None
        assert self.cache.get_only( "key2" ) == "value2"
        assert self.cache.get_only( "key3" ) == "value3"
        assert self.cache.get_only( "key4" ) == "value4"

        self.assert_backing_items()

    def test_update_and_set_and_get_only( self ):
        # Updating this values makes key1 no longer LRU.
        self.cache.set( "key1", "value1_updated" )
        self.cache.set( "key4", "value4" )

        # key2 is LRU, so it should be discarded from cache.
        assert self.cache.get_only( "key1" ) == "value1_updated"
        assert self.cache.get_only( "key2" ) is None
        assert self.cache.get_only( "key3" ) == "value3"
        assert self.cache.get_only( "key4" ) == "value4"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1_updated"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"

            assert self.cache.get( "key1" ) == "value1_updated"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"
            assert self.cache.get( "key4" ) == "value4"

    def test_update_and_update_and_set_and_get_only( self ):
        # Updating these values makes key1/key2 no longer LRU.
        self.cache.set( "key2", "value2_updated" )
        self.cache.set( "key1", "value1_updated" )
        self.cache.set( "key4", "value4" )

        # key3 is LRU, so it should be discarded from cache.
        assert self.cache.get_only( "key1" ) == "value1_updated"
        assert self.cache.get_only( "key2" ) == "value2_updated"
        assert self.cache.get_only( "key3" ) is None
        assert self.cache.get_only( "key4" ) == "value4"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1_updated"
            assert self.backing_cache.get_only( "key2" ) == "value2_updated"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"

            assert self.cache.get( "key1" ) == "value1_updated"
            assert self.cache.get( "key2" ) == "value2_updated"
            assert self.cache.get( "key3" ) == "value3"
            assert self.cache.get( "key4" ) == "value4"

    def test_set_thrice_and_get_only( self ):
        self.cache.set( "key4", "value4" )

        assert self.cache.get_only( "key1" ) is None
        assert self.cache.get_only( "key2" ) == "value2"
        assert self.cache.get_only( "key3" ) == "value3"
        assert self.cache.get_only( "key4" ) == "value4"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"

        self.cache.set( "key5", "value5" )

        assert self.cache.get_only( "key1" ) is None
        assert self.cache.get_only( "key2" ) is None
        assert self.cache.get_only( "key3" ) == "value3"
        assert self.cache.get_only( "key4" ) == "value4"
        assert self.cache.get_only( "key5" ) == "value5"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"
            assert self.backing_cache.get_only( "key5" ) == "value5"

        self.cache.set( "key6", "value6" )

        assert self.cache.get_only( "key1" ) is None
        assert self.cache.get_only( "key2" ) is None
        assert self.cache.get_only( "key3" ) == None
        assert self.cache.get_only( "key4" ) == "value4"
        assert self.cache.get_only( "key5" ) == "value5"
        assert self.cache.get_only( "key6" ) == "value6"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"
            assert self.backing_cache.get_only( "key4" ) == "value4"
            assert self.backing_cache.get_only( "key5" ) == "value5"
            assert self.backing_cache.get_only( "key6" ) == "value6"

            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"
            assert self.cache.get( "key4" ) == "value4"
            assert self.cache.get( "key5" ) == "value5"
            assert self.cache.get( "key6" ) == "value6"

    def test_update_and_get_only( self ):
        self.cache.set( "key1", "value1_updated" )

        assert self.cache.get_only( "key1" ) == "value1_updated"
        assert self.cache.get_only( "key2" ) == "value2"
        assert self.cache.get_only( "key3" ) == "value3"

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1_updated"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"

            assert self.cache.get( "key1" ) == "value1_updated"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"

    def test_trigger_compact_with_set( self ):
        heap_id = id( self.cache.item_heap )

        for i in range( 20 ):
            self.cache.set( "key2", i )

            # A compact() has occurred, so break.
            if id( self.cache.item_heap ) != heap_id:
                break

        # After compact(), the heap size should be shrunk down to the same
        # size as the item map.
        assert len( self.cache.item_heap ) == len( self.cache.item_map )

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == i
            assert self.backing_cache.get_only( "key3" ) == "value3"

            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == i
            assert self.cache.get( "key3" ) == "value3"

    def test_trigger_compact_with_get( self ):
        heap_id = id( self.cache.item_heap )

        for i in range( 20 ):
            self.cache.get( "key2" )

            if id( self.cache.item_heap ) != heap_id:
                break

        assert len( self.cache.item_heap ) == len( self.cache.item_map )

        if self.backing_cache:
            assert self.backing_cache.get_only( "key1" ) == "value1"
            assert self.backing_cache.get_only( "key2" ) == "value2"
            assert self.backing_cache.get_only( "key3" ) == "value3"

            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"

    def test_clear( self ):
        self.cache.clear()

        if self.backing_cache:
            assert self.cache.get_only( "key1" ) is None
            assert self.cache.get_only( "key2" ) is None
            assert self.cache.get_only( "key2" ) is None

            # Clearing the main cache should not clear the backing cache.
            assert self.backing_cache.get( "key1" ) == "value1"
            assert self.backing_cache.get( "key2" ) == "value2"
            assert self.backing_cache.get( "key3" ) == "value3"

            # Ensure that the main cache falls back to the backing cache.
            assert self.cache.get( "key1" ) == "value1"
            assert self.cache.get( "key2" ) == "value2"
            assert self.cache.get( "key3" ) == "value3"

            # When get() falls back to the backing cache, it does a set() on
            # the main cache with the retrieved values.
            assert self.cache.get_only( "key1" ) == "value1"
            assert self.cache.get_only( "key2" ) == "value2"
            assert self.cache.get_only( "key3" ) == "value3"
        else:
            assert self.cache.get( "key1" ) is None
            assert self.cache.get( "key2" ) is None
            assert self.cache.get( "key3" ) is None


class Test_cache_full_of_items_and_backing_cache( Test_cache_full_of_items ):
    def setUp( self ):
        self.max_item_count = 3
        self.backing_max_item_count = 50
        self.backing_cache = Cache( self.backing_max_item_count )
        self.cache = Cache( self.max_item_count, self.backing_cache )
        self.cache.set( "key1", "value1" )
        self.cache.set( "key2", "value2" )
        self.cache.set( "key3", "value3" )


class Test_cache_with_max_item_count_too_small:
    @nose.tools.raises( ValueError )
    def test_max_item_count_zero( self ):
        Cache( 0 )

    @nose.tools.raises( ValueError )
    def test_max_item_count_negative( self ):
        Cache( -1 )
