import os
import errno
import heapq
import shutil
import zipfile
import logging
import numpy as np
from Standard_paths import user_data_dir
from Cache_bulk_setter import Cache_bulk_setter
from Scheduler import Scheduler


class Disk_tile_cache:
    """
    A lossy disk tile cache that stores a maximum number of tiles, discarding
    least recently used tiles as necessary.

    :param max_item_count: maximum number of tiles to keep in the cache
    :type max_item_count: int (1 or greater)
    :param pause_writes: whether to initially queue up set() calls
    :type pause_writes: bool
    :raises ValueError: when max_item_count is zero or negative

    Once the cache is full, then adding any additional tile causes several of
    the least recently used tiles to be discarded.
    """
    def __init__( self, max_item_count, pause_writes = False ):
        if max_item_count < 1:
            raise ValueError( "max_item_count must be 1 or greater" )

        self.max_item_count = max_item_count
        self.paused = pause_writes
        self.tile_count = None
        self.bulk_setter = Cache_bulk_setter( self, self.paused )
        self.cache_dir = os.path.join(
            user_data_dir( os.environ ),
            "cache",
            "tiles",
        )
        self.logger = logging.getLogger( __name__ )

        if not os.path.exists( self.cache_dir ):
            os.makedirs( self.cache_dir )

    def filename( self, key ):
        unique_id = key[ 0 ]
        pixel_origin = "%s,%s" % key[ 1 ]
        scale_factor = str( key[ 2 ] )

        return os.path.join(
            self.cache_dir,
            unique_id,
            scale_factor,
            pixel_origin,
        )

    def get( self, key, default_value = None ):
        filename = self.filename( key )
        if not os.path.exists( filename ):
            return None

        in_file = open( filename, "rb" )

        try:
            value = np.load(
                in_file,
            )
        # If the file can't be loaded, remove it and bail.
        except ( IOError, OSError, zipfile.BadZipfile ):
            try:
                os.remove( filename )
            except ( IOError, OSError ):
                pass
            return

        return dict(
            data = value[ "data" ],
            pixel_origin = (
                int( value[ "pixel_origin" ][ 0 ] ),
                int( value[ "pixel_origin" ][ 1 ] ),
            ),
            pixel_size = (
                int( value[ "pixel_size" ][ 0 ] ),
                int( value[ "pixel_size" ][ 1 ] ),
            ),
            scale_factor = float( value[ "scale_factor" ] ),
        )

    get_only = get

    def set( self, key, value ):
        if self.paused:
            self.bulk_setter.inbox.send(
                request = "set",    
                key = key,
                value = value,
            )
            return

        filename = self.filename( key )
        dirname = os.path.dirname( filename )

        if not os.path.exists( dirname ):
            os.makedirs( dirname )

        # If the file already exists, assume that it's current and that we
        # don't need to write out its data again.
        if os.path.exists( filename ):
            return

        out_file = open( filename, "wb" )

        np.savez(
            out_file,
            data = value[ "data" ],
            pixel_origin = value[ "pixel_origin" ],
            pixel_size = value[ "pixel_size" ],
            scale_factor = value[ "scale_factor" ],
        )

        out_file.close()

        if self.tile_count is not None:
            self.tile_count += 1

        self.compact()

    def remove( self, key ):
        filename = self.filename( key )

        # Remove the file, raising if there's an error. However if the error
        # is simply that the file doesn't exist, silently swallow the error.
        try:
            os.remove( filename )
        except ( IOError, OSError ), error:
            if error.errno != errno.ENOENT:
                raise

    def compact( self ):
        # Periodically purge the least recently used tiles from disk.
        high_water_mark = self.max_item_count

        if self.tile_count is not None and \
           self.tile_count <= high_water_mark:
            return

        if self.tile_count is None:
            self.logger.debug(
                "Starting compaction of disk cache."
            )
        else:
            self.logger.debug(
                "Starting compaction of disk cache with %s tiles." %
                self.tile_count
            )

        self.tile_count = 0
        tile_filenames = []
        scheduler = Scheduler.current()

        # Walk the cache, generating a list of filenames for all tiles. Take
        # this opportunity to count the tiles as well.
        for path, dirs, files in os.walk( self.cache_dir ):
            for filename in files:
                tile_filenames.append( os.path.join( path, filename ) )
                self.tile_count += 1

            if self.tile_count % 100 == 0:
                scheduler.switch()

        # If we're under the high water mark, no need to compact, so bail.
        if self.tile_count <= high_water_mark:
            self.logger.debug(
                "Completed compaction of disk cache with %s tiles (deleted 0)." %
                self.tile_count
            )
            return

        access_times = []
        low_water_mark = high_water_mark * 0.9
        i = 0

        # Make a list of files sorted by ascending access time.
        for filename in tile_filenames:
            try:
                atime = os.stat( filename ).st_atime
            except ( IOError, OSError ):
                continue

            heapq.heappush(
                access_times,
                (
                    atime,
                    filename,
                ),
            )

            i += 1
            if i % 100 == 0:
                scheduler.switch()

        delete_count = 0

        # Delete the least recently accessed files until we're at the low
        # water mark.
        while self.tile_count > low_water_mark:
            ( access_time, filename ) = heapq.heappop( access_times )

            try:
                os.remove( filename )
            except ( IOError, OSError ):
                pass

            # Don't leave behind empty directories.
            try:
                os.removedirs( os.path.dirname( filename ) )
            except ( IOError, OSError ):
                pass

            delete_count += 1
            self.tile_count -= 1

            if delete_count % 100 == 0:
                scheduler.switch()

        self.logger.debug(
            "Completed compaction of disk cache with %s tiles (deleted %s)." %
            ( self.tile_count, delete_count )
        )

    def clear( self ):
        if os.path.isdir( self.cache_dir ):
            shutil.rmtree( self.cache_dir )

        os.makedirs( self.cache_dir )
        self.tile_count = 0

    def pause_writes( self ):
        self.paused = True
        self.bulk_setter.control_inbox.discard()
        self.bulk_setter.control_inbox.send(
            request = "pause",
        )

    def unpause_writes( self ):
        self.compact()
        self.paused = False
        self.bulk_setter.control_inbox.discard()
        self.bulk_setter.control_inbox.send(
            request = "unpause",
        )
