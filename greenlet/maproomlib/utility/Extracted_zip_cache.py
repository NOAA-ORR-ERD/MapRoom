import os
import heapq
import errno
import shutil
import zipfile
import logging
from Standard_paths import user_data_dir


class Extracted_zip_cache:
    """
    A lossy disk cache for files extracted from zip archives.

    :param max_zip_count: maximum number of extracted archives to keep in the
                          cache
    :type max_zip_count: int (1 or greater)
    :raises ValueError: when max_zip_count is zero or negative
    """
    def __init__( self, max_zip_count ):
        if max_zip_count < 1:
            raise ValueError( "max_zip_count must be 1 or greater" )

        self.max_zip_count = max_zip_count
        self.cache_dir = os.path.join(
            user_data_dir( os.environ ),
            "cache",
            "zips",
        )
        self.logger = logging.getLogger( __name__ )

        if not os.path.exists( self.cache_dir ):
            os.makedirs( self.cache_dir )

    def path( self, key ):
        return os.path.join(
            self.cache_dir,
            key,
        )

    def get( self, key, default_value = None ):
        path = self.path( key )
        if not os.path.exists( path ):
            return default_value

        return path

    def set( self, key, value ):
        path = self.path( key )
        if os.path.exists( path ):
            return

        os.makedirs( path )

        zip = zipfile.ZipFile( file = value, mode = "r" )

        for name in zip.namelist():
            if name.endswith( "/" ):
                try:
                    os.makedirs( os.path.join( path, name ) )
                except OSError:
                    pass
            else:
                try:
                    os.makedirs( os.path.join( path, os.path.dirname( name ) ) )
                except OSError:
                    pass

                out_file = open( os.path.join( path, name ), "wb" )
                out_file.write( zip.read( name ) )
                out_file.close()

        zip.close()

        self.compact()

    def remove( self, key ):
        path = self.filename( path )

        if os.path.isdir( path ):
            shutil.rmtree( path, ignore_errors = True )

    def compact( self ):
        # Periodically purge the least recently used extracted zip directories
        # from disk.
        self.logger.debug(
            "Starting compaction of zip cache."
        )

        dir_paths = [
            os.path.join( self.cache_dir, dir_name )
            for dir_name in os.listdir( self.cache_dir )
        ]
        dir_count = len( dir_paths )

        # If we're under the maximum zip count, no need to compact, so bail.
        if dir_count < self.max_zip_count:
            self.logger.debug(
                "Completed compaction of zip cache with %s extracted archives (deleted 0)." %
                dir_count
            )
            return 

        access_times = []

        # Make a list of paths sorted by ascending access time.
        for path in dir_paths:
            try:
                atime = os.stat( path ).st_atime
            except ( IOError, OSError ):
                continue

            heapq.heappush(
                access_times,
                (
                    atime,
                    path,
                ),
            )

        delete_count = 0

        # Delete the least recently accessed files until we're at the low
        # water mark.
        while dir_count > self.max_zip_count:
            ( access_time, path ) = heapq.heappop( access_times )

            shutil.rmtree( path, ignore_errors = True )

            delete_count += 1
            dir_count -= 1

        self.logger.debug(
            "Completed compaction of zip cache with %s extracted archives (deleted %s)." %
            ( dir_count, delete_count )
        )

    def clear( self ):
        if os.path.isdir( self.cache_dir ):
            shutil.rmtree( self.cache_dir )

        os.makedirs( self.cache_dir )

    @staticmethod
    def zip_file( filename ):
        try:
            zip = zipfile.ZipFile( file = filename, mode = "r" )
        except ( IOError, OSError, zipfile.BadZipfile ):
            return False

        zip.close()
        return True
