import os
import re
import hashlib
from osgeo import gdal
from maproomlib.plugin.Gdal_raster_layer import Gdal_raster_layer, Load_gdal_error


def Gdal_loader( filename, command_stack, plugin_loader, parent ):
    # Disable the default error handler so errors don't end up on stderr.
    gdal.PushErrorHandler( "CPLQuietErrorHandler" )

    dataset = gdal.Open( str( filename ) )

    if dataset is None:
        raise Load_gdal_error(
            "The raster file %s is invalid." % filename,
        )

    # If this is a KAP file, read the header to get the name and make a header
    # hash. Otherwise, just make a hash from the first 512 bytes of the file.
    if dataset.GetDriver().ShortName == "BSB":
        data_file = open( filename, "rb" )

        NAME_PATTERN = re.compile( "^BSB/NA=([^,\r\n]*)" )
        END_HEADER = "\x1a\x00"
        header_lines = []
        line = data_file.readline( 512 )

        while line and len( line ) >= 2 and line[ 0: 2 ] != END_HEADER:
            header_lines.append( line )
            match = NAME_PATTERN.search( line )
            if match:
                name = match.group( 1 )

            line = data_file.readline( 512 )

        header_hash = hashlib.sha1(
            "".join( header_lines ),
        ).hexdigest()

    else:
        data_file = open( filename, "rb" )
        header = data_file.read( 512 )
        data_file.close()

        name = None
        header_hash = hashlib.sha1( header ).hexdigest()

    name = name or os.path.basename( filename )

    return Gdal_raster_layer(
        filename, command_stack, plugin_loader, parent, name, dataset,
        header_hash,
    )


Gdal_loader.PLUGIN_TYPE = "file_loader"
