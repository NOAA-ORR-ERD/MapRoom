import time
import pyproj
import numpy as np
import maproomlib.utility as utility

from maproomlib.utility.accumulator import accumulator

from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer, Load_polygon_error
from maproomlib.plugin.Bna_saver import Bna_saver


class Bna_loader_class:

    __name__ = "Bna_loader"
    PLUGIN_TYPE = "file_loader"
    LAND_FEATURE_CODE = 1
    WATER_FEATURE_CODE = 2
    OTHER_FEATURE_CODE = 3
    WATER_POLYGON_COLOR = utility.color_to_int( 0, 0, 0.75, 0.5 )

    def load_bna( self, filename, file = file, file_scanner = None ):
        """
        used by the code below, to separate reading the file from creating the special maproom objects.
        reads the data in the file, and returns:
        
        (point_data, polygon_starts, identifiers, polygon_colors)
        """
        if file_scanner is None:
            from maproomlib.utility import file_scanner
        
        in_file = file( filename, "rU" )
       
        point_data = accumulator( block_shape=(2,), dtype=np.float32 )
        polygon_starts = accumulator( 0, dtype = np.uint32 )
        
        identifiers = []
        polygon_counts = []
        polygon_types = []
        polygon_colors = []
        
        while True:
            line = in_file.readline().strip()
            if not line: break

            ## fixme -- this will break if there are commas in any of the fields!
            pieces = line.split( "," )
            if len( pieces ) != 3:
                raise Load_polygon_error(
                    "The vector file %s is invalid." % filename,
                ) 
            identifiers.append( pieces[ 0 ].strip( '"' ) )

            try:
                feature_code = int( pieces[ 1 ].strip( '"' ) )
            except ValueError:
                feature_code = 0

            num_points = int( pieces[ 2 ] )
            polygon = False

            # A negative num_points value indicates that this is a line
            # rather than a polygon. And if a "polygon" only has 1 or 2
            # points, it's not a polygon.
            if num_points < 3:
                num_points = abs( num_points )
            else:
                polygon = True

            poly = file_scanner.FileScanN_single(in_file, num_points*2).reshape((-1,2))
            # If the last point is a duplicate of the first point, remove it
            if poly[-1,0] == poly[0,0] and poly[-1,1] == poly[0,1]:
                poly = poly[:-1]
                num_points -= 1

            ## fixme: should we be adding polylines and points?
            ##        or put them somewhere separate -- particularly points!
            polygon_starts.append( polygon_starts[-1] + num_points )
            polygon_counts.append( num_points )
            polygon_types.append( feature_code )

            if feature_code == self.WATER_FEATURE_CODE:
                polygon_colors.append( self.WATER_POLYGON_COLOR )
            else:
                polygon_colors.append(
                    Polygon_set_layer.DEFAULT_POLYGON_COLOR,
                )

            #create the segments:
            segs = np.arange(len(point_data), len(point_data)-1 + len(poly)) 
            segs = np.c_[ segs, segs+1 ]
            point_data.extend(poly)
        in_file.close()

        return (
            np.asarray(point_data, dtype=np.float32),
            # It has an extra at the end so that polygon_starts[i+1] is valid.
            polygon_starts[:-1],
            np.asarray( polygon_counts, dtype=np.uint32 ),
            identifiers,
            np.asarray( polygon_types, dtype=np.uint32 ),
            np.asarray( polygon_colors, dtype=np.uint32 ),
        )


    def __call__( self, filename, command_stack, plugin_loader, parent,
                  file = file, file_scanner = None ):
        """
        Returns a Polygon_point_layer
        """
        start = time.time()
        
        ( point_data, polygon_starts, polygon_counts, identifiers,
          polygon_types, polygon_colors ) = \
            self.load_bna( filename, file = file, file_scanner = file_scanner )
        
        lower_left = tuple(point_data.min(0))
        upper_right = tuple(point_data.max(0))
        origin = lower_left
        size = ( upper_right[ 0 ] - lower_left[ 0 ],
                 upper_right[ 1 ] - lower_left[ 1 ],
                 )
        point_count = len( point_data )

        points = Point_set_layer.make_points( point_count )
        points.view( Point_set_layer.POINTS_XY_DTYPE ).xy[ : point_count ] = \
            np.asarray( point_data, dtype = np.float32 )

        polygon_points = Polygon_set_layer.make_polygon_points( point_count )
        polygon_count = len( polygon_starts )

        # Convert the polygon starts list to an actual point adjacency list
        # for all the polygon points.
        for ( polygon_index, start ) in enumerate( polygon_starts ):
            if polygon_index == polygon_count - 1:
                next_start = point_count
            else:
                next_start = polygon_starts[ polygon_index + 1 ]

            for point_index in xrange( start, next_start - 1 ):
                # Point index -> index of next point in polygon
                polygon_points.next[ point_index ] = point_index + 1
                polygon_points.polygon[ point_index ] = polygon_index

            # Close up the polygon, linking its last point to its first.
            polygon_points.next[ next_start - 1 ] = start
            polygon_points.polygon[ next_start - 1 ] = polygon_index

        polygons = Polygon_set_layer.make_polygons( polygon_count )
        polygons.start[ : polygon_count ] = polygon_starts
        polygons.count[ : polygon_count ] = polygon_counts
        polygons.type[ : polygon_count ] = polygon_types
        polygons.color[ : polygon_count ] = polygon_colors
        polygons.group[ : polygon_count ] = polygon_starts

        projection = pyproj.Proj( "+proj=latlong" )
        points_layer = Point_set_layer(
            command_stack,
            Polygon_point_layer.POINTS_LAYER_NAME, points, point_count,
            Point_set_layer.DEFAULT_POINT_SIZE, projection,
            origin = origin, size = size,
        )

        polygons_layer = Polygon_set_layer(
            command_stack,
            Polygon_point_layer.POLYGONS_LAYER_NAME, points_layer,
            polygon_points, polygons, polygon_count,
        )

        return Polygon_point_layer(
            filename, command_stack, plugin_loader, parent,
            points_layer, polygons_layer,
            saver = Bna_saver,
        )

Bna_loader = Bna_loader_class()
