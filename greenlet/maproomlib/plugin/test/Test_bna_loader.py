import numpy as np

import maproomlib.plugin.Bna_loader as Bna_loader
from maproomlib.plugin.Polygon_point_layer import Polygon_point_layer, Load_polygon_error
from maproomlib.plugin.Polygon_set_layer import Polygon_set_layer
from maproomlib.utility.test.Mock_file import Mock_file
from maproomlib.utility.test.Mock_file_scanner import Mock_file_scanner


class Test_bna_loader:
    def setUp( self ):
        # Generate a really simple bna.
        self.filename = "test.bna"
        self.bna_file = Mock_file( self.filename, "w" )
        
        self.bna_file.write(
            '''"Another Name","1", 7                    
            -81.531753540039,31.134635925293
            -81.531150817871,31.134529113769
            -81.530662536621,31.134353637695
            -81.530502319336,31.134126663208
            -81.530685424805,31.133970260620
            -81.531112670898,31.134040832519
            -81.531753540039,31.134635925293
            "A third 'name'","2", 5
            -81.522369384766,31.122062683106
            -81.522109985352,31.121908187866
            -81.522010803223,31.121685028076
            -81.522254943848,31.121658325195
            -81.522483825684,31.121797561646
            "8223","3", 9                      
            -81.523277282715,31.122261047363
            -81.522987365723,31.121982574463
            -81.523200988770,31.121547698975
            -81.523361206055,31.121408462524
            -81.523818969727,31.121549606323
            -81.524078369141,31.121662139893
            -81.524009704590,31.121944427490
            -81.523925781250,31.122068405151
            -81.523277282715,31.122261047363
            '''
        )
        self.bna_file.close()

        self.point_count = 19
        self.line_segments = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (4, 5), (5, 0), (6, 7), (7, 8),
            (8, 9), (9, 10), (10, 6), (11, 12),
            (12, 13), (13, 14), (14, 15), (15, 16),
            (16, 17), (17, 18), (18, 11),
        )

        self.polygon_starts = ( 0, 6, 11 )

        self.command_stack = None
        self.plugin_loader = None
        self.parent = None

    def test_load( self ):
        self.layer = Bna_loader(
            self.filename,
            self.command_stack,
            self.plugin_loader,
            self.parent,
            file = Mock_file,
            file_scanner = Mock_file_scanner,
        )

        assert self.layer.filename == self.filename
        assert self.layer.points_layer
        assert self.layer.points_layer.add_index == self.point_count

        assert self.layer.points_layer.points.x[ 0 ] == np.float32( -81.531753540039 )
        assert self.layer.points_layer.points.y[ 0 ] == np.float32( 31.134635925293 )
        assert np.isnan( self.layer.points_layer.points.z[ 0 ] )

        assert self.layer.points_layer.points.x[ self.point_count - 1 ] == np.float32( -81.523925781250 )
        assert self.layer.points_layer.points.y[ self.point_count - 1 ] == np.float32( 31.122068405151 )
        assert np.isnan( self.layer.points_layer.points.z[ self.point_count - 1 ] )

        assert self.layer.polygons_layer
        assert self.layer.polygons_layer.polygon_add_index == 3
        assert len( self.layer.polygons_layer.polygons ) == 6
        assert self.layer.polygons_layer.polygons.start[ 0 ] == self.polygon_starts[ 0 ]
        assert self.layer.polygons_layer.polygons.start[ 1 ] == self.polygon_starts[ 1 ]
        assert self.layer.polygons_layer.polygons.start[ 2 ] == self.polygon_starts[ 2 ]
