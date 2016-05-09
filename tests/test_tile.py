import os

import numpy as np

from mock import *

from maproom.library.tile_utils import *
from maproom.library.host_utils import *


class TestTileCoordinates(object):
    def setup(self):
        self.host = LocalTileHost("Blank")

    def test_coord_to_tile(self):
        x, y = self.host.world_to_tile_num(0, 50.5, 89.0)
        assert (x, y), (0 == 0)

    def test_tile_to_coord(self):
        wr = self.host.tile_num_to_world_lb_rt(0, 0, 0)
        print wr
        wr = self.host.tile_num_to_world_lb_rt(1, 0, 0)
        print wr
        wr = self.host.tile_num_to_world_lb_rt(1, 0, 1)
        print wr
        wr = self.host.tile_num_to_world_lb_rt(2, 0, 0)
        print wr


if __name__ == "__main__":
    import time
    
    t = TestTileCoordinates()
    t.setup()
    t.test_coord_to_tile()
    t.test_tile_to_coord()
    