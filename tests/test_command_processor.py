import os

import unittest
from nose.tools import *

import numpy as np

from mock import *

from maproom.command import *
from maproom.mouse_commands import *

class TestVerdatUndo(object):
    def setup(self):
        self.manager = MockManager()
        self.layer = self.manager.load_first_layer("../TestData/Verdat/000689pts.verdat", "application/x-maproom-verdat")

    def test_add_points(self):
        world_point = (-118.0, 33.0)
        orig_points = np.copy(self.layer.points)
        eq_(689, np.alen(self.layer.points))
        
        cmd = InsertPointCommand(self.layer, world_point)
        self.manager.project.process_command(cmd)
        eq_(690, np.alen(self.layer.points))
        
        self.manager.project.undo()
        eq_(689, np.alen(self.layer.points))
        assert np.array_equal(orig_points, self.layer.points)

    def test_move_points(self):
        dx = -.5
        dy = 1.5
        indexes = [400, 410, 499]
        orig_points = np.copy(self.layer.points).view(np.recarray)
        
        cmd = MovePointsCommand(self.layer, indexes, dx, dy)
        self.manager.project.process_command(cmd)
        assert np.allclose(self.layer.points.x[indexes], orig_points.x[indexes] + dx)
        assert np.allclose(self.layer.points.y[indexes], orig_points.y[indexes] + dy)

        self.manager.project.undo()
        eq_(689, np.alen(self.layer.points))
        assert np.array_equal(orig_points, self.layer.points)

    def test_move_points_coelesce(self):
        dx = -.5
        dy = 1.5
        indexes = [400, 410, 499]
        orig_points = np.copy(self.layer.points).view(np.recarray)
        
        # move once
        cmd = MovePointsCommand(self.layer, indexes, dx, dy)
        self.manager.project.process_command(cmd)
        assert np.allclose(self.layer.points.x[indexes], orig_points.x[indexes] + dx)
        assert np.allclose(self.layer.points.y[indexes], orig_points.y[indexes] + dy)
        
        # move same points again
        cmd = MovePointsCommand(self.layer, indexes, dx, dy)
        self.manager.project.process_command(cmd)
        assert np.allclose(self.layer.points.x[indexes], orig_points.x[indexes] + dx + dx)
        assert np.allclose(self.layer.points.y[indexes], orig_points.y[indexes] + dy + dy)

        # single undo should return to original state
        self.manager.project.undo()
        eq_(689, np.alen(self.layer.points))
        assert np.array_equal(orig_points, self.layer.points)

if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestVerdatUndo()
    t.setup()
    t.test_add_points()
    t.test_move_points()
    t.test_move_points_coelesce()
