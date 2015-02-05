import os

import unittest
from nose.tools import *

import numpy as np

from mock import *

from maproom.layer_manager import LayerManager
from maproom.command import *
from maproom.mouse_commands import *
from maproom.menu_commands import *

class TestLogBase(object):
    logfile = None
    
    def setup(self):
        self.project = MockProject()
        self.manager = LayerManager.create(self.project)
        self.project.layer_manager = self.manager
        self.project.load_file(self.logfile, "application/x-maproom-command-log")

class TestBasic(TestLogBase):
    logfile = "../TestData/CommandLog/verdat1.mrc"

    def test_points(self):
        layer = self.manager.get_layer_by_name("000689pts.verdat")
        print layer
        eq_(692, np.alen(layer.points))

class TestPoints(TestLogBase):
    logfile = "../TestData/CommandLog/pt-line-del-line_to.mrc"

    def test_points(self):
        layer = self.manager.get_layer_by_name("000689pts.verdat")
        print layer
        eq_(698, np.alen(layer.points))

if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestPoints()
    t.setup()
    t.test_points()
