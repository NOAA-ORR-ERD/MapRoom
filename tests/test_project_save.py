import os

import numpy as np

from mock import *

from maproom.command import *
from maproom.mouse_commands import *
from maproom.menu_commands import *
from maproom.layers import *

class TestBasic(object):
    logfile = "../TestData/CommandLog/verdat1.mrc"
    
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager

    def test_save(self):
        lm = self.manager
        
        self.project.load_file(self.logfile, "application/x-maproom-command-log")

        a = AnnotationLayer(manager=lm)
        lm.insert_layer([3], a)
        
        a = OverlayTextObject(manager=lm)
        a.set_location((6.6637485204,-1.40163099748))
        a.set_style(lm.default_style)
        lm.insert_layer([3, 1], a)
        
        a = RectangleVectorObject(manager=lm)
        a.set_opposite_corners(
            (-16.6637485204,-1.40163099748),
            (9.65688930428,-19.545688433))
        a.set_style(lm.default_style)
        lm.insert_layer([3, 2], a)
        
        a = PolylineObject(manager=lm)
        a.set_points([
            (-15,-2),
            (5, -8),
            (10, -20),
            (8, -5),
            (-17, -10),
            ])
        a.set_style(lm.default_style)
        a.style.fill_style = 0
        lm.insert_layer([3, 3], a)
        
        a = AnnotationLayer(manager=lm)
        lm.insert_layer([4], a)
        
        lm.save_all("test.mrp")

class TestBasicLoad(object):
    logfile = "test.mrp"
    
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager
    
    def test_load(self):
        self.project.load_file(self.logfile, "application/x-maproom-project-json")


if __name__ == "__main__":
    import time
    
    t = TestBasic()
    t.setup()
    t.test_save()
    
    t = TestBasicLoad()
    t.setup()
    t.test_load()
