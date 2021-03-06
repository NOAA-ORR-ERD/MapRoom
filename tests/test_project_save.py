import os

import numpy as np

from mock import *

from maproom.command import *
from maproom.mouse_commands import *
from maproom.menu_commands import *
from maproom.layers import *


def compare_layer_managers(lm1, lm2):
    for layer in lm1.flatten():
        try:
            layer.manager
        except AttributeError:
            continue
        print(layer)
        mi = lm1.get_multi_index_of_layer(layer)
        print(mi)
        other = lm2.get_layer_by_multi_index(mi)
        print(other)
        try:
            other.manager
        except AttributeError:
            continue
        print(("%s: %s" % (mi, layer.test_contents_equal(other))))


class TestBasic(object):
    logfile = "../TestData/CommandLog/verdat1.mrc"
    
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager

    def save_setup(self):
        lm = self.manager
        
        self.project.load_file(self.logfile, "application/x-maproom-command-log")

        a1 = AnnotationLayer(manager=lm)
        lm.insert_layer([3], a1)
        
        a = OverlayTextObject(manager=lm)
        a.set_location((6.6637485204,-1.40163099748))
        lm.insert_layer([3, 1], a)
        
        a = RectangleVectorObject(manager=lm)
        a.set_opposite_corners(
            (-16.6637485204,-1.40163099748),
            (9.65688930428,-19.545688433))
        lm.insert_layer([3, 2], a)
        
        a = PolylineObject(manager=lm)
        a.set_points([
            (-15,-2),
            (5, -8),
            (10, -20),
            (8, -5),
            (-17, -10),
            ])
        a.style.fill_style = 0
        lm.insert_layer([3, 3], a)
        
        a2 = AnnotationLayer(manager=lm)
        lm.insert_layer([4], a2)
        return a1, a2

    def test_save_without_bounds(self):
        a1, a2 = self.save_setup()
        self.manager.save_all_zip("test.mrpz")

    def test_save_with_bounds(self):
        a1, a2 = self.save_setup()
        a1.update_bounds()
        a2.update_bounds()
        self.manager.save_all_zip("test.mrpz")

class TestBasicLoad(object):
    def setup(self):
        self.zip_project = MockProject()
    
    def test_load(self):
        self.zip_project.load_file("test.mrpz", "application/x-maproom-project-zip")
        zip_manager = self.zip_project.layer_manager
        print(zip_manager)

class TestImageLayer(object):
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager

        lm = self.manager
        self.project.load_file("../TestData/ChartsAndImages/12205_7.KAP", "image/x-gdal")

        a1 = AnnotationLayer(manager=lm)
        lm.insert_layer([3], a1)
        
        a = OverlayTextObject(manager=lm)
        a.set_location((6.6637485204,-1.40163099748))
        lm.insert_layer([3, 1], a)

        self.manager.save_all_zip("test_image.maproom")

    def test_load_image(self):
        self.zip_project = MockProject()
        self.zip_project.load_file("test_image.maproom", "application/x-maproom-project-zip")
        zip_manager = self.zip_project.layer_manager

class TestTileLayer(object):
    project_file = "a.mrp"
    
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager
    
    def test_save(self):
        lm = self.manager
        t = TileLayer(manager=lm)
        t.map_server_id = 5  # arbitrary
        lm.insert_layer([3], t)
        lm.save_all_zip(self.project_file)
    
    def test_load(self):
        self.project.load_file(self.project_file, "application/x-maproom-project-zip")
        t = self.project.layer_manager.get_nth_oldest_layer_of_type("tiles", 1)
        assert t.map_server_id == 5


if __name__ == "__main__":
    import time
    
    # t = TestBasic()
    # t.setup()
    # t.test_save_with_bounds()
    
    # t = TestBasicLoad()
    # t.setup()
    # t.test_load()

    t = TestImageLayer()
    t.setup()
    t.test_load_image()

