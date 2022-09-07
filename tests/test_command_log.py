import os

import numpy as np

from mock import *

from maproom.command import *
from maproom.mouse_commands import *
from maproom.menu_commands import *

class TestLogBase(object):
    logfile = None

    #### BROKEN TEST ####
    # these tests fail with
    # self = <maproom.menu_commands.LoadLayersCommand object at 0x16a684b50>, editor = <maproom.mock.MockProject object at 0x16a58d2e0>

#     def perform(self, editor):
#         print("*******\nperform called on:", self)
#         print("loader:", self.loader)
# >       self.undo_info = undo = self.loader.load_layers_from_uri(self.uri, editor.layer_manager)
# E       AttributeError: 'str' object has no attribute 'load_layers_from_uri'

    # I *think* this is because the logfile reader:
    # CommandLogLoader
    # is getting a uri (filename) and mimetype from the logfile, then passing those off
    # to:
    # class LoadLayersCommand(Command):
    #     short_name = "load"
    #     serialize_order = [
    #         ('uri', 'text'),
    #         ('loader', 'loader'),
    #     ]

    #     def __init__(self, uri, loader):
    #         Command.__init__(self)
    #         self.uri = uri
    #         self.loader = loader

    # which is expecting a Loader object, rather than a string mimetype.

    # so there should be code int ehre to convert the mime type to a loader, but
    # I'm not sure where that belongs, or if the logfile for the test is out of date.

    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager
        self.project.load_file(self.logfile, "application/x-maproom-command-log")

class TestBasic(TestLogBase):
    logfile = "../TestData/CommandLog/verdat1.mrc"

    def test_points(self):
        layer = self.manager.get_nth_oldest_layer_of_type("line", 1)
        assert 692 == len(layer.points)

class TestPoints(TestLogBase):
    logfile = "../TestData/CommandLog/pt-line-del-line_to.mrc"

    def test_points(self):
        layer = self.manager.get_nth_oldest_layer_of_type("line", 1)
        assert 698 == len(layer.points)

class TestTwoLayers(TestLogBase):
    logfile = "../TestData/CommandLog/two_layers.mrc"

    def test_points(self):
        lm = self.manager
        layer1 = lm.get_nth_oldest_layer_of_type("line", 1)
        assert 693 == len(layer1.points)
        layer2 = lm.get_nth_oldest_layer_of_type("line", 2)
        assert 26 == len(layer2.points)

    def test_rename(self):
        lm = self.manager
        layer1 = lm.get_nth_oldest_layer_of_type("line", 1)
        layer2 = lm.get_nth_oldest_layer_of_type("line", 2)
        assert "26 points" == layer2.name
        self.project.undo(8)
        assert "000026pts.verdat" == layer2.name
        self.project.redo()
        assert "26 points" == layer2.name

    def test_delete(self):
        lm = self.manager
        saved_invariant = lm.next_invariant
        layer1 = lm.get_nth_oldest_layer_of_type("line", 1)
        layer2 = lm.get_nth_oldest_layer_of_type("line", 2)
        cmd = DeleteLayerCommand(layer2)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        assert saved_invariant - 1 == lm.next_invariant
        self.project.undo()
        assert saved_invariant == lm.next_invariant

    def test_merge(self):
        lm = self.manager
        saved_invariant = lm.next_invariant
        layer1 = lm.get_nth_oldest_layer_of_type("line", 1)
        layer2 = lm.get_nth_oldest_layer_of_type("line", 2)
        cmd = MergeLayersCommand(layer1, layer2, layer1.depth_unit)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        assert saved_invariant + 1 == lm.next_invariant
        self.project.undo()
        assert saved_invariant == lm.next_invariant
        self.project.redo()
        assert saved_invariant + 1 == lm.next_invariant
        layer3 = lm.get_nth_oldest_layer_of_type("line", 3)
        assert 719 == len(layer3.points)

    def test_triangulate1(self):
        lm = self.manager
        layer1 = lm.get_nth_oldest_layer_of_type("scale", 1)
        cmd = TriangulateLayerCommand(layer1, None, None)
        undo = self.project.process_command(cmd)
        assert not undo.flags.success

    def test_triangulate2(self):
        lm = self.manager
        saved_invariant = lm.next_invariant
        layer2 = lm.get_nth_oldest_layer_of_type("line", 2)
        cmd = TriangulateLayerCommand(layer2, None, None)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        assert saved_invariant + 1 == lm.next_invariant
        self.project.undo()
        assert saved_invariant == lm.next_invariant
        self.project.redo()
        assert saved_invariant + 1 == lm.next_invariant

class TestInvariantOffset(object):
    def setup(self):
        self.project = MockProject()
        self.manager = self.project.layer_manager
        self.project.load_file("../TestData/Verdat/000011pts.verdat", "application/x-maproom-verdat")

    def test_offset1(self):
        self.project.load_file("../TestData/CommandLog/two_layers.mrc", "application/x-maproom-command-log")
        lm = self.manager
        layer = lm.get_nth_oldest_layer_of_type("line", 1)
        assert 11 == len(layer.points)
        layer = lm.get_nth_oldest_layer_of_type("line", 2)
        assert 693 == len(layer.points)
        layer = lm.get_nth_oldest_layer_of_type("line", 3)
        assert 26 == len(layer.points)

    def test_offset2(self):
        self.project.load_file("../TestData/CommandLog/two_layers.mrc", "application/x-maproom-command-log")
        self.project.load_file("../TestData/CommandLog/two_layers.mrc", "application/x-maproom-command-log")
        lm = self.manager
        layer = lm.get_nth_oldest_layer_of_type("line", 1)
        assert 11 == len(layer.points)
        layer = lm.get_nth_oldest_layer_of_type("line", 2)
        assert 693 == len(layer.points)
        layer = lm.get_nth_oldest_layer_of_type("line", 3)
        assert 26 == len(layer.points)
        cmd = DeleteLayerCommand(layer)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        layer = lm.get_nth_oldest_layer_of_type("line", 3)
        assert 693 == len(layer.points)
        layer = lm.get_nth_oldest_layer_of_type("line", 4)
        assert 26 == len(layer.points)

if __name__ == "__main__":
    import time

    t = TestInvariantOffset()
    t.setup()
    t.test_offset1()
    t.setup()
    t.test_offset2()
