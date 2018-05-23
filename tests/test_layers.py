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

    def test_sub_layer_delete(self):
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
        
        a2 = AnnotationLayer(manager=lm)
        lm.insert_layer([3, 3], a2)
        
        a = PolylineObject(manager=lm)
        a.set_points([
            (-15,-2),
            (5, -8),
            (10, -20),
            (8, -5),
            (-17, -10),
            ])
        a.style.fill_style = 0
        lm.insert_layer([3, 3, 1], a)
        
        # Calculate bounds of the annotation layers to set up their
        # points/lines arrays
        a2.update_bounds()
        a1.update_bounds()
        
        print(lm)
        mi = lm.get_multi_index_of_layer(a1)
        print("mi", mi)
        
        assert 7 == lm.next_invariant
        print(lm)
        
        cmd = DeleteLayerCommand(a1)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        
        print(lm)
        assert 7 == lm.next_invariant
        self.project.undo()
        assert 7 == lm.next_invariant
        print(lm)
        
        # remove last layer and see if invariant changes
        cmd = DeleteLayerCommand(a)
        undo = self.project.process_command(cmd)
        assert undo.flags.success
        
        print(lm)
        assert 6 == lm.next_invariant
        self.project.undo()
        assert 7 == lm.next_invariant
        print(lm)



if __name__ == "__main__":
    import time
    
    t = TestBasic()
    t.setup()
    t.test_sub_layer_delete()
