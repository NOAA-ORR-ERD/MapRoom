import numpy as np

from command import Command, UndoInfo
from layers.vector_object import *


class RectangleCommand(Command):
    short_name = "rect"
    serialize_order = [
        ('layer', 'layer'),
        ('cp1', 'point'),
        ('cp2', 'point'),
        ]
    
    def __init__(self, parent_layer, cp1, cp2):
        Command.__init__(self, parent_layer)
        self.cp1 = cp1
        self.cp2 = cp2
    
    def __str__(self):
        return "Rectangle"
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        saved_invariant = lm.next_invariant
        layer = RectangleVectorObject(manager=lm)
        layer.set_opposite_corners(self.cp1, self.cp2)
        parent_layer = lm.get_layer_by_invariant(self.layer)
        lm.insert_loaded_layer(layer, editor, after=parent_layer)
        
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        lf.layer_loaded = True
        undo.data = (layer.invariant, saved_invariant)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        invariant, saved_invariant = self.undo_info.data
        layer = editor.layer_manager.get_layer_by_invariant(invariant)
        insertion_index = lm.get_multi_index_of_layer(layer)
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = saved_invariant
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        return undo
