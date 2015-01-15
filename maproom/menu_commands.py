import numpy as np

from command import Command, UndoInfo
from layers import Grid, LineLayer, TriangleLayer


class AddLayerCommand(Command):
    def __init__(self, type, before=None, after=None):
        Command.__init__(self)
        self.type = type
        self.before = before
        self.after = after
    
    def __str__(self):
        return "Add %s Layer" % self.type.title()
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        if self.type == "grid":
            layer = Grid(manager=lm)
        elif self.type == "triangle":
            layer = TriangleLayer(manager=lm)
        else:
            layer = LineLayer(manager=lm)
        
        layer.new()
        lm.insert_loaded_layer(layer, editor, self.before, self.after)
        
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        undo.flags.select_layer = layer
        undo.flags.layer_loaded = layer
        undo.data = (layer,)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layer, = self.undo_info.data
        insertion_index = lm.get_multi_index_of_layer(layer)
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        return undo

class DeleteLayerCommand(Command):
    def __init__(self, layer):
        Command.__init__(self)
        self.layer = layer
    
    def __str__(self):
        return "Delete Layer %s" % self.layer.name
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        insertion_index = lm.get_multi_index_of_layer(self.layer)
        undo.data = (insertion_index,)
        undo.flags.layers_changed = True
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        insertion_index, = self.undo_info.data
        lm.insert_layer(insertion_index, self.layer)
        return self.undo_info

class MergeLayersCommand(Command):
    def __init__(self, layer_a, layer_b):
        Command.__init__(self)
        self.layer_a = layer_a
        self.layer_b = layer_b
    
    def __str__(self):
        return "Merge Layers %s & %s" % (self.layer_a.name, self.layer_b.name)
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        
        layer = self.layer_a.merge_layer_into_new(self.layer_b)
        lm.insert_layer(None, layer)
        undo.flags.select_layer = layer
        undo.flags.layer_loaded = layer

        undo.data = (layer,)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layer, = self.undo_info.data
        insertion_index = lm.get_multi_index_of_layer(layer)
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        return undo
