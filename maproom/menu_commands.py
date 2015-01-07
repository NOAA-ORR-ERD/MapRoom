import numpy as np

from command import Command, UndoInfo

class DeleteLayerCommand(Command):
    def __init__(self, layer):
        self.layer = layer
        self.undo_info = None
    
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
