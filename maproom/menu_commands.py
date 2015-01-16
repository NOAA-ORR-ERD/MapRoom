import numpy as np

from peppy2.framework.errors import ProgressCancelError

from command import Command, UndoInfo
from layers import Grid, LineLayer, TriangleLayer

import logging
progress_log = logging.getLogger("progress")


class AddLayerCommand(Command):
    serialize_params =  [
            ('type', 'string'),
            ]
    
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
    serialize_order =  [
            ('layer', 'layer'),
            ]
    
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
    serialize_order =  [
            ('layer_a', 'layer'),
            ('layer_b', 'layer'),
            ]
    
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

class TriangulateLayerCommand(Command):
    serialize_order =  [
            ('layer', 'layer'),
            ('q', 'float'),
            ('a', 'float'),
            ]

    def __init__(self, layer, q, a):
        Command.__init__(self, layer)
        self.q = q
        self.a = a
    
    def __str__(self):
        return "Triangulate Layer %s" % self.layer.name
    
    def perform(self, editor):
        lm = editor.layer_manager
        self.undo_info = undo = UndoInfo()
        t_layer = TriangleLayer(manager=lm)
        try:
            progress_log.info("START=Triangulating layer %s" % self.layer.name)
            t_layer.triangulate_from_layer(self.layer, self.q, self.a)
        except ProgressCancelError, e:
            self.undo_info.flags.success = False
        except Exception as e:
            progress_log.info("END")
            print traceback.format_exc(e)
            self.layer.highlight_exception(e)
            editor.window.error(e.message, "Triangulate Error")
        finally:
            progress_log.info("END")

        if self.undo_info.flags.success:
            t_layer.name = "Triangulated %s" % self.layer.name
            old_t_layer = lm.find_dependent_layer(self.layer, "triangles")
            if old_t_layer is not None:
                lm.remove_layer(old_t_layer)
            lm.insert_loaded_layer(t_layer, editor, after=self.layer)
            lm.set_dependent_layer(self.layer, "triangles", t_layer)
                
            undo.flags.layers_changed = True
            undo.flags.refresh_needed = True
            undo.flags.select_layer = t_layer
            undo.flags.layer_loaded = t_layer

            undo.data = (t_layer, old_t_layer)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        t_layer, old_t_layer = self.undo_info.data
        
        insertion_index = lm.get_multi_index_of_layer(t_layer)
        lm.remove_layer_at_multi_index(insertion_index)
        if old_t_layer is not None:
            lm.insert_layer(insertion_index, old_t_layer)
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        undo.flags.select_layer = old_t_layer
        return undo
