import numpy as np

from peppy2.framework.errors import ProgressCancelError
from peppy2.utils.file_guess import FileMetadata

from command import Command, UndoInfo
from layers import loaders, Grid, LineLayer, TriangleLayer

import logging
progress_log = logging.getLogger("progress")


class LoadLayersCommand(Command):
    short_name = "load"
    serialize_order =  [
            ('metadata', 'file_metadata'),
            ]
    
    def __init__(self, metadata):
        Command.__init__(self)
        self.metadata = metadata
    
    def __str__(self):
        return "Load Layers From %s" % self.metadata.uri
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        loader = loaders.get_loader(self.metadata)
        try:
            progress_log.info("START=Loading %s" % self.metadata.uri)
            layers = loader.load_layers(self.metadata, manager=lm)
        except ProgressCancelError, e:
            undo.flags.success = False
            undo.errors = [e.message]
        finally:
            progress_log.info("END")
        
        if not undo.flags.success:
            return undo
        
        if layers is None:
            undo.flags.success = False
            undo.errors = ["Unknown file type %s for %s" % (metadata.mime, metadata.uri)]
        else:
            errors = []
            for layer in layers:
                if layer.load_error_string != "":
                    errors.append(layer.load_error_string)
            if errors:
                undo.flags.success = False
                undo.errors = errors

        if undo.flags.success:
            lm.add_layers(layers, False, editor)
            for layer in layers:
                lf = undo.flags.add_layer_flags(layer)
                lf.select_layer = True
                lf.layer_loaded = True
                lf.zoom_to_layer = True
                
            undo.flags.layers_changed = True
            undo.flags.refresh_needed = True
            undo.data = (layers,)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layers, = self.undo_info.data
        
        for layer in layers:
            lm.remove_layer(layer)
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        return undo

class AddLayerCommand(Command):
    short_name = "add_layer"
    serialize_order =  [
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
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        lf.layer_loaded = True
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
    short_name = "del_layer"
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
    short_name = "merge_layers"
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
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        lf.layer_loaded = True

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
    short_name = "triangulate"
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
            import traceback
            print traceback.format_exc(e)
            progress_log.info("END")
            self.undo_info.flags.success = False
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
            lf = undo.flags.add_layer_flags(t_layer)
            lf.select_layer = True
            lf.layer_loaded = True

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
        lf = undo.flags.add_layer_flags(old_t_layer)
        lf.select_layer = True
        return undo
