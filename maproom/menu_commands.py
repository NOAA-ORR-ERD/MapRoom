import numpy as np

from peppy2.framework.errors import ProgressCancelError
from peppy2.utils.file_guess import FileMetadata

from command import Command, UndoInfo
from layers import loaders, Grid, LineLayer, TriangleLayer, AnnotationLayer

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
        saved_invariant = lm.next_invariant
        loader = loaders.get_loader(self.metadata)
        try:
            progress_log.info("START=Loading %s" % self.metadata.uri)
            layers = loader.load_layers(self.metadata, manager=lm)
        except ProgressCancelError, e:
            undo.flags.success = False
            undo.errors = [e.message]
        except IOError, e:
            undo.flags.success = False
            undo.errors = [str(e)]
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
            undo.data = (layers, saved_invariant)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layers, saved_invariant = self.undo_info.data
        
        for layer in layers:
            lm.remove_layer(layer)
        lm.next_invariant = saved_invariant
        
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
        saved_invariant = lm.next_invariant
        if self.type == "grid":
            layer = Grid(manager=lm)
        elif self.type == "triangle":
            layer = TriangleLayer(manager=lm)
        elif self.type == "annotation":
            layer = AnnotationLayer(manager=lm)
        else:
            layer = LineLayer(manager=lm)
        
        layer.new()
        lm.insert_loaded_layer(layer, editor, self.before, self.after)
        
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

class RenameLayerCommand(Command):
    short_name = "rename_layer"
    serialize_order =  [
            ('layer', 'layer'),
            ('name', 'string'),
            ]

    def __init__(self, layer, name):
        Command.__init__(self, layer)
        self.name = name
    
    def __str__(self):
        return "Rename Layer to %s" % self.name
    
    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.name = next_command.name
                return True
    
    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.data = (layer.name,)
        
        layer.name = self.name
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_metadata_changed = True
        
        return self.undo_info

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        name, = self.undo_info.data
        layer.name = name
        return self.undo_info

class DeleteLayerCommand(Command):
    short_name = "del_layer"
    serialize_order =  [
            ('layer', 'layer'),
            ]

    def __init__(self, layer):
        Command.__init__(self, layer)
        self.name = layer.name
    
    def __str__(self):
        return "Delete Layer %s" % self.name
    
    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        insertion_index = lm.get_multi_index_of_layer(layer)
        undo.data = (layer, insertion_index, layer.invariant)
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = lm.roll_back_invariant(layer.invariant)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layer, insertion_index, saved_invariant = self.undo_info.data
        lm.insert_layer(insertion_index, layer, invariant=saved_invariant)
        return self.undo_info

class MergeLayersCommand(Command):
    short_name = "merge_layers"
    serialize_order =  [
            ('layer_a', 'layer'),
            ('layer_b', 'layer'),
            ]
    
    def __init__(self, layer_a, layer_b):
        Command.__init__(self)
        self.layer_a = layer_a.invariant
        self.name_a = str(layer_a.name)
        self.layer_b = layer_b.invariant
        self.name_b = str(layer_b.name)
    
    def __str__(self):
        return "Merge Layers %s & %s" % (self.name_a, self.name_b)
    
    def perform(self, editor):
        lm = editor.layer_manager
        saved_invariant = lm.next_invariant
        self.undo_info = undo = UndoInfo()
        layer_a = lm.get_layer_by_invariant(self.layer_a)
        layer_b = lm.get_layer_by_invariant(self.layer_b)
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        
        layer = layer_a.merge_layer_into_new(layer_b)
        lm.insert_layer(None, layer)
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        lf.layer_loaded = True

        undo.data = (layer.invariant, saved_invariant)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        invariant, saved_invariant = self.undo_info.data
        layer = lm.get_layer_by_invariant(invariant)
        insertion_index = lm.get_multi_index_of_layer(layer)
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = saved_invariant
        
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
        self.name = layer.name
        self.q = q
        self.a = a
    
    def __str__(self):
        return "Triangulate Layer %s" % self.name
    
    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        saved_invariant = lm.next_invariant
        self.undo_info = undo = UndoInfo()
        t_layer = TriangleLayer(manager=lm)
        try:
            progress_log.info("START=Triangulating layer %s" % layer.name)
            t_layer.triangulate_from_layer(layer, self.q, self.a)
        except ProgressCancelError, e:
            self.undo_info.flags.success = False
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            progress_log.info("END")
            self.undo_info.flags.success = False
            layer.highlight_exception(e)
            editor.window.error(e.message, "Triangulate Error")
        finally:
            progress_log.info("END")

        if self.undo_info.flags.success:
            t_layer.name = "Triangulated %s" % layer.name
            old_t_layer = lm.find_dependent_layer(layer, t_layer.type)
            if old_t_layer is not None:
                invariant = old_t_layer.invariant
                lm.remove_layer(old_t_layer)
            else:
                invariant = None
            lm.insert_loaded_layer(t_layer, editor, after=layer, invariant=invariant)
            t_layer.set_dependent_of(layer)
            
            undo.flags.layers_changed = True
            undo.flags.refresh_needed = True
            lf = undo.flags.add_layer_flags(t_layer)
            lf.select_layer = True
            lf.layer_loaded = True

            undo.data = (t_layer, old_t_layer, invariant, saved_invariant)
        
        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        t_layer, old_t_layer, invariant, saved_invariant = self.undo_info.data
        
        insertion_index = lm.get_multi_index_of_layer(t_layer)
        lm.remove_layer_at_multi_index(insertion_index)
        if old_t_layer is not None:
            lm.insert_layer(insertion_index, old_t_layer, invariant=invariant)
        lm.next_invariant = saved_invariant
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(old_t_layer)
        lf.select_layer = True
        return undo
