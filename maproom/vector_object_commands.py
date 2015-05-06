import numpy as np

from command import Command, UndoInfo
from layers.vector_object import *


class DrawVectorObjectCommand(Command):
    short_name = "vector_object"
    serialize_order = [
        ('layer', 'layer'),
        ('cp1', 'point'),
        ('cp2', 'point'),
        ]
    ui_name = None
    vector_object_class = None
    
    def __init__(self, event_layer, cp1, cp2):
        Command.__init__(self, event_layer)
        self.cp1 = cp1
        self.cp2 = cp2
    
    def __str__(self):
        return self.ui_name
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        saved_invariant = lm.next_invariant
        layer = self.vector_object_class(manager=lm)
        layer.set_opposite_corners(self.cp1, self.cp2)
        event_layer = lm.get_layer_by_invariant(self.layer)
        if event_layer.type == "annotation":
            kwargs = {'first_child_of': event_layer}
        else:
            parent_layer = lm.get_folder_of_layer(event_layer)
            if parent_layer is not None:
                kwargs = {'first_child_of': parent_layer}
            else:
                kwargs = {'first_child_of': event_layer}
        lm.insert_loaded_layer(layer, editor, **kwargs)
        
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


class DrawRectangleCommand(DrawVectorObjectCommand):
    short_name = "rect_obj"
    ui_name = "Rectangle"
    vector_object_class = RectangleVectorObject


class DrawEllipseCommand(DrawVectorObjectCommand):
    short_name = "ellipse_obj"
    ui_name = "Ellipse"
    vector_object_class = EllipseVectorObject


class DrawLineCommand(DrawVectorObjectCommand):
    short_name = "line_obj"
    ui_name = "Line"
    vector_object_class = LineVectorObject
