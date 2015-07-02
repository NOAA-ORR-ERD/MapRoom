import numpy as np

from command import Command, UndoInfo
from layers.vector_object import *


class DrawVectorObjectCommand(Command):
    short_name = "vector_object"
    ui_name = None
    vector_object_class = None
    serialize_order = [
        ('layer', 'layer'),
        ('cp1', 'point'),
        ('cp2', 'point'),
        ('style', 'style'),
        ]
    
    def __init__(self, event_layer, cp1, cp2, style):
        Command.__init__(self, event_layer)
        self.cp1 = cp1
        self.cp2 = cp2
        self.style = style.get_copy()  # Make sure not sharing objects
    
    def __str__(self):
        return self.ui_name
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        saved_invariant = lm.next_invariant
        layer = self.get_vector_object_layer(lm)
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
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_opposite_corners(self.cp1, self.cp2)
        layer.set_style(self.style)
        return layer

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
    serialize_order = [
        ('layer', 'layer'),
        ('cp1', 'point'),
        ('cp2', 'point'),
        ('style', 'style'),
        ('snapped_layer', 'layer'),
        ('snapped_cp', 'int'),
        ]
    
    def __init__(self, event_layer, cp1, cp2, style, snapped_layer, snapped_cp):
        DrawVectorObjectCommand.__init__(self, event_layer, cp1, cp2, style)
        if snapped_layer is not None:
            self.snapped_layer = snapped_layer.invariant
        else:
            self.snapped_layer = None
        self.snapped_cp = snapped_cp
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_opposite_corners(self.cp1, self.cp2)
        layer.set_style(self.style)
        return layer

    def perform(self, editor):
        undo = DrawVectorObjectCommand.perform(self, editor)
        if self.snapped_layer is not None:
            # Convoluted way to get the layer that isn't returned
            # by DrawVectorObjectCommand.perform.  Calling
            # get_vector_object_layer will return a brand new object!
            layer = undo.flags.layer_flags[0].layer
            
            lm = editor.layer_manager
            sl = lm.get_layer_by_invariant(self.snapped_layer)
            print "sl", sl
            print "snapped_cp", self.snapped_cp
            lm.set_control_point_link(layer, 1, sl, self.snapped_cp)
            lm.update_linked_control_points(sl, undo.flags)
        return undo


class DrawPolylineCommand(DrawVectorObjectCommand):
    short_name = "polyline_obj"
    ui_name = "Polyline"
    vector_object_class = PolylineObject
    serialize_order = [
        ('layer', 'layer'),
        ('points', 'points'),
        ('style', 'style'),
        ]
    
    def __init__(self, event_layer, points, style):
        Command.__init__(self, event_layer)
        self.points = points
        self.style = style.get_copy()  # Make sure not sharing objects
        self.style.fill_style = 0
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_points(self.points)
        layer.set_style(self.style)
        return layer


class AddTextCommand(DrawVectorObjectCommand):
    short_name = "text_obj"
    ui_name = "Create Text"
    vector_object_class = OverlayTextObject
    serialize_order = [
        ('layer', 'layer'),
        ('point', 'point'),
        ('style', 'style'),
        ]
    
    def __init__(self, event_layer, point, style):
        Command.__init__(self, event_layer)
        self.point = point
        self.style = style.get_copy()  # Make sure not sharing objects
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_location(self.point)
        layer.set_style(self.style)
        return layer


class AddIconCommand(DrawVectorObjectCommand):
    short_name = "icon_obj"
    ui_name = "Create Text"
    vector_object_class = OverlayImageObject
    serialize_order = [
        ('layer', 'layer'),
        ('point', 'point'),
        ('style', 'style'),
        ]
    
    def __init__(self, event_layer, point, style):
        Command.__init__(self, event_layer)
        self.point = point
        self.style = style.get_copy()  # Make sure not sharing objects
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_location(self.point)
        layer.set_style(self.style)
        return layer
