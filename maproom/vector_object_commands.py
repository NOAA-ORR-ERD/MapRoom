import numpy as np

from command import Command, UndoInfo
from layers.vector_object import *


def update_parent_bounds(layer, undo):
    affected = layer.parents_affected_by_move()
    parent_layer_data = get_parent_layer_data(affected, undo)
    layer.update_bounds()
    return parent_layer_data

def get_parent_layer_data(affected, undo):
    parent_layer_data = []
    for layer in affected:
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        points = layer.copy_points()
        parent_layer_data.append((layer.invariant, points, layer.copy_bounds()))
    return parent_layer_data

def restore_layers(editor, old_layer_data, undo=None):
    lm = editor.layer_manager
    for invariant, old_points, old_bounds in old_layer_data:
        layer = lm.get_layer_by_invariant(invariant)
        layer.points = old_points
        layer.bounds = old_bounds
        if undo:
            lf = undo.flags.add_layer_flags(layer)
            lf.layer_items_moved = True


class MoveControlPointCommand(Command):
    short_name = "move_cpt"
    serialize_order =  [
        ('layer', 'layer'),
        ('drag', 'int'),
        ('anchor', 'int'),
        ('dx', 'float'),
        ('dy', 'float'),
        ('snapped_layer', 'layer'),
        ('snapped_cp', 'int'),
        ]
    
    def __init__(self, layer, drag, anchor, dx, dy, snapped_layer, snapped_cp):
        Command.__init__(self, layer)
        self.drag = drag
        self.anchor = anchor
        self.dx = dx
        self.dy = dy
        if snapped_layer is not None:
            self.snapped_layer = snapped_layer.invariant
        else:
            self.snapped_layer = None
        self.snapped_cp = snapped_cp
    
    def __str__(self):
        return "Move Control Point #%d" % self.drag
    
    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and next_command.drag == self.drag and next_command.anchor == self.anchor:
                self.dx += next_command.dx
                self.dy += next_command.dy
                self.snapped_layer = next_command.snapped_layer
                self.snapped_cp = next_command.snapped_cp
                return True
    
    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        old_links = layer.remove_from_master_control_points(self.drag, self.anchor)
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        affected = layer.children_affected_by_move()
        child_layer_data = []
        for la in affected:
            lf = undo.flags.add_layer_flags(la)
            lf.layer_items_moved = True
            child_layer_data.append((la.invariant, la.copy_points(), la.copy_bounds()))
        
        layer.move_control_point(self.drag, self.anchor, self.dx, self.dy)
        
        parent_layer_data = update_parent_bounds(layer, undo)

        undo.data = (old_links, child_layer_data, parent_layer_data)

        if self.snapped_layer is not None:
            sl = lm.get_layer_by_invariant(self.snapped_layer)
            #print "sl", sl
            #print "snapped_cp", self.snapped_cp
            lm.set_control_point_link(layer, self.drag, sl, self.snapped_cp)
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        (old_links, child_layer_data, parent_layer_data) = self.undo_info.data
        child_layer_data.reverse()
        
        restore_layers(editor, child_layer_data)
        restore_layers(editor, parent_layer_data)
        
        layer = lm.get_layer_by_invariant(self.layer)
        links = layer.remove_from_master_control_points(self.drag, self.anchor)
        for dep, master in old_links:
            lm.set_control_point_link(dep, master)
        return self.undo_info


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
        
        parent_layer_data = update_parent_bounds(layer, undo)
        
        undo.data = (layer.invariant, saved_invariant, parent_layer_data)
        
        self.perform_post(editor, lm, layer, undo)
        
        return self.undo_info
    
    def perform_post(self, editor, lm, layer, undo):
        pass

    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_opposite_corners(self.cp1, self.cp2)
        layer.set_style(self.style)
        return layer

    def undo(self, editor):
        lm = editor.layer_manager
        invariant, saved_invariant, parent_layer_data = self.undo_info.data
        layer = editor.layer_manager.get_layer_by_invariant(invariant)
        insertion_index = lm.get_multi_index_of_layer(layer)
        
        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = saved_invariant
        
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        
        self.undo_post(editor, lm, layer, undo)
        
        restore_layers(editor, parent_layer_data, undo)

        return undo

    def undo_post(self, editor, lm, layer, undo):
        pass


class DrawRectangleCommand(DrawVectorObjectCommand):
    short_name = "rect_obj"
    ui_name = "Rectangle"
    vector_object_class = RectangleVectorObject


class DrawEllipseCommand(DrawVectorObjectCommand):
    short_name = "ellipse_obj"
    ui_name = "Ellipse"
    vector_object_class = EllipseVectorObject


class DrawCircleCommand(DrawVectorObjectCommand):
    short_name = "circle_obj"
    ui_name = "Circle"
    vector_object_class = CircleVectorObject

    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_center_and_radius(self.cp1, self.cp2)
        layer.set_style(self.style)
        return layer


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

    def perform_post(self, editor, lm, layer, undo):
        if self.snapped_layer is not None:
            sl = lm.get_layer_by_invariant(self.snapped_layer)
            print "sl", sl
            print "snapped_cp", self.snapped_cp
            # The control point is always 1 because it's only possible to snap
            # to the endpoint
            lm.set_control_point_link(layer, 1, sl, self.snapped_cp)

    def undo_post(self, editor, lm, layer, undo):
        if self.snapped_layer is not None:
            # Since the drag point is always the end, the anchor point is
            # always the beginning, i.e.  0
            layer.remove_from_master_control_points(1, 0)


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
        self.init_style(style)
    
    def init_style(self, style):
        self.style = style.get_copy()  # Make sure not sharing objects
        self.style.fill_style = 0  # Turn off fill by default because it's a polyLINE
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_points(self.points)
        layer.set_style(self.style)
        return layer


class DrawPolygonCommand(DrawPolylineCommand):
    short_name = "polygon_obj"
    ui_name = "Polygon"
    vector_object_class = PolygonObject
    
    def init_style(self, style):
        self.style = style.get_copy()  # Make sure not sharing objects


class AddTextCommand(DrawVectorObjectCommand):
    short_name = "text_obj"
    ui_name = "Create Text"
    vector_object_class = OverlayTextObject
    serialize_order = [
        ('layer', 'layer'),
        ('point', 'point'),
        ('style', 'style'),
        ('screen_width', 'int'),
        ('screen_height', 'int'),
        ]
    
    def __init__(self, event_layer, point, style, screen_width, screen_height):
        Command.__init__(self, event_layer)
        self.point = point
        self.style = style.get_copy()  # Make sure not sharing objects
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_location_and_size(self.point, self.screen_width, self.screen_height)
        layer.set_style(self.style)
        return layer


class AddIconCommand(DrawVectorObjectCommand):
    short_name = "icon_obj"
    ui_name = "Create Text"
    vector_object_class = OverlayIconObject
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
        layer.set_location_and_size(self.point, 32, 32)
        layer.set_style(self.style)
        return layer
