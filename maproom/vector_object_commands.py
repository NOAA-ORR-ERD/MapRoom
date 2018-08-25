
from .command import Command, UndoInfo
from .layers.vector_object import LineVectorObject, RectangleVectorObject, EllipseVectorObject, CircleVectorObject, OverlayTextObject, OverlayIconObject, OverlayLineObject, PolylineObject, PolygonObject, AnnotationLayer, ArrowTextBoxLayer, ArrowTextIconLayer

import logging
log = logging.getLogger(__name__)


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
        parent_layer_data.append((layer.invariant, layer.get_undo_info()))
    return parent_layer_data


def update_linked_layers(lm, layer, undo):
    """Update truth layer control point in response to dependent layer control
    point being moved by the user
    """
    parent_layer_data = []
    for dep_cp, truth_inv, truth_cp, locked in lm.get_control_point_links(layer):
        truth = lm.get_layer_by_invariant(truth_inv)
        lf = undo.flags.add_layer_flags(truth)
        lf.layer_items_moved = True
        parent_layer_data.append((truth_inv, truth.get_undo_info()))
        truth.copy_control_point_from(truth_cp, layer, dep_cp)
        truth.update_bounds()
    return parent_layer_data


def restore_layers(editor, old_layer_data, undo=None):
    lm = editor.layer_manager
    for invariant, undo_info in old_layer_data:
        layer = lm.get_layer_by_invariant(invariant)
        layer.restore_undo_info(undo_info)
        if undo:
            lf = undo.flags.add_layer_flags(layer)
            lf.layer_items_moved = True


class MoveControlPointCommand(Command):
    short_name = "move_cpt"
    serialize_order = [
        ('layer', 'layer'),
        ('drag', 'int'),
        ('anchor', 'int'),
        ('dx', 'float'),
        ('dy', 'float'),
        ('snapped_layer', 'layer'),
        ('snapped_cp', 'int'),
    ]

    def __init__(self, layer, drag, anchor, dx, dy, snapped_layer, snapped_cp, about_center=False):
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
        self.about_center = about_center

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
        undo.flags.immediate_refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        affected = layer.children_affected_by_move()
        child_layer_data = []
        for la in affected:
            lf = undo.flags.add_layer_flags(la)
            lf.layer_items_moved = True
            child_layer_data.append((la.invariant, la.get_undo_info()))

        log.debug("\n\n\nBegin moving control point: layer=%s children=%s" % (layer, str(affected)))
        layer.select_point(self.drag)
        layer.move_control_point(self.drag, self.anchor, self.dx, self.dy, self.about_center)
        linked_layer_data = update_linked_layers(lm, layer, undo)

        parent_layer_data = update_parent_bounds(layer, undo)

        undo.data = (old_links, child_layer_data, linked_layer_data, parent_layer_data)

        if self.snapped_layer is not None:
            sl = lm.get_layer_by_invariant(self.snapped_layer)
            # print "sl", sl
            # print "snapped_cp", self.snapped_cp
            lm.set_control_point_link(layer, self.drag, sl, self.snapped_cp)
        log.debug("Finished moving control point\n\n\n")
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        (old_links, child_layer_data, linked_layer_data, parent_layer_data) = self.undo_info.data
        restore_layers(editor, parent_layer_data)
        restore_layers(editor, linked_layer_data)
        child_layer_data.reverse()
        restore_layers(editor, child_layer_data)

        layer = lm.get_layer_by_invariant(self.layer)
        layer.remove_from_master_control_points(self.drag, self.anchor)
        for dep, master in old_links:
            lm.set_control_point_link(dep, master)
        return self.undo_info


class UnlinkControlPointCommand(Command):
    short_name = "unlink_cpt"
    serialize_order = [
        ('layer', 'layer'),
        ('anchor', 'int'),
    ]

    def __init__(self, layer, anchor):
        Command.__init__(self, layer)
        self.anchor = anchor

    def __str__(self):
        return "Unlink Control Point #%d" % self.anchor

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.flags.refresh_needed = True
        undo.flags.add_layer_flags(layer)
        old_links = layer.remove_from_master_control_points(self.anchor, -1, force=True)
        undo.data = (old_links,)
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        (old_links,) = self.undo_info.data
        for dep, master in old_links:
            lm.set_control_point_link(dep, master)
        return self.undo_info


class RotateObjectCommand(Command):
    short_name = "rotate_obj"
    serialize_order = [
        ('layer', 'layer'),
        ('drag', 'int'),
        ('dx', 'float'),
        ('dy', 'float'),
    ]

    def __init__(self, layer, drag, dx, dy):
        Command.__init__(self, layer)
        self.drag = drag
        self.dx = dx
        self.dy = dy

    def __str__(self):
        return "Rotate Object"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and next_command.drag == self.drag:
                self.dx += next_command.dx
                self.dy += next_command.dy
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        affected = layer.children_affected_by_move()
        child_layer_data = []
        for la in affected:
            lf = undo.flags.add_layer_flags(la)
            lf.layer_items_moved = True
            child_layer_data.append((la.invariant, la.get_undo_info()))

        layer.rotate_point(self.drag, self.dx, self.dy)

        parent_layer_data = update_parent_bounds(layer, undo)

        undo.data = (child_layer_data, parent_layer_data)
        return undo

    def undo(self, editor):
        (child_layer_data, parent_layer_data) = self.undo_info.data
        restore_layers(editor, parent_layer_data)
        child_layer_data.reverse()
        restore_layers(editor, child_layer_data)
        return self.undo_info


class StyledCommand(Command):
    short_name = "_base_styled_object"
    ui_name = None
    vector_object_class = None
    serialize_order = [
        ('layer', 'layer'),
        ('style', 'style'),
    ]

    def __init__(self, event_layer, style=None):
        Command.__init__(self, event_layer)
        if style is not None:
            style = style.get_copy()  # Make sure not sharing objects
        self.style = style


class DrawVectorObjectCommand(StyledCommand):
    short_name = "vector_object"
    ui_name = None
    vector_object_class = None
    serialize_order = [
        ('layer', 'layer'),
        ('cp1', 'point'),
        ('cp2', 'point'),
        ('style', 'style'),
    ]

    def __init__(self, event_layer, cp1, cp2, style=None):
        StyledCommand.__init__(self, event_layer, style)
        self.cp1 = cp1
        self.cp2 = cp2

    def __str__(self):
        return self.ui_name

    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        saved_invariant = lm.next_invariant
        layer = self.get_vector_object_layer(lm)
        event_layer = lm.get_layer_by_invariant(self.layer)
        parent_layer = lm.find_vector_object_insert_layer(event_layer)
        if parent_layer is None:
            undo.flags.refresh_needed = True
            undo.flags.success = False
            undo.flags.errors = ["All annotation layers are grouped. Objects can't be added to grouped layers"]
            return undo

        kwargs = {'first_child_of': parent_layer}
        lm.insert_loaded_layer(layer, editor, **kwargs)

        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        lf.layer_loaded = True
        lf.collapse = False

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


class DrawArrowTextBoxCommand(DrawVectorObjectCommand):
    short_name = "arrow_text_obj"
    ui_name = "Arrow Text Box"
    vector_object_class = ArrowTextBoxLayer

    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_style(self.style)
        return layer

    def perform_post(self, editor, lm, layer, undo):
        layer.grouped = False
        layer.name = self.ui_name
        style = layer.style  # use annotation layer parent style

        halfway = ((self.cp1[0] + self.cp2[0]) / 2.0, (self.cp1[1] + self.cp2[1]) / 2.0)
        line = OverlayLineObject(manager=lm)
        line.set_opposite_corners(self.cp1, halfway)
        line.set_style(style)
        line.style.line_start_marker = 2  # Turn on arrow
        kwargs = {'first_child_of': layer}
        lm.insert_loaded_layer(line, editor, **kwargs)
        lf = undo.flags.add_layer_flags(line)
        lf.layer_loaded = True

        #text = RectangleVectorObject(manager=lm)
        text = OverlayTextObject(manager=lm, show_flagged_anchor_point=False)
        text.set_style(style)
        text.set_opposite_corners(halfway, self.cp2)
        c = editor.layer_canvas
        sp1 = c.get_screen_point_from_world_point(halfway)
        sp2 = c.get_screen_point_from_world_point(self.cp2)
        text.text_width = abs(sp2[0] - sp1[0]) - (2 * text.border_width)
        text.text_height = abs(sp2[1] - sp1[1]) - (2 * text.border_width)
        lm.insert_loaded_layer(text, editor, **kwargs)
        lf = undo.flags.add_layer_flags(text)
        lf.layer_loaded = True

        # The line's control point is always 1 because it's the endpoint,
        # and the text box's control point is zero because it's the one
        # corresponding to the first control point at layer creation time
        text.calc_control_points_from_screen(c)
        cp = text.find_nearest_corner(self.cp1)
        text.anchor_point_index = cp

        # line is now the truth layer; its changes will be forced to the text
        # box
        lm.set_control_point_link(text, cp, line, 1)
        self.undo_post_data = (text, cp)
        self.save_line = line

    def undo_post(self, editor, lm, layer, undo):
        layer, cp = self.undo_post_data
        lm.remove_control_point_links(layer, cp)


class DrawArrowTextIconCommand(DrawArrowTextBoxCommand):
    short_name = "arrow_text_icon_obj"
    ui_name = "Arrow Text Icon"
    vector_object_class = ArrowTextIconLayer

    def perform_post(self, editor, lm, layer, undo):
        DrawArrowTextBoxCommand.perform_post(self, editor, lm, layer, undo)
        icon = OverlayIconObject(manager=lm)
        icon.set_location_and_size(self.cp1, 32, 32)
        icon.set_style(layer.style)
        kwargs = {'first_child_of': layer}
        lm.insert_loaded_layer(icon, editor, **kwargs)
        lf = undo.flags.add_layer_flags(icon)
        lf.layer_loaded = True
        # Set the control point link to the icon. The fixed (anchor) point of
        # the line is control point 0 and the end attached to the text box is
        # control point 1
        lm.set_control_point_link(self.save_line, 0, icon, icon.center_point_index, locked=True)

    def undo_post(self, editor, lm, layer, undo):
        DrawArrowTextBoxCommand.undo_post(self, editor, lm, layer, undo)
        # remove point linked to the icon object.
        lm.remove_control_point_links(self.save_line, 0, force=True)


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
            log.debug("snapped layer: %s, cp=%d" % (sl, self.snapped_cp))
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

    def __init__(self, event_layer, points, style=None):
        StyledCommand.__init__(self, event_layer, style)
        self.points = points

    def check_style(self, layer):
        layer.style.fill_style = 0  # force unfilled because it's a polyLINE

    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_points(self.points)
        layer.set_style(self.style)
        self.check_style(layer)
        return layer


class DrawPolygonCommand(DrawPolylineCommand):
    short_name = "polygon_obj"
    ui_name = "Polygon"
    vector_object_class = PolygonObject

    def check_style(self, style):
        pass  # allow filled/unfilled according to style


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

    def __init__(self, event_layer, point, style=None, screen_width=-1, screen_height=-1):
        StyledCommand.__init__(self, event_layer, style)
        self.point = point
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

    def __init__(self, event_layer, point, style=None):
        StyledCommand.__init__(self, event_layer, style)
        self.point = point

    def get_vector_object_layer(self, lm):
        layer = self.vector_object_class(manager=lm)
        layer.set_location_and_size(self.point, 32, 32)
        layer.set_style(self.style)
        return layer
