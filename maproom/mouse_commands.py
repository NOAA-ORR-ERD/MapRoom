import numpy as np

from .command import Command, UndoInfo

import logging
log = logging.getLogger(__name__)


class ViewportCommand(Command):
    short_name = "viewport"
    serialize_order = [
        ('layer', 'layer'),
        ('center', 'point'),
        ('units_per_pixel', 'float'),
        ('regime', 'int'),
    ]

    def __init__(self, layer, center=None, units_per_pixel=None, regime=0):
        Command.__init__(self, layer)
        self.center = center
        self.units_per_pixel = units_per_pixel
        self.regime = regime

    def __str__(self):
        return "Viewport Change"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.center = next_command.center
                self.units_per_pixel = next_command.units_per_pixel
                return True

    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        c = editor.layer_canvas
        undo.data = (c.projected_point_center, c.projected_units_per_pixel)
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        if layer is not None:
            center, units_per_pixel = c.calc_zoom_to_world_rect(layer.bounds)
        elif self.center is not None:
            center, units_per_pixel = self.center, self.units_per_pixel

        # Only shift the viewport if requesting the 0-360 regime AND the center
        # point is in the -360 - 0 range. It's possible that the center is
        # already in the 0 - 360 range.
        if self.regime > 0:
            shifted = c.get_world_point_from_projected_point(center)
            if shifted[0] < 0:
                shifted = (shifted[0] + self.regime, shifted[1])
                center = c.get_projected_point_from_world_point(shifted)
        c.set_viewport(center, units_per_pixel)
        undo.flags.immediate_refresh_needed = True

        # reset command params so autosave file set zoom without recalculation
        self.layer = None
        self.center = center
        self.units_per_pixel = units_per_pixel
        return undo

    def undo(self, editor):
        (old_center, old_units_per_pixel) = self.undo_info.data
        editor.layer_canvas.set_viewport(old_center, old_units_per_pixel)
        return self.undo_info


class InsertPointCommand(Command):
    short_name = "pt"
    serialize_order = [
        ('layer', 'layer'),
        ('world_point', 'point'),
    ]

    def __init__(self, layer, world_point):
        Command.__init__(self, layer)
        self.world_point = world_point

    def __str__(self):
        try:
            return "Add Point #%d" % self.undo_info.index
        except AttributeError:
            return "Add Point"

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = layer.insert_point(self.world_point)
        layer.select_point(self.undo_info.index)
        lf = undo.flags.add_layer_flags(layer)
        lf.hidden_layer_check = True
        return undo

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        undo_info = layer.delete_point(self.undo_info.index)
        return undo_info


class MovePointsCommand(Command):
    short_name = "move_pt"
    serialize_order = [
        ('layer', 'layer'),
        ('indexes', 'list_int'),
        ('dx', 'float'),
        ('dy', 'float'),
    ]

    def __init__(self, layer, indexes, dx, dy):
        Command.__init__(self, layer)
        self.indexes = indexes
        self.dx = dx
        self.dy = dy

    def __str__(self):
        if len(self.indexes) == 1:
            return "Move Point #%d" % self.indexes[0]
        return "Move %d Points" % len(self.indexes)

    def transient_geometry_update(self, tlayer):
        new_points = tlayer.get_new_points_after_move(self.indexes)
        tlayer.parent_layer.rebuild_geometry_from_points(tlayer.object_type, tlayer.object_index, new_points)

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and np.array_equal(next_command.indexes, self.indexes):
                self.dx += next_command.dx
                self.dy += next_command.dy
                return True

    def is_recordable(self):
        return len(self.indexes) > 0

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        old_x = np.copy(layer.points.x[self.indexes])
        old_y = np.copy(layer.points.y[self.indexes])
        undo.data = (old_x, old_y)
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.indexes_of_points_affected = self.indexes
        lf.layer_contents_added = True
        layer.points.x[self.indexes] += self.dx
        layer.points.y[self.indexes] += self.dy
        return undo

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        (old_x, old_y) = self.undo_info.data
        layer.points.x[self.indexes] = old_x
        layer.points.y[self.indexes] = old_y
        return self.undo_info


class NormalizeLongitudeCommand(Command):
    short_name = "norm_long"
    serialize_order = [
        ('layer', 'layer'),
    ]

    def __str__(self):
        return "Normalize Longitude"

    def recurse_perform(self, layer, undo):
        layer_undo = layer.get_undo_info()
        layer.normalize_longitude()
        undo.data.append((layer.invariant, layer_undo))
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        for child in layer.manager.get_layer_children(layer):
            self.recurse_perform(child, undo)

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.data = []
        undo.flags.refresh_needed = True
        self.recurse_perform(layer, undo)
        undo.flags.add_layer_flags(layer)
        return undo

    def undo(self, editor):
        for invariant, layer_undo in self.undo_info.data:
            layer = editor.layer_manager.get_layer_by_invariant(invariant)
            layer.restore_undo_info(layer_undo)
        return self.undo_info


class SwapLatLonCommand(Command):
    short_name = "swap_lat_lon"
    serialize_order = [
        ('layer', 'layer'),
    ]

    def __str__(self):
        return "Swap Latitude & Longitude"

    def recurse_perform(self, layer, undo):
        layer_undo = layer.get_undo_info()
        layer.swap_lat_lon()
        undo.data.append((layer.invariant, layer_undo))
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        for child in layer.manager.get_layer_children(layer):
            self.recurse_perform(child, undo)

    def perform(self, editor):
        c = editor.layer_canvas
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.data = [c.projected_point_center, c.projected_units_per_pixel]
        undo.flags.refresh_needed = True
        self.recurse_perform(layer, undo)
        undo.flags.add_layer_flags(layer)
        layer.update_bounds()
        center, units_per_pixel = c.calc_zoom_to_world_rect(layer.bounds)
        c.set_viewport(center, units_per_pixel)
        undo.flags.immediate_refresh_needed = True
        return undo

    def undo(self, editor):
        c = editor.layer_canvas
        center, units_per_pixel = self.undo_info.data[0:2]
        c.set_viewport(center, units_per_pixel)
        for invariant, layer_undo in self.undo_info.data[2:]:
            layer = editor.layer_manager.get_layer_by_invariant(invariant)
            layer.restore_undo_info(layer_undo)
        return self.undo_info


class ChangeDepthCommand(Command):
    short_name = "depth"
    serialize_order = [
        ('layer', 'layer'),
        ('indexes', 'list_int'),
        ('depth', 'float'),
    ]

    def __init__(self, layer, indexes, depth):
        Command.__init__(self, layer)
        self.indexes = indexes
        self.depth = depth

    def __str__(self):
        return "Set Depth to %s" % str(self.depth)

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and np.array_equal(next_command.indexes, self.indexes):
                self.depth = next_command.depth
                return True

    def is_recordable(self):
        return len(self.indexes) > 0

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        old_depths = np.copy(layer.points.z[self.indexes])
        undo.data = old_depths
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        layer.points.z[self.indexes] = self.depth
        return undo

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        (old_depths) = self.undo_info.data
        layer.points.z[self.indexes] = old_depths
        return self.undo_info


class InsertLineCommand(Command):
    short_name = "line_to"
    serialize_order = [
        ('layer', 'layer'),
        ('index', 'int'),
        ('world_point', 'point'),
    ]

    def __init__(self, layer, index, world_point):
        Command.__init__(self, layer)
        self.index = index
        self.world_point = world_point
        self.undo_point = None
        self.undo_line = None

    def __str__(self):
        return "Line From Point %d" % self.index

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_point = layer.insert_point(self.world_point)
        self.undo_line = layer.insert_line_segment(self.undo_point.index, self.index)
        layer.select_point(self.undo_point.index)
        lf = self.undo_point.flags.add_layer_flags(layer)
        lf.hidden_layer_check = True
        # FIXME: merge undo status
        return self.undo_point

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        # FIXME: merge undo status
        undo_info = layer.delete_line_segment(self.undo_line.index)
        undo_info = layer.delete_point(self.undo_point.index)
        return undo_info


class ConnectPointsCommand(Command):
    short_name = "line"
    serialize_order = [
        ('layer', 'layer'),
        ('index1', 'int'),
        ('index2', 'int'),
    ]

    def __init__(self, layer, index1, index2):
        Command.__init__(self, layer)
        self.index1 = index1
        self.index2 = index2
        self.undo_line = None

    def __str__(self):
        return "Line Connecting Points %d & %d" % (self.index1, self.index2)

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_line = layer.insert_line_segment(self.index1, self.index2)
        layer.select_point(self.index2)
        lf = self.undo_line.flags.add_layer_flags(layer)
        lf.hidden_layer_check = True
        return self.undo_line

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        undo_info = layer.delete_line_segment(self.undo_line.index)
        return undo_info


class SplitLineCommand(Command):
    short_name = "split"
    serialize_order = [
        ('layer', 'layer'),
        ('index', 'int'),
        ('world_point', 'point'),
    ]

    def __init__(self, layer, index, world_point):
        Command.__init__(self, layer)
        self.index = index
        self.world_point = world_point
        self.undo_point = None
        self.undo_delete = None
        self.undo_line1 = None
        self.undo_line2 = None
        self.point_index_1 = None
        self.point_index_2 = None

    def __str__(self):
        return "Split Line #%d" % self.index

    def transient_geometry_update(self, tlayer):
        new_points = tlayer.get_new_points_after_insert(self.point_index_1, self.point_index_2, self.undo_point.index)
        tlayer.parent_layer.rebuild_geometry_from_points(tlayer.object_type, tlayer.object_index, new_points)
        tlayer.rebuild_from_parent_layer()
        tlayer.select_nearest_point(self.world_point)

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_point = layer.insert_point(self.world_point)

        layer.select_point(self.undo_point.index)
        self.point_index_1 = layer.line_segment_indexes.point1[self.index]
        self.point_index_2 = layer.line_segment_indexes.point2[self.index]
        color = layer.line_segment_indexes.color[self.index]
        state = layer.line_segment_indexes.state[self.index]
        depth = (layer.points.z[self.point_index_1] + layer.points.z[self.point_index_2]) / 2
        layer.points.z[self.undo_point.index] = depth
        self.undo_delete = layer.delete_line_segment(self.index)
        self.undo_line1 = layer.insert_line_segment_at_index(len(layer.line_segment_indexes), self.point_index_1, self.undo_point.index, color, state)
        self.undo_line2 = layer.insert_line_segment_at_index(len(layer.line_segment_indexes), self.undo_point.index, self.point_index_2, color, state)

        lf = self.undo_point.flags.add_layer_flags(layer)
        lf.hidden_layer_check = True
        lf.layer_items_moved = True
        lf.layer_contents_added = True
        return self.undo_point

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        # FIXME: merge undo status
        undo_info = layer.delete_line_segment(self.undo_line2.index)
        undo_info = layer.delete_line_segment(self.undo_line1.index)
        layer.line_segment_indexes = np.insert(layer.line_segment_indexes, self.index, self.undo_delete.data).view(np.recarray)
        undo_info = layer.delete_point(self.undo_point.index)
        return undo_info


class DeleteLinesCommand(Command):
    short_name = "del"
    serialize_order = [
        ('layer', 'layer'),
        ('point_indexes', 'list_int'),
        ('line_indexes', 'list_int'),
    ]

    def __init__(self, layer, point_indexes, line_indexes=None):
        Command.__init__(self, layer)
        self.point_indexes = point_indexes
        self.line_indexes = line_indexes
        self.undo_point = None
        self.undo_line = None

    def __str__(self):
        try:
            old_points, old_line_segments, old_line_indexes = self.undo_info.data
        except AttributeError:
            return "Delete Points/Lines"
        if len(old_line_indexes) == 0:
            if len(self.point_indexes) == 1:
                return "Delete Point #%d" % self.point_indexes[0]
            return "Delete %d Points" % len(self.point_indexes)
        elif len(old_line_indexes) == 1:
            if len(self.line_indexes) > 0:
                line = self.line_indexes[0]
            else:
                line = old_line_indexes[0]
            return "Delete Line #%d" % line
        return "Delete %d Lines" % len(old_line_indexes)

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        old_line_indexes = layer.get_lines_connected_to_points(self.point_indexes)
        if self.line_indexes is not None:
            # handle degenerate list as well as zero-length numpy array using length test
            if len(list(self.line_indexes)) > 0:
                old_line_indexes = np.unique(np.append(old_line_indexes, self.line_indexes))
        old_line_segments = np.copy(layer.line_segment_indexes[old_line_indexes])
        old_points = np.copy(layer.points[self.point_indexes])
        undo.data = (old_points, old_line_segments, old_line_indexes)
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_deleted = True
        layer.remove_points_and_lines(self.point_indexes, old_line_indexes)
        return undo

    def undo(self, editor):
        """
        Using the batch numpy.insert, it expects the point indexes to be
        relative to the current state of the array, not the original indexes.

        >>> a=np.arange(10)
        >>> a
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> indexes=[2,5,7,9]
        >>> b=np.delete(a,indexes,0)
        >>> b
        array([0, 1, 3, 4, 6, 8])
        >>> np.insert(b, indexes, indexes)
        IndexError: index 12 is out of bounds for axis 1 with size 10
        >>> fixed = indexes - np.arange(4)
        >>> np.insert(b, fixed, indexes)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        old_points, old_line_segments, old_line_indexes = self.undo_info.data
        offset = np.arange(len(self.point_indexes))
        indexes = self.point_indexes - offset
        layer.points = np.insert(layer.points, indexes, old_points).view(np.recarray)

        # adjust existing indexes to allow for inserted points
        offsets1 = np.zeros(np.alen(layer.line_segment_indexes)).astype(np.uint32)
        offsets2 = np.zeros(np.alen(layer.line_segment_indexes)).astype(np.uint32)
        for index in indexes:
            offsets1 += np.where(layer.line_segment_indexes.point1 >= index, 1, 0).astype(np.uint32)
            offsets2 += np.where(layer.line_segment_indexes.point2 >= index, 1, 0).astype(np.uint32)
        layer.line_segment_indexes.point1 += offsets1
        layer.line_segment_indexes.point2 += offsets2

        offset = np.arange(len(old_line_indexes))
        indexes = old_line_indexes - offset
        layer.line_segment_indexes = np.insert(layer.line_segment_indexes, indexes, old_line_segments).view(np.recarray)
        undo = UndoInfo()
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_items_moved = True
        lf.layer_contents_deleted = True
        return undo


class MergePointsCommand(DeleteLinesCommand):
    short_name = "merge_pt"
    serialize_order = [
        ('layer', 'layer'),
        ('point_indexes', 'list_int'),
    ]

    def __init__(self, layer, point_indexes):
        DeleteLinesCommand.__init__(self, layer, point_indexes, None)

    def __str__(self):
        return "Merge Points"


class CropRectCommand(Command):
    short_name = "crop"
    serialize_order = [
        ('layer', 'layer'),
        ('world_rect', 'rect'),
    ]

    def __init__(self, layer, world_rect):
        Command.__init__(self, layer)
        self.world_rect = world_rect

    def __str__(self):
        return "Crop"

    def perform(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        self.undo_info = layer.crop_rectangle(self.world_rect)
        return self.undo_info

    def undo(self, editor):
        layer = editor.layer_manager.get_layer_by_invariant(self.layer)
        old_state = self.undo_info.data
        undo_info = layer.set_state(old_state)
        return undo_info


class StyleChangeCommand(Command):
    short_name = "style"
    serialize_order = [
        ('layer', 'layer'),
        ('style', 'style'),
    ]

    def __init__(self, layer, style):
        Command.__init__(self, layer)
        self.style = style

    def __str__(self):
        return "Layer Style"

    def can_coalesce(self, next_command):
        return next_command.style.has_same_keywords(self.style)

    def coalesce_merge(self, next_command):
        self.style = next_command.style

    def perform_on_layer(self, editor, layer, lm, lf):
        lf.layer_display_properties_changed = True
        layer_undo_info = (layer.style.get_copy(), self.style)
        layer.set_style(self.style)
        return layer_undo_info

    def perform_on_parent(self, editor, layer, lm, lf):
        saved_style = lm.get_default_style_for(layer)
        lm.update_default_style_for(layer, self.style)
        layer_undo_info = self.perform_on_layer(editor, layer, lm, lf)
        return (saved_style, layer_undo_info)

    def undo_on_layer(self, editor, layer, lm, layer_undo_info):
        old_style, style = layer_undo_info
        layer.set_style(old_style)

    def undo_on_parent(self, editor, layer, lm, parent_undo_info):
        saved_style, layer_undo_info = parent_undo_info
        self.undo_on_layer(editor, layer, lm, layer_undo_info)
        lm.update_default_style_for(layer, saved_style)


class TextCommand(Command):
    short_name = "text"
    serialize_order = [
        ('layer', 'layer'),
        ('text', 'text'),
    ]

    def __init__(self, layer, text):
        Command.__init__(self, layer)
        self.text = text

    def __str__(self):
        return "Edit Text"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.text = next_command.text
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.data = (layer.user_text, self.text)
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_display_properties_changed = True
        layer.user_text = self.text
        layer.rebuild_needed = True
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        old_text, text = self.undo_info.data
        layer.user_text = old_text
        layer.rebuild_needed = True
        return self.undo_info


class SetAnchorCommand(Command):
    short_name = "text_anchor"
    serialize_order = [
        ('layer', 'layer'),
        ('anchor', 'int'),
    ]

    def __init__(self, layer, anchor):
        Command.__init__(self, layer)
        self.anchor = anchor

    def __str__(self):
        return "Set Anchor Point"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.anchor = next_command.anchor
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        all_links = lm.get_all_control_point_links_copy()
        linked = lm.remove_control_point_links(layer, layer.anchor_point_index)
        log.debug("old linked: %s" % linked)
        undo.data = (layer.anchor_point_index, self.anchor, all_links)
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_display_properties_changed = True
        layer.set_anchor_index(self.anchor)
        for other in linked:
            (dep, dep_cp), (truth, truth_cp), locked = other
            lm.set_control_point_link((self.layer, self.anchor), (truth, truth_cp))
        layer.rebuild_needed = True
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        old_anchor, anchor, all_links = self.undo_info.data
        layer.set_anchor_index(old_anchor)
        lm.restore_all_control_point_links(all_links)
        layer.rebuild_needed = True
        return self.undo_info


class StatusCodeColorCommand(Command):
    short_name = "status_code_color"
    serialize_order = [
        ('layers', 'layers'),
        ('code', 'int'),
        ('color', 'int'),
    ]

    def __init__(self, layers, code, color):
        Command.__init__(self)
        self.layers = [layer.invariant for layer in layers]
        self.code = code
        self.color = color

    def __str__(self):
        return "Status Code Color"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if set(next_command.layers) == set(self.layers) and next_command.code == self.code:
                self.color = next_command.color
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        self.undo_info = undo = UndoInfo()
        undo.data = []
        for invariant in self.layers:
            layer = lm.get_layer_by_invariant(invariant)
            lf = undo.flags.add_layer_flags(layer)
            lf.layer_display_properties_changed = True
            undo.data.append((layer.invariant, np.copy(layer.points.color), dict(layer.status_code_colors)))
            layer.set_status_code_color(self.code, self.color)
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        for invariant, colors, status_code_colors in self.undo_info.data:
            layer = lm.get_layer_by_invariant(invariant)
            layer.points.color = colors
            layer.status_code_colors = status_code_colors
        return self.undo_info


class BorderWidthCommand(Command):
    short_name = "border_width"
    serialize_order = [
        ('layer', 'layer'),
        ('width', 'int'),
    ]

    def __init__(self, layer, width):
        Command.__init__(self, layer)
        self.width = width

    def __str__(self):
        return "Set Anchor Point"

    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer:
                self.width = next_command.width
                return True

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        undo.data = (layer.border_width,)
        lf = undo.flags.add_layer_flags(layer)
        lf.layer_display_properties_changed = True
        layer.set_border_width(self.width)
        layer.rebuild_needed = True
        return undo

    def undo(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        old_width, = self.undo_info.data
        layer.set_border_width(old_width)
        layer.rebuild_needed = True
        return self.undo_info
