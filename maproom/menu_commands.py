import json


from omnivore.framework.errors import ProgressCancelError

from command import Command, UndoInfo
from layers import AnnotationLayer
from layers import CompassRose
from layers import EmptyLayer
from layers import Grid
from layers import LineEditLayer
from layers import LineLayer
from layers import PolygonLayer
from layers import TileLayer
from layers import TriangleLayer
from layers import WMSLayer
from layers import RNCLoaderLayer
from layers import loaders
from vector_object_commands import get_parent_layer_data
from vector_object_commands import restore_layers

import logging
progress_log = logging.getLogger("progress")


class LoadLayersCommand(Command):
    short_name = "load"
    serialize_order = [
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
        if hasattr(loader, "load_query"):
            loader.load_query(self.metadata, lm)
        try:
            progress_log.info("START=Loading %s" % self.metadata.uri)
            layers = loader.load_layers(self.metadata, manager=lm)
        except ProgressCancelError, e:
            undo.flags.success = False
            undo.flags.errors = [e.message]
        except IOError, e:
            undo.flags.success = False
            undo.flags.errors = [str(e)]
        finally:
            progress_log.info("END")

        if not undo.flags.success:
            return undo

        if layers is None:
            undo.flags.success = False
            undo.flags.errors = ["Unknown file type %s for %s" % (self.metadata.mime, self.metadata.uri)]
        else:
            errors = []
            warnings = []
            for layer in layers:
                if layer.load_error_string != "":
                    errors.append(layer.load_error_string)
                if layer.load_warning_string != "":
                    warnings.append(layer.load_warning_string)
            if errors:
                undo.flags.success = False
                undo.flags.errors = errors
            if warnings:
                undo.flags.message = warnings

        if undo.flags.success:
            lm.add_layers(layers, False, editor)
            for layer in layers:
                lf = undo.flags.add_layer_flags(layer)
                lf.select_layer = True
                lf.layer_loaded = True

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
    serialize_order = [
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
        elif self.type == "compass_rose":
            layer = CompassRose(manager=lm)
        elif self.type == "triangle":
            layer = TriangleLayer(manager=lm)
        elif self.type == "annotation":
            layer = AnnotationLayer(manager=lm)
        elif self.type == "wms":
            layer = WMSLayer(manager=lm)
        elif self.type == "tile":
            layer = TileLayer(manager=lm)
        elif self.type == "rnc":
            layer = RNCLoaderLayer(manager=lm)
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


class PasteLayerCommand(Command):
    short_name = "paste_layer"
    serialize_order = [
        ('layer', 'layer'),
        ('json_text', 'string'),
        ('center', 'point'),
    ]

    def __init__(self, layer, json_text, center):
        Command.__init__(self, layer)
        self.json_text = json_text
        self.center = center

    def __str__(self):
        return "Paste Layer"

    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        lm = editor.layer_manager
        before = lm.get_layer_by_invariant(self.layer)
        saved_invariant = lm.next_invariant
        json_data = json.loads(self.json_text)
        mi = lm.get_insertion_multi_index(before)
        old_invariant_map = {}
        layer = lm.insert_json(json_data, editor, mi, old_invariant_map)
        layer.name = "Copy of %s" % layer.name

        drag = layer.center_point_index
        x = layer.points.x[drag]
        y = layer.points.y[drag]
        layer.move_control_point(drag, drag, self.center[0] - x, self.center[1] - y)
        print "AFTER NEW LAYER POSITION"
        layer.update_bounds()

        new_links = []
        for old_invariant, new_layer in old_invariant_map.iteritems():
            for dep_cp, truth_invariant, truth_cp, locked in new_layer.control_point_links:
                # see if new truth layer has changed because it's also in the
                # copied group
                if truth_invariant in old_invariant_map:
                    truth_invariant = old_invariant_map[truth_invariant].invariant
                new_links.append((new_layer, dep_cp))
                lm.set_control_point_link((new_layer.invariant, dep_cp), (truth_invariant, truth_cp), locked=locked)

        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True

        affected = layer.parents_affected_by_move()
        for parent in affected:
            print "AFFECTED!", parent
            lf = undo.flags.add_layer_flags(parent)
            lf.layer_items_moved = True
        undo.data = (layer.invariant, saved_invariant, new_links)

        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        invariant, saved_invariant, cp_links = self.undo_info.data
        layer = editor.layer_manager.get_layer_by_invariant(invariant)
        insertion_index = lm.get_multi_index_of_layer(layer)

        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = saved_invariant

        for layer, cp in cp_links:
            lm.remove_control_point_links(layer, cp, force=True)

        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        return undo


class RenameLayerCommand(Command):
    short_name = "rename_layer"
    serialize_order = [
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
    serialize_order = [
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
        children = lm.get_children(layer)
        parents = layer.parents_affected_by_move()
        links = lm.remove_all_links_to_layer(layer)
        for child in children:
            links.extend(lm.remove_all_links_to_layer(child))
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        new_selected_layer = parents[-1]
        lf = undo.flags.add_layer_flags(new_selected_layer)
        lf.select_layer = True

        # Only remove the reference to the layer in the layer manager, leave
        # all the layer info around so that it can be undone
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = lm.roll_back_invariant(layer.invariant)

        parent_layer_data = get_parent_layer_data(parents, undo)

        undo.data = (layer, insertion_index, layer.invariant, children, links, parent_layer_data)

        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layer, insertion_index, saved_invariant, children, links, parent_layer_data = self.undo_info.data
        lm.insert_layer(insertion_index, layer, invariant=saved_invariant)
        lm.insert_children(layer, children)
        lm.restore_all_links_to_layer(layer, links)
        restore_layers(editor, parent_layer_data)
        lf = self.undo_info.flags.add_layer_flags(layer)
        lf.select_layer = True
        return self.undo_info


class MergeLayersCommand(Command):
    short_name = "merge_layers"
    serialize_order = [
        ('layer_a', 'layer'),
        ('layer_b', 'layer'),
        ('depth_unit', 'string'),
    ]

    def __init__(self, layer_a, layer_b, depth_unit):
        Command.__init__(self)
        self.layer_a = layer_a.invariant
        self.name_a = str(layer_a.name)
        self.layer_b = layer_b.invariant
        self.name_b = str(layer_b.name)
        self.depth_unit = depth_unit

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

        layer = layer_a.merge_layer_into_new(layer_b, self.depth_unit)
        if layer is None:
            undo.flags.success = False
            undo.flags.errors = ["Incompatible layer types for merge"]
        else:
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


class MoveLayerCommand(Command):
    short_name = "move_layer"
    serialize_order = [
        ('moved_layer', 'layer'),
        ('target_layer', 'layer'),
        ('before', 'bool'),
        ('in_folder', 'bool'),
    ]

    def __init__(self, moved_layer, target_layer, before, in_folder):
        Command.__init__(self)
        self.moved_layer = moved_layer.invariant
        self.name = str(moved_layer.name)
        self.target_layer = target_layer.invariant
        self.before = before
        self.in_folder = in_folder

    def __str__(self):
        return "Move Layer %s" % (self.name)

    def perform(self, editor):
        lm = editor.layer_manager
        self.undo_info = undo = UndoInfo()
        source_layer = lm.get_layer_by_invariant(self.moved_layer)
        target_layer = lm.get_layer_by_invariant(self.target_layer)
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(source_layer)
        lf.select_layer = True
        lf.metadata_changed = True

        mi_source = lm.get_multi_index_of_layer(source_layer)
        mi_target = lm.get_multi_index_of_layer(target_layer)

        # here we "re-get" the source layer so that it's replaced by a
        # placeholder and temporarily removed from the tree
        temp_layer = EmptyLayer(layer_manager=lm)
        source_layer, children = lm.replace_layer(mi_source, temp_layer)

        # if we are inserting onto a folder, insert as the second item in the folder
        # (the first item in the folder is the folder pseudo-layer)
        if (target_layer.is_root()):
            mi_target = [1]
        elif target_layer.is_folder() and self.in_folder:
            mi_target.append(1)
        else:
            if not self.before:
                mi_target[-1] = mi_target[-1] + 1
        lm.insert_layer(mi_target, source_layer, invariant=self.moved_layer)
        mi_temp = lm.get_multi_index_of_layer(temp_layer)
        lm.remove_layer(temp_layer)
        lm.insert_children(source_layer, children)

        undo.data = (mi_temp, )

        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        mi_temp, = self.undo_info.data
        temp_layer = EmptyLayer(layer_manager=lm)
        lm.insert_layer(mi_temp, temp_layer)

        source_layer = lm.get_layer_by_invariant(self.moved_layer)
        children = lm.get_children(source_layer)
        lm.remove_layer(source_layer)
        mi_temp = lm.get_multi_index_of_layer(temp_layer)
        lm.replace_layer(mi_temp, source_layer)
        lm.insert_children(source_layer, children)

        return self.undo_info


class TriangulateLayerCommand(Command):
    short_name = "triangulate"
    serialize_order = [
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
            self.undo_info.flags.errors = [e.message]
            layer.highlight_exception(e)
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


class ToPolygonLayerCommand(Command):
    short_name = "to_polygon"
    serialize_order = [
        ('layer', 'layer'),
    ]

    def __init__(self, layer):
        Command.__init__(self, layer)
        self.name = layer.name

    def __str__(self):
        return "Polygon Layer from %s" % self.name

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        saved_invariant = lm.next_invariant
        self.undo_info = undo = UndoInfo()
        p = PolygonLayer(manager=lm)
        try:
            progress_log.info("START=Boundary to polygon layer %s" % layer.name)
            boundaries = layer.get_all_boundaries()
        except ProgressCancelError, e:
            self.undo_info.flags.success = False
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            progress_log.info("END")
            self.undo_info.flags.success = False
            layer.highlight_exception(e)
            editor.window.error(e.message, "Boundary Error")
        finally:
            progress_log.info("END")

        if self.undo_info.flags.success:
            p.set_data_from_boundaries(boundaries)
            p.name = "Polygons from %s" % layer.name
            lm.insert_loaded_layer(p, editor, after=layer)

            undo.flags.layers_changed = True
            undo.flags.refresh_needed = True
            lf = undo.flags.add_layer_flags(p)
            lf.select_layer = True
            lf.layer_loaded = True

            undo.data = (p, p.invariant, saved_invariant)

        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        p, invariant, saved_invariant = self.undo_info.data

        insertion_index = lm.get_multi_index_of_layer(p)
        lm.remove_layer_at_multi_index(insertion_index)
        lm.next_invariant = saved_invariant

        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        layer = lm.get_layer_by_invariant(self.layer)
        lf = undo.flags.add_layer_flags(layer)
        lf.select_layer = True
        return undo


class ToVerdatLayerCommand(ToPolygonLayerCommand):
    short_name = "to_verdat"
    serialize_order = [
        ('layer', 'layer'),
    ]

    def __init__(self, layer):
        Command.__init__(self, layer)
        self.name = layer.name

    def __str__(self):
        return "Line Layer from %s" % self.name

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        saved_invariant = lm.next_invariant
        self.undo_info = undo = UndoInfo()
        p = LineLayer(manager=lm)
        points, segments = layer.get_points_lines()
        p.set_data(points, 0, segments)
        p.name = "Verdat from %s" % layer.name
        lm.insert_loaded_layer(p, editor, after=layer)

        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(p)
        lf.select_layer = True
        lf.layer_loaded = True

        undo.data = (p, p.invariant, saved_invariant)

        return self.undo_info


class PolygonEditLayerCommand(Command):
    short_name = "polygon_edit"
    serialize_order = [
        ('layer', 'layer'),
        ('obj_type', 'int'),
        ('obj_index', 'int'),
    ]

    def __init__(self, layer, obj_type, obj_index):
        Command.__init__(self, layer)
        self.obj_type = obj_type
        self.obj_index = obj_index
        self.name = layer.name

    def __str__(self):
        return "Editing Polygon from %s" % self.name

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        p = LineEditLayer(manager=lm, parent_layer=layer, object_type=self.obj_type, object_index=self.obj_index)

        # arbitrarily choose sub_index 0 and ring_index 0; we only need to get
        # the shapely geometry object
        geom, ident = layer.get_geometry_from_object_index(self.obj_index, 0, 0)

        p.set_data_from_geometry(geom)
        p.name = "%d %d Editing Polygon from %s" % (self.obj_type, self.obj_index, layer.name)
        old_layer, old_insertion_index = lm.replace_transient_layer(p, editor, after=layer)
        insertion_index = lm.get_multi_index_of_layer(p)

        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        lf = undo.flags.add_layer_flags(p)
        lf.select_layer = True
        lf.layer_loaded = True

        undo.data = (p, insertion_index, old_layer, old_insertion_index)

        return self.undo_info

    def undo(self, editor):
        lm = editor.layer_manager
        layer, insertion_index, old_layer, old_insertion_index = self.undo_info.data
        lm.remove_layer_at_multi_index(insertion_index)
        undo = UndoInfo()
        undo.flags.layers_changed = True
        undo.flags.refresh_needed = True
        if old_layer:
            lm.insert_layer(old_insertion_index, old_layer)
            lf = undo.flags.add_layer_flags(old_layer)
            lf.select_layer = True
        return undo


class SavepointCommand(Command):
    short_name = "savepoint"
    serialize_order = [
        ('layer', 'layer'),
        ('world_rect', 'rect'),
    ]

    def __init__(self, layer, world_rect):
        Command.__init__(self, layer)
        self.world_rect = world_rect

    def __str__(self):
        return "Save Point"

    def perform(self, editor):
        lm = editor.layer_manager
        layer = lm.get_layer_by_invariant(self.layer)
        self.undo_info = undo = UndoInfo()
        if layer is not None:
            lf = undo.flags.add_layer_flags(layer)
            lf.select_layer = True

        return self.undo_info

    def undo(self, editor):
        undo = UndoInfo()
        undo.flags.refresh_needed = True
        return undo
