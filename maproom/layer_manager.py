import os
import json
import zipfile
import functools

import wx

from sawx.filesystem import fsopen as open
from sawx.filesystem import filesystem_path
from sawx.events import EventHandler

from .library import rect

from . import layers as ly
from . import loaders
from . import styles
from .command import UndoStack, BatchStatus
from .menu_commands import LoadLayersCommand

# Enthought library imports.
from traits.api import Any
from traits.api import Dict
from traits.api import Event
from traits.api import Int
from traits.api import List

from sawx.document import SawxDocument
from sawx.utils.jsonutil import collapse_json
from sawx.utils.fileutil import ExpandZip
from .library import colormap

import logging
log = logging.getLogger(__name__)


class LayerManager(SawxDocument):

    """
    Manages the layers (a tree of ly.Layer).

    A "multi_index" is a set of indexes taken in order, that give the location
    of a sub-tree or leaf within a tree. For instance,
    in the tree [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ], the multi_index [ 0 ] refers to subtree [ a, b ],
    the multi_index [ 1, 1, 1 ] refers to the leaf e, and the multi_index [ 3, 0 ] refers to the leaf g.

    The first layer in the overall list and in each sublist is assumed to be a "folder" layer, whose only
    purpose at present is to hold the folder name.
    """

    # Transient layer always uses invariant
    transient_invariant = -99

    def __init__(self, file_metadata):
        self.default_styles = styles.copy_default_styles()
        self.layers = []

        self.layer_loaded_event = EventHandler(self)
        self.layers_changed_event = EventHandler(self)
        self.layer_contents_changed_event = EventHandler(self)
        self.layer_contents_changed_in_place_event = EventHandler(self)
        
        # when points are deleted from a layer the indexes of the points in the
        # merge dialog box become invalid; so this event will trigger the
        # user to re-find duplicates in order to create a valid list again
        self.layer_contents_deleted_event = EventHandler(self)

        self.layer_metadata_changed_event = EventHandler(self)
        self.projection_changed_event = EventHandler(self)
        self.refresh_needed_event = EventHandler(self)
        self.background_refresh_needed_event = EventHandler(self)
        self.threaded_image_loaded_event = EventHandler(self)

        # if the project is loaded from a zip file, the ExpandZip object is
        # stored here so the unpacked directory can be referenced when re-
        # saving the project
        self.zip_file_source = None

        # Linked control points are slaves of a truth layer: a dict that maps
        # the dependent layer/control point to the truth layer/control point
        self.control_point_links = {}

        # In order for the serializer to correcly map the layer invariant to
        # the actual layer, the next_invariant must be preset so first user
        # added layer will use 1.  If the number of default layers added below
        # changes, modify next_invariant to match! next_invariant = 1 - (# of
        # calls to insert_layer)
        index = 0
        self.root_invariant = -3
        self.next_invariant = self.root_invariant
        layer = ly.RootLayer(manager=self)
        self.insert_layer([index], layer)

        index += 1
        grid = ly.Timestamp(manager=self)
        self.insert_layer([index], grid)

        index += 1
        grid = ly.Graticule(manager=self)
        self.insert_layer([index], grid)
        
        index += 1
        scale = ly.Scale(manager=self)
        self.insert_layer([index], scale)

        SawxDocument.__init__(self, file_metadata)
        self.undo_stack = UndoStack()  # replace sawx undo stack with our own

        # LayerManagers are *almost* independent of the project. Right now it
        # isn't possible to have multiple views of a project because there are
        # references to the project in a few vector object classes that are
        # positioned relative to the screen. The project will be set once it is
        # added into a ProjectEditor
        self.project = None

        # # Add hook to create layer instances for debugging purposes
        # if "--debug-objects" in self.project.window.application.command_line_args:
        #     from . import debug
        #     debug.debug_objects(self)

    ##### Python special methods

    def __str__(self):
        root = self.get_layer_by_multi_index([0])
        layers = self.get_children(root)
        return str(layers)

    ##### destruction

    def destroy(self):
        # fixme: why do layers need a destroy() method???
        for layer in self.flatten():
            layer.destroy()
        self.layers = []

    def destroy_recursive(self, layer):
        if (layer.is_folder()):
            for item in self.get_layer_children(layer):
                self.destroy_recursive(item)
        self.delete_undo_operations_for_layer(layer)

    ##### Debug functions

    def debug_invariant(self):
        layers = self.flatten()
        print("next invariant: %d" % self.next_invariant)
        for layer in layers:
            print("  %s: invariant=%d" % (layer, layer.invariant))

    def debug_structure(self, indent=""):
        lines = self.debug_structure_recursive(self.layers, indent)
        return ("\n" + indent).join(lines)

    def debug_structure_recursive(self, tree, indent=""):
        result = []

        for item in tree:
            if (isinstance(item, ly.Layer)):
                result.append(indent + str(self.get_multi_index_of_layer(item)) + " " + item.debug_info(indent))
            else:
                result.extend(self.debug_structure_recursive(item, indent + "    "))

        return result

    #### Load

    def load_raw_data(self):
        loader = self.file_metadata["loader"]
        batch_flags = BatchStatus()
        # FIXME: Add load project command that clears all layers
        try:
            loader.load_project  # test
        except AttributeError:
            cmd = LoadLayersCommand(self.uri, loader)
            extra = {'command_from_load': cmd}
            # undo = loader.load_layers_from_uri(self.uri, self)
            # batch_flags = undo.flags
            # extra = {'batch_flags_from_load': batch_flags}
        else:
            extra = loader.load_project(self.uri, self, batch_flags)
            extra['batch_flags_from_load'] = batch_flags
        self.extra_metadata = extra

        # Clear modified flag
        self.undo_stack.set_save_point()
        # self.dirty = self.layer_manager.undo_stack.is_dirty()
        # self.mouse_mode_factory = mouse_handler.PanMode
        # self.view_document(self.document)
    
    def calc_raw_data(self, raw):
        pass

    ##### Serialization

    def get_to_json_attrs(self):
        return [(m[0:-8], getattr(self, m)) for m in dir(self) if hasattr(self, m[0:-8]) and m.endswith("_to_json")]

    def get_from_json_attrs(self):
        return [(m[0:-10], getattr(self, m)) for m in dir(self) if hasattr(self, m[0:-10]) and m.endswith("_from_json")]

    def control_point_links_to_json(self):
        # json can't handle dictionaries with tuples as their keys, so have
        # to compress
        cplist = []
        for entry, (truth, locked) in self.control_point_links.items():
            # retain compatibility with old versions, only add locked flag if
            # present
            if locked:
                cplist.append((entry, truth, locked))
            else:
                cplist.append((entry, truth))
            log.debug(("cplinks:", entry, truth, locked, cplist[-1]))
        return cplist

    def control_point_links_from_json(self, json_data):
        cplist = json_data['control_point_links']
        cpdict = {}
        for item in cplist:
            try:
                entry, truth, locked = item
            except ValueError:
                entry, truth = item
                locked = False
            cpdict[tuple(entry)] = (tuple(truth), locked)
        self.control_point_links = cpdict

    def default_styles_to_json(self):
        return styles.styles_to_json(self.default_styles)

    def default_styles_from_json(self, json_data):
        sdict = json_data['default_styles']
        d = styles.parse_styles_from_json(sdict)
        self.update_default_styles(d)

    ##### Multi-index calculations

    def get_insertion_multi_index(self, before=None, after=None, first_child_of=None, last_child_of=None, background=False, opaque=False, bounded=True):
        if first_child_of is not None:
            mi = self.get_multi_index_of_layer(first_child_of)
            mi.append(1)
        elif last_child_of is not None:
            mi = self.get_multi_index_of_layer(last_child_of)
            children = self.get_layer_children(last_child_of)
            mi.append(len(children) + 1)
        elif before is not None:
            mi = self.get_multi_index_of_layer(before)
        elif after is not None:
            mi = self.get_multi_index_of_layer(after)
            mi[-1] += 1
        else:
            if opaque or background:
                insert_above = None
                for layer in reversed(self.flatten()):  # bottom up
                    if background and layer.background:
                        if opaque and layer.opaque or not opaque:
                            insert_above = layer
                        else:
                            break
                    elif bounded and not layer.bounded:
                        # finite layers go above infinite layers
                        insert_above = layer
                    else:
                        break
                if insert_above is None:
                    mi = [100000000]  # there better be fewer than one hundred million layers...
                else:
                    mi = self.get_multi_index_of_layer(insert_above)
            else:
                mi = None
        return mi

    # def get_multi_index_for_background_layer(self, background, opaque, bounded):
    #     #
    #     insert_above = None
    #     for layer in reversed(self.flatten()):  # bottom up
    #         if background and layer.background:
    #             if opaque and layer.opaque or not opaque:
    #                 insert_above = layer
    #             else:
    #                 break
    #         elif not background and not layer.bounded:
    #             insert_above = layer
    #         else:
    #             break
    #     if insert_above is None:
    #         insert_above = [100000000]  # there better be fewer than one hundred million layers...
    #     else:
    #         insert_above = self.get_multi_index_of_layer(insert_above)
    #     return insert_above

    def get_multi_index_of_layer(self, layer):
        return self.get_multi_index_of_layer_recursive(layer, self.layers)

    def get_multi_index_of_layer_recursive(self, layer, tree):
        for i, item in enumerate(tree):
            if (isinstance(item, ly.Layer)):
                if (item == layer):
                    # in the case of folders, we return the multi-index to the parent,
                    # since the folder "layer" itself is just a pseudo-layer
                    if layer.is_folder():
                        return []
                    else:
                        return [i]
            else:
                result = self.get_multi_index_of_layer_recursive(layer, item)
                if (result is not None):
                    r = [i]
                    r.extend(result)

                    return r

        return None

    def get_multi_index_of_previous_sibling(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        if mi[-1] == 0:  # at root of a subtree
            mi = mi[0:-1]
        mi[-1] = mi[-1] - 1
        return mi

    def get_layer_multi_index_from_file_path(self, file_path):
        for layer in self.flatten():
            if (layer.file_path == file_path):
                return self.get_multi_index_of_layer(layer)
        #
        return None

    ##### Boundary calculations

    def recalc_all_bounds(self):
        # calculate bound starting at leaf layers and working back up to folder
        # layers
        for layer in reversed(self.flatten()):
            layer.update_bounds()

    def recalc_overlay_bounds(self):
        # overlay layers need to adjust their world-space boinding box after a
        # viewport zoom, starting at leaf layers and working back up to folder
        # layers
        affected = []
        for layer in reversed(self.flatten()):
            if layer.is_overlay:
                log.debug("Updating overlay bounds for %s" % str(layer))
                layer.update_overlay_bounds()
            elif layer.contains_overlays:
                log.debug("Updating overlay bounds for %s" % str(layer))
                layer.compute_bounding_rect()
                layer.update_overlay_bounds()
            else:
                continue
            self.layer_contents_changed = layer
            # layer.update_overlay_bounds()
            # self.layer_contents_changed = layer
            affected.append(layer)
            log.debug("  updated to: %s" % str(layer))
        return affected

    def accumulate_layer_bounds(self, layers):
        result = rect.NONE_RECT

        for layer in layers:
            if (result == rect.NONE_RECT):
                result = layer.bounds
            else:
                result = rect.accumulate_rect(result, layer.bounds)

        return result

    ##### Style

    def update_default_styles(self, new_styles):
        self.default_styles = new_styles

        # Make sure "other" is a valid style
        try:
            s = styles.LayerStyle()
            s.parse(str(self.default_styles["other"]))
        except Exception:
            log.warning("Invalid style for other, using default")
            self.default_styles["other"] = styles.LayerStyle() # minimal default styles
        for type_name in sorted(new_styles.keys()):
            style = self.default_styles[type_name]
            log.debug("style %s: %s" % (type_name, str(style)))

    def update_default_style_for(self, layer, style):
        log.debug("updating default style for %s: %s" % (layer.style_name, str(style)))
        self.default_styles[layer.style_name] = style.get_copy()

    def get_default_style_for(self, layer):
        style = self.default_styles.get(layer.style_name, self.default_styles["other"])
        log.debug("get default style for: %s: %s" % (layer.style_name, str(style)))
        return style.get_copy()

    def apply_default_styles(self):
        # for each layer, change the current style to that of the default style
        # for the type of layer
        for layer in self.flatten():
            try:
                style = self.default_styles[layer.style_name]
            except KeyError:
                log.debug("No style for %s" % layer)
                continue
            layer.style = style.get_copy()

    ##### Flatten

    def flatten(self):
        return self.flatten_recursive(self.layers)

    def flatten_recursive(self, tree):
        result = []

        for item in tree:
            if (isinstance(item, ly.Layer)):
                result.append(item)
            else:
                result.extend(self.flatten_recursive(item))

        return result

    def flatten_with_indexes(self):
        return self.flatten_with_indexes_recursive(self.layers, [])

    def flatten_with_indexes_recursive(self, tree, indexes):
        result = []

        index = 0
        indexes = list(indexes)  # copy so recursive doesn't affect parent
        indexes.append(index)
        for item in tree:
            indexes[-1] = index
            if (isinstance(item, ly.Layer)):
                result.append((tuple(indexes), item))
            else:
                result.extend(self.flatten_with_indexes_recursive(item, indexes))
            index += 1

        return result

    ##### Invariant handling

    # invariants are unique identifiers for each layer that don't change when
    # the layer is renamed or reordered

    def get_next_invariant(self, invariant=None):
        if invariant is None:
            invariant = self.next_invariant
        if invariant == self.next_invariant:
            self.next_invariant += 1
        return invariant

    def roll_back_invariant(self, invariant):
        """ Roll back the next_invariant if the supplied invariant is the
        last invariant used.

        This method is used to correctly restore the invariant when the last
        layer is deleted.  If the passed-in invariant doesn't represent the
        last layer added, leave things alone because the invariant of the
        undo/redo will be set to the invariant stored in the undo data.
        """
        if invariant + 1 == self.next_invariant:
            self.next_invariant = invariant
        return self.next_invariant

    def get_invariant_offset(self):
        return self.next_invariant - 1

    ##### Locate layers

    def get_nth_oldest_layer_of_type(self, layer_type, count=1):
        oldest = []
        layers = self.flatten()
        for layer in layers:
            if layer.type == layer_type:
                oldest.append((layer.invariant, layer))
        oldest.sort()  # smaller invariants are inserted before larger ones
        try:
            return oldest[count-1][1]
        except IndexError:
            return None

    def get_layer_by_invariant(self, invariant):
        layers = self.flatten()
        for layer in layers:
            if layer.invariant == invariant:
                return layer
        return None

    def find_default_insert_layer(self):
        # By default, lat/lon layers stay at the top and other layers will
        # be inserted below them.  If the lat/lon layer has been moved down,
        # other layers will be inserted in the first position.
        pos = 0
        for layer in self.layers:
            if isinstance(layer, list):
                layer = layer[0]
            if layer.skip_on_insert:
                pos += 1
            else:
                break
        return [pos]

    def find_vector_object_insert_layer(self, event_layer, vector_object):
        """Find the appropriate layer to insert a vector object, given the
        layer on which the event occurred.

        If the event_layer is an annotation layer it will typically be used,
        *unless* the layer is grouped, in which case it will look
        upwards in the hierarchy to find the first parent annotation layer
        that is not grouped.

        None is returned if the top-level annotation layer is also grouped.
        """
        while not event_layer.is_root():
            if event_layer.is_folder() and event_layer.can_contain_vector_object(vector_object):
                return event_layer
            event_layer = self.get_folder_of_layer(event_layer)
        return None

    def get_folder_of_layer(self, layer):
        """Returns the containing folder of the specified layer

        """
        mi = self.get_multi_index_of_layer(layer)
        mi = mi[:-1]
        mi.append(0)
        return self.get_layer_by_multi_index(mi)

    def get_layer_by_multi_index(self, at_multi_index):
        if (at_multi_index == []):
            return self.layers

        return self.get_layer_by_multi_index_recursive(at_multi_index, self.layers)

    def get_layer_by_multi_index_recursive(self, at_multi_index, tree):
        item = tree[at_multi_index[0]]
        if (len(at_multi_index) == 1):
            return item
        else:
            return self.get_layer_by_multi_index_recursive(at_multi_index[1:], item)

    def get_layer_by_name(self, name):
        layers = self.flatten()
        for layer in layers:
            if layer.name == name:
                return layer
        return None

    def find_dependent_layer(self, layer, dependent_type):
        for child in self.flatten():
            if child.dependent_of == layer.invariant:
                return child
        return None

    def find_parent_of_dependent_layer(self, child, dependent_type):
        for layer in self.flatten():
            if child.dependent_of == layer.invariant:
                return layer
        return None

    def find_transient_layer(self):
        for child in self.flatten():
            if child.transient_edit_layer:
                return child
        return None

    # returns a list of the child layers of a root or folder layer
    def get_layer_children(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        if mi is None:
            return []
        l = self.get_layer_by_multi_index(mi)
        if not isinstance(l, list):
            return []

        ret = []
        for item in l[1:]:
            i = item
            # a list means the first element in the list is the folder layer containing the other elements in the list
            if (isinstance(item, list)):
                i = item[0]
            ret.append(i)

        return ret

    # returns a list of all generations of layers of a root or folder layer
    def get_layer_descendants(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        if mi is None:
            return []
        l = self.get_layer_by_multi_index(mi)
        if not isinstance(l, list):
            return []

        ret = self.flatten_recursive(l)
        return ret[1:]

    def get_layer_parent(self, layer):
        """Return the immediate parent
        """
        parent = None
        mi = self.get_multi_index_of_layer(layer)
        if mi is not None and len(mi) > 1:
            mi[-1] = 0
            parent = self.get_layer_by_multi_index(mi)
        else:
            parent = self.get_layer_by_multi_index([0])
        return parent

    def get_layer_parents(self, layer):
        """Return a list of parent layers, starting from the immediate parent
        and continuing to older ancestors but ignoring the root layer
        """
        parents = []
        mi = self.get_multi_index_of_layer(layer)
        if mi is not None:
            while len(mi) > 1:
                mi[-1] = 0
                l = self.get_layer_by_multi_index(mi)
                parents.append(l)
                mi.pop()
        return parents

    def get_previous_sibling(self, layer):
          mi = self.get_multi_index_of_previous_sibling(layer)
          l = self.get_layer_by_multi_index(mi)
          return l

    def get_children(self, layer):
        """Return a list containing the hierarchy starting at the specified
        layer and containing any children and descendents.

        This potentially could include lists of lists of lists, as deep as the
        hierarchy goes.
        """
        mi = self.get_multi_index_of_layer(layer)
        l = self.get_layer_by_multi_index(mi)
        if not isinstance(l, list):
            return []

        ret = []
        for item in l[1:]:
            # a list means the first element in the list is the folder layer containing the other elements in the list
            if (isinstance(item, list)):
                sub = [item[0]]
                sub.extend(self.get_children(item[0]))
                ret.append(sub)
            else:
                ret.append(item)

        return ret

    def get_flattened_children(self, layer):
        """Return a list containing the hierarchy starting at the specified
        layer and containing any children and descendents.

        This potentially could include lists of lists of lists, as deep as the
        hierarchy goes.
        """
        mi = self.get_multi_index_of_layer(layer)
        l = self.get_layer_by_multi_index(mi)
        if not isinstance(l, list):
            return []

        ret = []
        for item in l[1:]:
            if (isinstance(item, list)):
                ret.append(item[0])
                ret.extend(self.get_children(item[0]))
            else:
                ret.append(item)
        return ret

    def get_layer_by_flattened_index(self, index):
        flattened = self.flatten()
        if index < len(flattened):
            return flattened[index]

        return None

    def get_visible_layers(self, layer_visibility, only_visible_layers=True):
        layers = []
        for layer in self.flatten():
            if (only_visible_layers and not layer_visibility[layer]["layer"]):
                continue
            layers.append(layer)
        return layers

    def get_mergeable_layers(self):
        layers = [layer for layer in self.flatten() if layer.find_merge_layer_class(layer) is not None]
        layers.reverse()
        return layers

    def get_layers_of_type(self, layer_type):
        layers = []
        for layer in self.flatten():
            if layer.type == layer_type:
                layers.append(layer)
        return layers

    def get_timestamped_layers(self, layer_visibility, only_visible_layers=True):
        possible = self.get_visible_layers(layer_visibility, only_visible_layers)
        layers = []
        earliest_time = 0.0
        latest_time = 0.0
        for layer in possible:
            if layer.start_time > 0.0:
                layers.append(layer)
                if earliest_time == 0.0 or layer.start_time < earliest_time:
                    earliest_time = layer.start_time
                if layer.start_time > latest_time:
                    latest_time = layer.start_time
                if layer.end_time > latest_time and layer.end_time > layer.start_time:
                    latest_time = layer.end_time
        if latest_time == 0.0:
            latest_time = earliest_time
        return layers, earliest_time, latest_time

    def get_playback_layers(self, skip_layer=None, tree=None):
        # layers = [layer for layer in self.flatten()[1:] if layer != skip_layer]
        # return layers

        result = []
        if tree is None: tree = self.layers
        for item in tree:
            if (isinstance(item, ly.Layer)):
                if not (item.is_root() or item == skip_layer):
                    result.append(item)
                if item.grouped:
                    # skip grouped children
                    return result
            else:
                result.extend(self.get_playback_layers(skip_layer, item))
        return result

    ##### ly.Layer info

    def has_user_created_layers(self):
        """Returns true if all the layers can be recreated automatically

        If any layer has any user-created data, this will return False.
        """
        for layer in self.flatten():
            if not layer.skip_on_insert:
                return True
        return False

    def check_layer(self, layer):
        if layer is not None:
            try:
                layer.check_for_problems()
            except Exception as e:
                if hasattr(e, 'error_points'):
                    return e
                else:
                    raise
        return None

    # fixme -- why wouldn't is_raisable, etc be an attribute of the layer???
    def is_raisable(self, layer):
        if not layer.is_root():
            mi = self.get_multi_index_of_layer(layer)
            if mi is not None:
                return mi[len(mi) - 1] >= 2
        return False

    def is_lowerable(self, layer):
        if not layer.is_root():
            mi = self.get_multi_index_of_layer(layer)
            if mi is not None:
                n = mi[len(mi) - 1]
                mi2 = mi[: len(mi) - 1]
                parent_list = self.get_layer_by_multi_index(mi2)
                total = len(parent_list)

                return n < (total - 1)
        return False

    def count_layers(self):
        n = 0
        for layer in self.flatten():
            if not layer.is_root():
                n += 1
        #
        return n

    def count_raster_layers(self):
        # fixme -- what  in the world are these used for?
        # and if there is a need, maybe it should be more  like
        # count_layer_of_type(self, layer_type="")
        n = 0
        for layer in self.flatten():
            if (hasattr(layer, "images") and layer.images is not None):
                n += 1
        #
        return n

    def count_vector_layers(self):
        n = 0
        for layer in self.flatten():
            if (hasattr(layer, "points") and (layer.points is not None or
                                              layer.rings is not None)):
                n += 1
        #
        return n

    ##### ly.Layer modification

    def update_map_server_ids(self, layer_type, before, after):
        """Change map server IDs after the server list has been reordered.
        """
        affected = self.get_layers_of_type(layer_type)
        log.debug("affected layers of %s: %s" % (layer_type, str(affected)))
        for i, layer in enumerate(after):
            if layer.default:
                default = i
                break
        else:
            default = 0
        for layer in affected:
            old_id = layer.map_server_id
            old_host = before[old_id]
            try:
                new_id = after.index(old_host)
            except ValueError:
                new_id = default
            layer.map_server_id = new_id

    ##### Add layers

    def add_layers(self, layers, editor=None):
        layers = layers[:]  # work on copy so we don't mess up the caller's list
        parent = None
        if layers[0].is_folder():
            parent = layers.pop(0)
            self.insert_loaded_layer(parent, editor)

        for layer in layers:
            if editor is not None:
                layer.check_projection(editor.task)
            if not layer.load_error_string:
                self.insert_loaded_layer(layer, editor, last_child_of=parent)
        return layers

    def add_all(self, layer_order, editor=None):
        existing = self.flatten()
        for layer in existing:
            if not layer.is_root():
                self.remove_layer(layer)
        layers = []
        log.debug("layer order: %s" % str(layer_order))
        for mi, layer_list in layer_order:
            for layer in layer_list:
                log.debug("adding %s at %s" % (str(layer), str(mi)))
                if editor is not None:
                    layer.check_projection(editor.task)
                if len(mi) > 1:
                    # check to see if it's a folder layer, and if so it would
                    # have been serialized with a 0 as the last multi index.
                    # On insert, this barfs, so strip it off.
                    if mi[-1] == 0:
                        mi = mi[:-1]
                # LEGACY:
                if layer.invariant == -999:
                    log.warning("old json format: invariant not set for %s" % layer)
                    layer.invariant = self.get_next_invariant()
                log.debug("loaded invariant: %s" % layer.invariant)
                self.insert_loaded_layer(layer, mi=mi, skip_invariant=True)
                layers.append(layer)
                log.debug("inserted invariant: %s" % layer.invariant)

                # Automatically adjust next_invariant to reflect highest
                # invariant seen
                if layer.invariant >= self.next_invariant:
                    self.next_invariant = layer.invariant + 1
        self.recalc_all_bounds()
        return layers

    def insert_loaded_layer(self, layer, editor=None, before=None, after=None, invariant=None, first_child_of=None, last_child_of=None, mi=None, skip_invariant=None):
        self.layer_loaded_event(layer)
        if mi is None:
            mi = self.get_insertion_multi_index(before, after, first_child_of, last_child_of, layer.background, layer.opaque, layer.bounded)
        self.insert_layer(mi, layer, invariant=invariant, skip_invariant=skip_invariant)
        return mi

    def insert_json(self, json_data, editor, mi, old_invariant_map=None):
        mi = list(mi)  # operate on a copy, otherwise changes get returned
        layer = ly.Layer.load_from_json(json_data, self)[0]
        if old_invariant_map is not None:
            old_invariant_map[json_data['invariant']] = layer
        self.layer_loaded_event(layer)
        self.insert_layer(mi, layer)
        if json_data['children']:
            mi.append(1)
            for c in json_data['children']:
                self.insert_json(c, editor, mi, old_invariant_map)
                mi[-1] = mi[-1] + 1
        layer.update_bounds()
        return layer

    def insert_children(self, in_layer, children):
        """Insert a list of children as children of the speficied folder

        The list of children should be generated from :func:`get_children`
        """
        log.debug("before: layers are " + str(self.layers))
        mi = self.get_multi_index_of_layer(in_layer)
        mi.append(1)
        log.debug("inserting children " + str(children) + " using multi_index = " + str(mi))
        for layer in children:
            if isinstance(layer, list):
                child = layer[0]
                self.insert_layer(mi, child, child.invariant)
                self.insert_children(child, layer[1:])
            else:
                self.insert_layer(mi, layer, layer.invariant)
            mi[-1] += 1
        log.debug("after: layers are " + str(self.layers))

    def insert_layer(self, at_multi_index, layer, invariant=None, skip_invariant=False):
        if (at_multi_index is None or at_multi_index == []):
            at_multi_index = self.find_default_insert_layer()

        log.debug("before: layers are " + str(self.layers))
        log.debug("inserting layer " + str(layer) + " using multi_index = " + str(at_multi_index))
        if (not isinstance(layer, list)):
            if layer.transient_edit_layer:
                layer.invariant = self.transient_invariant
            elif not skip_invariant:
                # ly.Layers being loaded from a project file will have their
                # invariants already saved, so don't mess with them.
                layer.invariant = self.get_next_invariant(invariant)
            if layer.is_folder() and not layer.is_root():
                layer = [layer]
        self.insert_layer_recursive(at_multi_index, layer, self.layers)
        log.debug("after: layers are " + str(self.layers))

    def insert_layer_recursive(self, at_multi_index, layer, tree):
        if (len(at_multi_index) == 1):
            tree.insert(at_multi_index[0], layer)
        else:
            item = tree[at_multi_index[0]]
            self.insert_layer_recursive(at_multi_index[1:], layer, item)

    ##### Replace layers

    def replace_layer(self, at_multi_index, layer):
        """Replace a layer with another

        Returns a tuple containing the replaced layer as the first component
        and a list (possibly empty) of its children as the second component.
        """
        if (at_multi_index is None or at_multi_index == []):
            at_multi_index = self.find_default_insert_layer()

        log.debug("before: layers are " + str(self.layers))
        log.debug("inserting layer " + str(layer) + " using multi_index = " + str(at_multi_index))
        if (not isinstance(layer, list)):
            if layer.is_folder() and not layer.is_root():
                layer = [layer]
        replaced = self.replace_layer_recursive(at_multi_index, layer, self.layers)
        log.debug("after: layers are " + str(self.layers))
        if isinstance(replaced, list):
            return replaced[0], replaced[1:]
        return replaced, []

    def replace_layer_recursive(self, at_multi_index, layer, tree):
        if (len(at_multi_index) == 1):
            replaced = tree[at_multi_index[0]]
            tree[at_multi_index[0]] = layer
            return replaced
        else:
            item = tree[at_multi_index[0]]
            return self.replace_layer_recursive(at_multi_index[1:], layer, item)

    def remove_transient_layer(self):
        old = self.find_transient_layer()
        if old:
            insertion_index = self.get_multi_index_of_layer(old)
            self.remove_layer_at_multi_index(insertion_index)
        else:
            insertion_index = None
        return old, insertion_index

    def replace_transient_layer(self, layer, editor, **kwargs):
        old, insertion_index = self.remove_transient_layer()
        self.insert_loaded_layer(layer, editor, **kwargs)
        return old, insertion_index

    ##### Remove layers

    # FIXME: layer removal commands should return the hierarchy of layers
    # removed so that the operation can be undone correctly.
    def remove_layer(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        self.remove_layer_at_multi_index(mi)

    def remove_layer_at_multi_index(self, at_multi_index):
        self.remove_layer_recursive(at_multi_index, self.layers)

    def remove_layer_recursive(self, at_multi_index, tree):
        index = at_multi_index[0]
        if (len(at_multi_index) == 1):
            layer = tree[index]
            del tree[index]
            return layer
        else:
            sublist = tree[index]
            return self.remove_layer_recursive(at_multi_index[1:], sublist)

    ##### Control points: links between layers

    def set_control_point_link(self, dep_or_layer, truth_or_cp, truth_layer=None, truth_cp=None, locked=False):
        """Links a control point to a truth (master) layer

        Parameters can be passed two ways: if only two parameters
        are passed in, they will each be tuples of (layer.invariant,
        control_point_index), the first tuple being the dependent layer &
        second the master layer.  Otherwise, all 4 parameters are needed,
        individually specifying the dependent layer and its control point,
        followed by the truth layer and its control point.

        Passing in two arguments is a convenience for using the return
        data from remove_control_point_links in the undo method of
        MoveControlPointCommand.
        """
        if truth_layer is None:
            entry = dep_or_layer
            truth = truth_or_cp
        else:
            entry = (dep_or_layer.invariant, truth_or_cp)
            truth = (truth_layer.invariant, truth_cp)
        log.debug("control_point_links: adding %s child of %s" % (entry, truth))
        self.control_point_links[entry] = (truth, locked)

    def get_control_point_links(self, layer):
        """Returns the list of control points that the specified layer links to

        """
        links = []
        for dep, (truth, locked) in self.control_point_links.items():
            dep_invariant, dep_cp = dep[0], dep[1]
            if dep_invariant == layer.invariant:
                truth_invariant, truth_cp = truth[0], truth[1]
                links.append((dep_cp, truth_invariant, truth_cp, locked))
        return links

    def remove_control_point_links(self, layer, remove_cp=-1, force=False):
        """Remove links to truth layer control points from the specified
        dependent layer.

        If a remove_cp is specified, only remove that control point's
        reference, otherwise remove all control points links that are on the
        dependent layer.
        """
        to_remove = []
        for dep, (truth, locked) in self.control_point_links.items():
            log.debug("control_point_links: %s child of %s" % (dep, truth))
            dep_layer_invariant, dep_cp = dep[0], dep[1]
            if dep_layer_invariant == layer.invariant and (remove_cp < 0 or remove_cp == dep_cp) and (not locked or force):
                to_remove.append((dep, truth, locked))
        for dep, truth, locked in to_remove:
            log.debug("control_point_links: removing %s from %s" % (dep, truth))
            del self.control_point_links[dep]
        return to_remove

    def update_linked_control_points(self):
        """Update control points in depedent layers from the truth layers.

        The truth_layer is the layer that control point values are taken from
        and propagated to the dependent layer
        """
        layers = []
        for dep, (truth, locked) in self.control_point_links.items():
            truth_layer, truth_cp = self.get_layer_by_invariant(truth[0]), truth[1]
            dep_layer, dep_cp = self.get_layer_by_invariant(dep[0]), dep[1]
            log.debug(f"control_point_links: update {dep} (layer {dep_layer}) child of {truth} (layer {truth}))")
            try:
                dep_layer.copy_control_point_from(dep_cp, truth_layer, truth_cp)
            except AttributeError as e:
                log.error(f"{e}: failed copying control point: {dep} (layer {dep_layer}) child of {truth} (layer {truth}))")
            else:
                layers.append(dep_layer)
        return layers

    def remove_all_links_to_layer(self, layer):
        """Remove all links to the specified layer, whether it's a truth layer
        or a dependent layer.

        Used when deleting a layer.
        """
        to_remove = []
        for dep, (truth, locked) in self.control_point_links.items():
            invariant, _ = dep[0], dep[1]
            if invariant == layer.invariant:
                to_remove.append((dep, truth, locked))
            else:
                invariant, _ = truth[0], truth[1]
                if invariant == layer.invariant:
                    to_remove.append((dep, truth, locked))
        for dep, truth, locked in to_remove:
            log.debug("control_point_links: removing %s from %s" % (dep, truth))
            del self.control_point_links[dep]
        return to_remove

    def restore_all_links_to_layer(self, layer, links):
        for dep, truth, locked in links:
            log.debug("control_point_links: restoring %s from %s" % (dep, truth))
            self.control_point_links[dep] = (truth, locked)

    def get_all_control_point_links_copy(self):
        return self.control_point_links.copy()

    def restore_all_control_point_links(self, copy):
        self.control_point_links = copy.copy()


    ##### Layer load

    def load_all_from_json(self, json, batch_flags=None):
        order = []
        if json[0] == "extra json data":
            extra_json = json[1]
            json = json[2:]
        else:
            extra_json = None
        for serialized_data in json:
            try:
                loaded = ly.Layer.load_from_json(serialized_data, self, batch_flags)
                index = serialized_data['index']
                order.append((index, loaded))
                log.debug("processed json from layer %s" % loaded)
            except RuntimeError as e:
                batch_flags.messages.append("ERROR: %s" % str(e))
        order.sort()
        log.debug("load_all_from_json: order: %s" % str(order))

        self.load_extra_json_attrs(extra_json, batch_flags)
        return order, extra_json

    def load_extra_json_attrs(self, extra_json, batch_flags):
        for attr, from_json in self.get_from_json_attrs():
            try:
                from_json(extra_json)
            except KeyError:
                message = "%s not present in layer %s; attempting to continue" % (attr, self.name)
                log.warning(message)
                batch_flags.messages.append("WARNING: %s" % message)

    def load_all_from_zip(self, zf, batch_flags=None):
        expanded_zip = ExpandZip(zf, ["pre json data", "post json data", "extra json data", "json layer description"])
        order = []
        try:
            text = zf.read("pre json data")
        except KeyError:
            pass  # optional file, so skip if doesn't exist
        else:
            pre_json = json.loads(text)
            self.process_pre_json_data(pre_json)
        try:
            text = zf.read("post json data")
        except KeyError:
            text = zf.read("extra json data")
        extra_json = json.loads(text)
        for info in zf.infolist():
            if info.filename.endswith("json layer description"):
                text = zf.read(info.filename)
                serialized_data = json.loads(text)
                if 'url' in serialized_data:
                    # recreate the url to point to the the file in the temp dir
                    # resulting from expanding the zipfile. The project save
                    # code can then get the pathname of the file from the
                    # file_path member of the layer
                    relname = serialized_data['url']
                    serialized_data['url'] = os.path.join(expanded_zip.root, relname)
                    log.debug("layer url %s" % serialized_data['url'])
                try:
                    loaded = ly.Layer.load_from_json(serialized_data, self, batch_flags)
                    index = serialized_data['index']
                    order.append((index, loaded))
                    log.debug("processed json from layer %s" % loaded)
                except RuntimeError as e:
                    batch_flags.messages.append("ERROR: %s" % str(e))
        order.sort()
        log.debug("load_all_from_zip: order: %s" % str(order))

        self.load_extra_json_attrs(extra_json, batch_flags)
        self.zip_file_source = expanded_zip
        self.add_cleanup_function(functools.partial(expanded_zip.cleanup))
        return order, extra_json

    def restore_layer_relationships_after_load(self):
        # calculate bound starting at leaf layers and working back up to folder
        # layers
        for layer in self.flatten():
            layer.restore_layer_relationships_after_load()

    ##### Layer save

    def verify_ok_to_save(self):
        prefs = self.project.preferences
        if prefs.check_errors_on_save:
            return self.project.check_all_layers_for_errors(True)
        else:
            return True

    def save_raw_data(self, uri, raw_data):
        extra_json = self.project.current_extra_json
        return self.save_all_zip(uri, extra_json)

    def calc_raw_data_to_save(self):
        return None

    def process_pre_json_data(self, json):
        # pre json data is stuff that layers need to exist at the time they are
        # created
        pass

    def calc_pre_json_data(self, pre_json_data=None):
        if pre_json_data is None:
            pre_json_data = {}
        log.debug("pre json data:\n%s" % repr(pre_json_data))
        return pre_json_data

    def calc_post_json_data(self, extra_json_data=None):
        if extra_json_data is None:
            extra_json_data = {}
        for attr, to_json in self.get_to_json_attrs():
            extra_json_data[attr] = to_json()
        log.debug("post json data:\n%s" % repr(extra_json_data))
        return extra_json_data

    def save_all_zip(self, file_path, extra_json_data=None):
        """Save all layers into a zip file that includes any referenced images,
        shapefiles, etc. so the file becomes portable and usable on other
        systems.

        """
        log.debug("saving layers in project file: " + file_path)
        layer_info = self.flatten_with_indexes()
        log.debug("layers are " + str(self.layers))
        log.debug("layer info is:\n" + "\n".join([str(s) for s in layer_info]))
        log.debug("layer subclasses:\n" + "\n".join(["%s -> %s" % (t, str(s)) for t, s in ly.Layer.get_subclasses().items()]))

        pre_json_data = self.calc_pre_json_data()
        post_json_data = self.calc_post_json_data(extra_json_data)
        try:
            with open(file_path, "wb") as fh:
                zf = zipfile.ZipFile(fh, mode='w', compression=zipfile.ZIP_DEFLATED)
                try:
                    zf.writestr("pre json data", json.dumps(pre_json_data))
                except TypeError:
                    log.error("Failed encoding pre json data:\n%s" % repr(pre_json_data))
                    raise
                try:
                    zf.writestr("post json data", json.dumps(post_json_data))
                except TypeError:
                    log.error("Failed encoding post json data:\n%s" % repr(post_json_data))
                    raise
                for index, layer in layer_info:
                    zip_root = "/".join([str(a) for a in index]) + "-" + layer.name + "/"
                    log.debug("index=%s, layer=%s, path=%s" % (index, layer, layer.file_path))
                    data = layer.serialize_json(index)
                    if data is not None:
                        # only store extra files for layers that aren't
                        # encoded entirely in the JSON
                        paths = layer.extra_files_to_serialize()
                        if paths:
                            # point to reparented data file
                            basename = os.path.basename(paths[0])
                            data['url'] = zip_root + basename

                            # save all files into zip file
                            for p in paths:
                                try:
                                    p = filesystem_path(p)
                                except:
                                    raise RuntimeError("Can't yet handle URIs not on local filesystem")
                                basename = os.path.basename(p)
                                archive_name = zip_root + basename
                                zf.write(p, archive_name, zipfile.ZIP_STORED)

                        try:
                            text = json.dumps(data, indent=4)
                        except Exception as e:
                            log.error("JSON failure %s, layer %s: data=%s" % (e, layer.name, repr(data)))
                            errors = []
                            for k, v in data.items():
                                small = {k: v}
                                try:
                                    _ = json.dumps(small)
                                except Exception:
                                    errors.append((k, v))
                            log.error("JSON failures at: %s" % ", ".join(["%s: %s" % (k, v) for k, v in errors]))
                            return "Failed saving data in layer %s.\n\n%s" % (layer.name, e)

                        zip_path = zip_root + "json layer description"
                        processed = collapse_json(text, 12)
                        zf.writestr(zip_path, processed)
                zf.close()

        except RuntimeError as e:
            log.error("file save error: %s" % str(e))
            return "Failed saving %s: %s" % (file_path, e)
        except Exception as e:
            import traceback
            log.error("file save error: %s\n%s" % (str(e), traceback.format_exc()))
            return "Failed saving %s: %s" % (file_path, e)
        # zf = zipfile.ZipFile(file_path)
        # print("\n".join(zf.namelist()))
        return ""

    def save_layer(self, layer, file_path, loader=None):
        if layer is not None:
            error = loaders.save_layer(layer, file_path, loader)
            if not error:
                layer.name = os.path.basename(layer.file_path)
            return error
        return "No selected layer."

    #### file recognition

    @classmethod
    def can_load_file_exact(cls, file_metadata):
        return "loader" in file_metadata
 
    @classmethod
    def can_load_file_generic(cls, file_metadata):
        False

# def test():
#     a = ly.Layer()
#     a.name = "a"
#     b = ly.Layer()
#     b.name = "b"
#     c = ly.Layer()
#     c.name = "c"
#     d = ly.Layer()
#     d.name = "d"
#     e = ly.Layer()
#     e.name = "e"
#     f = ly.Layer()
#     f.name = "f"
#     g = ly.Layer()
#     g.name = "g"
#     h = ly.Layer()
#     h.name = "h"

#     tree = [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ]
#     print a
#     print tree

#     lm = LayerManager()
#     lm.layers = tree

#     print "lm.flatten()"
#     print lm.flatten()

#     for layer in [ a, b, c, d, e, f, g, h ]:
#         print "lm.get_multi_index_of_layer( {0} )".format( layer )
#         mi = lm.get_multi_index_of_layer( layer )
#         print mi
#         print lm.get_layer_by_multi_index( mi )

#     print "removing a"
#     lm.remove_layer_at_multi_index( [ 0, 0 ] )
#     print lm.layers
#     print "removing d"
#     lm.remove_layer_at_multi_index( [ 1, 1, 0 ] )
#     print lm.layers
#     print "removing e"
#     lm.remove_layer_at_multi_index( [ 1, 1, 0 ] )

#     print lm.layers
#     print "inserting a"
#     lm.insert_layer( [ 1, 1, 1 ], a )
#     print lm.layers
