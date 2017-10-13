import os
import json
import zipfile
import functools

from fs.opener import opener, fsopen

import library.rect as rect

from layers import Grid
from layers import Layer
from layers import LayerStyle
from layers import LineLayer
from layers import RootLayer
from layers import Scale
from layers import TriangleLayer
from layers import loaders
from layers import LayerStyle, parse_styles_from_json, styles_to_json
from command import UndoStack

# Enthought library imports.
from traits.api import Any
from traits.api import Dict
from traits.api import Event
from traits.api import Int
from traits.api import List
from pyface.api import GUI

from omnivore.framework.document import BaseDocument
from omnivore.utils.jsonutil import collapse_json
from omnivore.utils.fileutil import ExpandZip

import logging
log = logging.getLogger(__name__)


class LayerManager(BaseDocument):

    """
    Manages the layers (a tree of Layer).

    A "multi_index" is a set of indexes taken in order, that give the location
    of a sub-tree or leaf within a tree. For instance,
    in the tree [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ], the multi_index [ 0 ] refers to subtree [ a, b ],
    the multi_index [ 1, 1, 1 ] refers to the leaf e, and the multi_index [ 3, 0 ] refers to the leaf g.

    The first layer in the overall list and in each sublist is assumed to be a "folder" layer, whose only
    purpose at present is to hold the folder name.
    """
    project = Any

    # if the project is loaded from a zip file, the ExpandZip object is stored
    # here so the unpacked directory can be referenced when re-saving the
    # project
    zip_file_source = Any

    layers = List(Any)

    next_invariant = Int(0)

    default_styles = Any

    layer_loaded = Event

    layers_changed = Event

    layer_contents_changed = Event

    layer_contents_changed_in_place = Event

    # when points are deleted from a layer the indexes of the points in the
    # merge dialog box become invalid; so this event will trigger the user to
    # re-find duplicates in order to create a valid list again
    layer_contents_deleted = Event

    layer_metadata_changed = Event

    projection_changed = Event

    refresh_needed = Event

    background_refresh_needed = Event

    threaded_image_loaded = Event

    # Linked control points are slaves of a truth layer: a dict that maps the
    # dependent layer/control point to the truth layer/control point
    control_point_links = Dict(Any)

    # Transient layer always uses invariant
    transient_invariant = -3

    def _undo_stack_default(self):
        return UndoStack()

    ##### Python special methods

    def __str__(self):
        root = self.get_layer_by_multi_index([0])
        layers = self.get_children(root)
        return str(layers)

    ##### Creation/destruction

    @classmethod
    def create(cls, project):
        """Convenience function to create a new, empty LayerManager

        Since classes that use Traits can't seem to use an __init__ method,
        we are forced to use a convenience function to initialize non-traits
        members.  Trying to define layers using _layers_default results
        in recursion from the print statement in insert_layers trying to
        references self.layers which isn't yet defined.
        """
        self = cls()
        self.project = project

        # In order for the serializer to correcly map the layer invariant to
        # the actual layer, the next_invariant must be preset so first user
        # added layer will use 1.  If the number of default layers added below
        # changes, modify next_invariant to match! next_invariant = 1 - (# of
        # calls to insert_layer)
        self.next_invariant = -2
        self.default_styles = self.project.task.default_styles
        layer = RootLayer(manager=self)
        self.insert_layer([0], layer)
        grid = Grid(manager=self)
        self.insert_layer([1], grid)
        scale = Scale(manager=self)
        self.insert_layer([2], scale)

        # Add hook to create layer instances for debugging purposes
        if "--debug-objects" in self.project.window.application.command_line_args:
            import debug
            debug.debug_objects(self)
        return self

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
        print "next invariant: %d" % self.next_invariant
        for layer in layers:
            print "  %s: invariant=%d" % (layer, layer.invariant)

    def debug_structure(self):
        lines = self.debug_structure_recursive(self.layers)
        return "\n".join(lines)

    def debug_structure_recursive(self, tree, indent=""):
        result = []

        for item in tree:
            if (isinstance(item, Layer)):
                result.append(indent + str(self.get_multi_index_of_layer(item)) + " " + item.debug_info(indent))
            else:
                result.extend(self.debug_structure_recursive(item, indent + "    "))

        return result

    ##### Serialization

    def get_to_json_attrs(self):
        return [(m[0:-8], getattr(self, m)) for m in dir(self) if hasattr(self, m[0:-8]) and m.endswith("_to_json")]

    def get_from_json_attrs(self):
        return [(m[0:-10], getattr(self, m)) for m in dir(self) if hasattr(self, m[0:-10]) and m.endswith("_from_json")]

    def control_point_links_to_json(self):
        # json can't handle dictionaries with tuples as their keys, so have
        # to compress
        cplist = []
        for entry, (truth, locked) in self.control_point_links.iteritems():
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
        return styles_to_json(self.default_styles)

    def default_styles_from_json(self, json_data):
        sdict = json_data['default_styles']
        d = parse_styles_from_json(sdict)
        self.update_default_styles(d)

    ##### Multi-index calculations

    def get_insertion_multi_index(self, before=None, after=None, first_child_of=None, last_child_of=None):
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
            mi = None
        return mi

    def get_multi_index_of_layer(self, layer):
        return self.get_multi_index_of_layer_recursive(layer, self.layers)

    def get_multi_index_of_layer_recursive(self, layer, tree):
        for i, item in enumerate(tree):
            if (isinstance(item, Layer)):
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

    def accumulate_layer_bounds(self, layers):
        result = rect.NONE_RECT

        for layer in layers:
            if (result == rect.NONE_RECT):
                result = layer.bounds
            else:
                result = rect.accumulate_rect(result, layer.bounds)

        return result

    ##### Style

    def update_default_styles(self, styles):
        self.default_styles = styles

        # Make sure "other" is a valid style
        try:
            s = LayerStyle()
            s.parse(str(self.default_styles["other"]))
        except Exception:
            log.warning("Invalid style for other, using default")
            self.default_styles["other"] = LayerStyle() # minimal default styles
        for type_name in sorted(styles.keys()):
            style = self.default_styles[type_name]
            log.debug("style %s: %s" % (type_name, str(style)))

    def update_default_style_for(self, layer, style):
        if hasattr(layer, "type"):
            type_name = layer.type
        else:
            type_name = layer
        log.debug("updating default style for %s: %s" % (type_name, str(style)))
        self.default_styles[type_name] = style.get_copy()

    def get_default_style_for(self, layer):
        if hasattr(layer, "type"):
            type_name = layer.type
        else:
            type_name = layer
        style = self.default_styles.get(type_name, self.default_styles["other"])
        log.debug("get default style for: %s: %s" % (type_name, str(style)))
        return style.get_copy()

    ##### Flatten

    def flatten(self):
        return self.flatten_recursive(self.layers)

    def flatten_recursive(self, tree):
        result = []

        for item in tree:
            if (isinstance(item, Layer)):
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
            if (isinstance(item, Layer)):
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

    def find_vector_object_insert_layer(self, event_layer):
        """Find the appropriate layer to insert a vector object, given the
        layer on which the event occurred.

        If the event_layer is an annotation layer it will typically be used,
        *unless* the layer is grouped, in which case it will look
        upwards in the hierarchy to find the first parent annotation layer
        that is not grouped.

        None is returned if the top-level annotation layer is also grouped.
        """
        while not event_layer.is_root():
            if event_layer.is_folder():
                if not event_layer.grouped:
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

    def get_untimestamped_layers(self):
        layers = []
        for layer in self.flatten():
            if layer.start_time > 0.0:
                continue
            layers.append(layer)
        return layers


    ##### Layer info

    def has_user_created_layers(self):
        """Returns true if all the layers can be recreated automatically

        If any layer has any user-created data, this will return False.
        """
        for layer in self.flatten():
            if not layer.skip_on_insert:
                return True
        return False

    def check_layer(self, layer, window):
        if layer is not None:
            try:
                layer.check_for_problems(window)
            except Exception, e:
                if hasattr(e, 'points'):
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

    def dispatch_event(self, event, value=True):  # refactor out
        log.debug("dispatching event %s = %s" % (event, value))
        setattr(self, event, value)

    def post_event(self, event_name, *args):  # refactor out
        log.debug("event: %s.  args=%s" % (event_name, str(args)))

    def get_event_callback(self, event):  # refactor out
        import functools
        callback = functools.partial(self.post_event, event)
        return callback

    ##### Layer modification

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

    def add_layer(self, type=None, editor=None, before=None, after=None):
        if type == "grid":
            layer = Grid(manager=self)
        elif type == "triangle":
            layer = TriangleLayer(manager=self)
        else:
            layer = LineLayer(manager=self)
        layer.new()
        self.insert_loaded_layer(layer, editor, before, after)
        self.dispatch_event('layers_changed')
        if editor is not None:
            GUI.invoke_later(editor.layer_tree_control.set_edit_layer, layer)
        return layer

    def add_layers(self, layers, is_project, editor):
        parent = None
        if is_project:
            # remove all other layers so the project can be inserted in the
            # correct order
            existing = self.flatten()
            for layer in existing:
                if not layer.is_root():
                    self.remove_layer(layer)

            # layers are inserted from the beginning, so reverse loaded layers
            # so they won't show up backwards
            layers.reverse()
        else:
            if layers[0].is_folder():
                parent = layers.pop(0)
                self.insert_loaded_layer(parent, editor)

        for layer in layers:
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
                self.insert_loaded_layer(layer, mi=mi, skip_invariant=True)
                layers.append(layer)

                # Automatically adjust next_invariant to reflect highest
                # invariant seen
                if layer.invariant >= self.next_invariant:
                    self.next_invariant = layer.invariant + 1
        self.recalc_all_bounds()
        return layers

    def insert_loaded_layer(self, layer, editor=None, before=None, after=None, invariant=None, first_child_of=None, last_child_of=None, mi=None, skip_invariant=None):
        self.dispatch_event('layer_loaded', layer)
        if mi is None:
            mi = self.get_insertion_multi_index(before, after, first_child_of, last_child_of)
        self.insert_layer(mi, layer, invariant=invariant, skip_invariant=skip_invariant)
        return mi

    def insert_json(self, json_data, editor, mi, old_invariant_map=None):
        mi = list(mi)  # operate on a copy, otherwise changes get returned
        layer = Layer.load_from_json(json_data, self)[0]
        if old_invariant_map is not None:
            old_invariant_map[json_data['invariant']] = layer
        self.dispatch_event('layer_loaded', layer)
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
                # Layers being loaded from a project file will have their
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

    def replace_transient_layer(self, layer, editor, **kwargs):
        old = self.find_transient_layer()
        if old:
            insertion_index = self.get_multi_index_of_layer(old)
            self.remove_layer_at_multi_index(insertion_index)
        else:
            insertion_index = None
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
        for dep, (truth, locked) in self.control_point_links.iteritems():
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
        for dep, (truth, locked) in self.control_point_links.iteritems():
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
        for dep, (truth, locked) in self.control_point_links.iteritems():
            truth_layer, truth_cp = self.get_layer_by_invariant(truth[0]), truth[1]
            dep_layer, dep_cp = self.get_layer_by_invariant(dep[0]), dep[1]
            dep_layer.copy_control_point_from(dep_cp, truth_layer, truth_cp)
            layers.append(dep_layer)
        return layers

    def remove_all_links_to_layer(self, layer):
        """Remove all links to the specified layer, whether it's a truth layer
        or a dependent layer.

        Used when deleting a layer.
        """
        to_remove = []
        for dep, (truth, locked) in self.control_point_links.iteritems():
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
                loaded = Layer.load_from_json(serialized_data, self, batch_flags)
                index = serialized_data['index']
                order.append((index, loaded))
                log.debug("processed json from layer %s" % loaded)
            except RuntimeError, e:
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

    def load_all_from_zip(self, archive_path, zf, batch_flags=None):
        expanded_zip = ExpandZip(zf, ["extra json data", "json layer description"])
        order = []
        text = zf.read("extra json data")
        extra_json = json.loads(text)
        for info in zf.infolist():
            print("info: %s" % info.filename)
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
                    loaded = Layer.load_from_json(serialized_data, self, batch_flags)
                    index = serialized_data['index']
                    order.append((index, loaded))
                    log.debug("processed json from layer %s" % loaded)
                except RuntimeError, e:
                    batch_flags.messages.append("ERROR: %s" % str(e))
        order.sort()
        log.debug("load_all_from_zip: order: %s" % str(order))

        self.load_extra_json_attrs(extra_json, batch_flags)
        self.zip_file_source = expanded_zip
        self.add_cleanup_function(functools.partial(expanded_zip.cleanup))
        return order, extra_json

    ##### Layer save

    def save_all(self, file_path, extra_json_data=None):
        return self.save_all_zip(file_path, extra_json_data)

    def save_all_zip(self, file_path, extra_json_data=None):
        """Save all layers into a zip file that includes any referenced images,
        shapefiles, etc. so the file becomes portable and usable on other
        systems.

        """
        log.debug("saving layers in project file: " + file_path)
        layer_info = self.flatten_with_indexes()
        log.debug("layers are " + str(self.layers))
        log.debug("layer info is:\n" + "\n".join([str(s) for s in layer_info]))
        log.debug("layer subclasses:\n" + "\n".join(["%s -> %s" % (t, str(s)) for t, s in Layer.get_subclasses().iteritems()]))

        if extra_json_data is None:
            extra_json_data = {}
        for attr, to_json in self.get_to_json_attrs():
            extra_json_data[attr] = to_json()
        log.debug("extra json data")
        log.debug(str(extra_json_data))
        try:
            zf = zipfile.ZipFile(file_path, mode='w', compression=zipfile.ZIP_DEFLATED)
            zf.writestr("extra json data", json.dumps(extra_json_data))
            for index, layer in layer_info:
                zip_root = "/".join([str(a) for a in index]) + "/"
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
                            basename = os.path.basename(p)
                            if "://" in p:
                                # handle URI format
                                fs, relpath = opener.parse(p)
                                if fs.hassyspath(relpath):
                                    p = fs.getsyspath(relpath)
                                else:
                                    raise RuntimeError("Can't yet handle URIs not on local filesystem")
                            archive_name = zip_root + basename
                            zf.write(p, archive_name, zipfile.ZIP_STORED)

                    try:
                        text = json.dumps(data, indent=4)
                    except Exception, e:
                        log.error("JSON failure, layer %s: data=%s" % (layer.name, repr(data)))
                        errors = []
                        for k, v in data.iteritems():
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

        except RuntimeError, e:
            return "Failed saving %s: %s" % (file_path, e)
        finally:
            zf.close()
        zf = zipfile.ZipFile(file_path)
        print("\n".join(zf.namelist()))
        return ""

    def save_all_text(self, file_path, extra_json_data=None):
        log.debug("saving layers in project file: " + file_path)
        layer_info = self.flatten_with_indexes()
        log.debug("layers are " + str(self.layers))
        log.debug("layer info is:\n" + "\n".join([str(s) for s in layer_info]))
        log.debug("layer subclasses:\n" + "\n".join(["%s -> %s" % (t, str(s)) for t, s in Layer.get_subclasses().iteritems()]))
        project = []
        if extra_json_data is None:
            extra_json_data = {}
        for attr, to_json in self.get_to_json_attrs():
            extra_json_data[attr] = to_json()
        project.append("extra json data")
        project.append(extra_json_data)
        for index, layer in layer_info:
            log.debug("index=%s, layer=%s, path=%s" % (index, layer, layer.file_path))
            data = layer.serialize_json(index)
            if data is not None:
                try:
                    text = json.dumps(data)
                except Exception, e:
                    log.error("JSON failure, layer %s: data=%s" % (layer.name, repr(data)))
                    errors = []
                    for k, v in data.iteritems():
                        small = {k: v}
                        try:
                            _ = json.dumps(small)
                        except Exception:
                            errors.append((k, v))
                    log.error("JSON failures at: %s" % ", ".join(["%s: %s" % (k, v) for k, v in errors]))
                    return "Failed saving data in layer %s.\n\n%s" % (layer.name, e)

                project.append(data)

        try:
            with fsopen(file_path, "wb") as fh:
                fh.write("# -*- MapRoom project file -*-\n")
                text = json.dumps(project, indent=4)
                processed = collapse_json(text, 12)
                fh.write(processed)
                fh.write("\n")
        except Exception, e:
            return "Failed saving %s: %s" % (file_path, e)
        return ""

    def save_layer(self, layer, file_path, loader=None):
        if layer is not None:
            error = loaders.save_layer(layer, file_path, loader)
            if not error:
                layer.name = os.path.basename(layer.file_path)
            return error
        return "No selected layer."


# def test():
#     a = Layer()
#     a.name = "a"
#     b = Layer()
#     b.name = "b"
#     c = Layer()
#     c.name = "c"
#     d = Layer()
#     d.name = "d"
#     e = Layer()
#     e.name = "e"
#     f = Layer()
#     f.name = "f"
#     g = Layer()
#     g.name = "g"
#     h = Layer()
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
