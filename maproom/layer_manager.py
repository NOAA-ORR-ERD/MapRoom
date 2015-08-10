import os
import shutil
import tempfile
import time
import traceback
import json

import numpy as np
import library.rect as rect
from library.accumulator import flatten

from layers import Layer, RootLayer, Grid, LineLayer, TriangleLayer, RasterLayer, AnnotationLayer, constants, loaders, LayerStyle
from command import UndoStack
from renderer import color_floats_to_int, int_to_color_floats

# Enthought library imports.
from traits.api import HasTraits, Int, Any, List, Set, Bool, Event, Dict, Set
from pyface.api import YES, NO, GUI


import logging
log = logging.getLogger(__name__)


class LayerManager(HasTraits):

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
    
    undo_stack = Any
    
    layers = List(Any)
    
    next_invariant = Int(0)
    
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

    pick_layer_index_map = {} # fixme: managed by the layer_control_wx -- horrible coupling!
    
    # Linked control points are slaves of a truth layer: a dict that maps the
    # dependent layer/control point to the truth layer/control point
    control_point_links = Dict(Any)

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
        self.undo_stack = UndoStack()
        self.next_invariant = -1  # preset so first user added layer will use 1
        self.default_style = LayerStyle()
        layer = RootLayer(manager=self)
        self.insert_layer([0], layer)
        grid = Grid(manager=self)
        self.insert_layer([1], grid)
        
        # Add hook to create layer instances for debugging purposes
        if "--debug-objects" in self.project.window.application.command_line_args:
            import debug
            debug.debug_objects(self)
        return self
    
    def __str__(self):
        root = self.get_layer_by_multi_index([0])
        layers = self.get_children(root)
        return str(layers)
    
    def update_default_style(self, style):
        self.default_style.copy_from(style)
    
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
    
    # Invariant handling: invariants are unique identifiers for each layer that
    # don't change when the layer is renamed or reordered
    
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

    def get_layer_by_invariant(self, invariant):
        layers = self.flatten()
        for layer in layers:
            if layer.invariant == invariant:
                return layer
        return None
    
    def get_invariant_offset(self):
        return self.next_invariant - 1
    
    def has_user_created_layers(self):
        """Returns true if all the layers can be recreated automatically
        
        If any layer has any user-created data, this will return False.
        """
        for layer in self.flatten():
            if not layer.skip_on_insert:
                return True
        return False
    
    def get_default_visibility(self):
        layer_visibility = dict()
        for layer in self.flatten():
            layer_visibility[layer] = layer.get_visibility_dict()
        return layer_visibility

    def destroy(self):
        ## fixme: why do layers need a destroy() method???
        for layer in self.flatten():
            layer.destroy()
        self.layers = []
    
    def dispatch_event(self, event, value=True):
        log.debug("dispatching event %s = %s" % (event, value))
        setattr(self, event, value)
    
    def post_event(self, event_name, *args):
        log.debug("event: %s.  args=%s" % (event_name, str(args)))
        
    def get_event_callback(self, event):
        import functools
        callback = functools.partial(self.post_event, event)
        return callback
    
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
            layer.check_projection(editor.window)
            if not layer.load_error_string:
                self.insert_loaded_layer(layer, editor, last_child_of=parent)
        return layers
    
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
    
    def load_all_from_json(self, json):
        order = []
        if json[0] == "extra json data":
            extra_json = json[1]
            json = json[2:]
        else:
            extra_json = None
        for serialized_data in json:
            loaded = Layer.load_from_json(serialized_data, self)
            index = serialized_data['index']
            order.append((index, loaded))
            print "processed json from layer", loaded
        order.sort()
        print "load_all_from_json: order", order
        return order, extra_json
    
    def add_all(self, layer_order, editor=None):
        existing = self.flatten()
        for layer in existing:
            if not layer.is_root():
                self.remove_layer(layer)
        layers = []
        print "layer order", layer_order
        for mi, layer_list in layer_order:
            for layer in layer_list:
                print "adding ", layer, "at", mi
                if editor is not None:
                    layer.check_projection(editor.window)
                if len(mi) > 1:
                    # check to see if it's a folder layer, and if so it would
                    # have been serialized with a 0 as the last multi index.
                    # On insert, this barfs, so strip it off.
                    if mi[-1] == 0:
                        mi = mi[:-1]
                self.insert_loaded_layer(layer, mi=mi)
                layers.append(layer)
        return layers
    
    def save_all(self, file_path, extra_json_data=None):
        log.debug("saving layers in project file: " + file_path)
        layer_info = self.flatten_with_indexes()
        log.debug("layers are " + str(self.layers))
        log.debug("layer info is:\n" + "\n".join([str(s) for s in layer_info]))
        log.debug("layer subclasses:\n" + "\n".join(["%s -> %s" % (t, str(s)) for t,s  in Layer.get_subclasses().iteritems()]))
        project = []
        if extra_json_data is not None:
            project.append("extra json data")
            project.append(extra_json_data)
        for index, layer in layer_info:
            print "index=%s, layer=%s, path=%s" % (index, layer, layer.file_path)
            data = layer.serialize_json(index)
            if data is not None:
                project.append(data)
        
        try:
            with open(file_path, "w") as fh:
                fh.write("# -*- MapRoom project file -*-\n")
                json.dump(project, fh)
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
    
    def insert_loaded_layer(self, layer, editor=None, before=None, after=None, invariant=None, first_child_of=None, last_child_of=None, mi=None):
        self.dispatch_event('layer_loaded', layer)
        if mi is None:
            mi = self.get_insertion_multi_index(before, after, first_child_of, last_child_of)
        self.insert_layer(mi, layer, invariant=invariant)
    
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

    def insert_layer(self, at_multi_index, layer, invariant=None):
        if (at_multi_index is None or at_multi_index == []):
            at_multi_index = self.find_default_insert_layer()

        log.debug("before: layers are " + str(self.layers))
        log.debug("inserting layer " + str(layer) + " using multi_index = " + str(at_multi_index))
        if (not isinstance(layer, list)):
            if invariant is None:
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

    # FIXME: layer removal commands should return the hierarchy of layers
    # removed so that the operation can be undone correctly.
    def remove_layer(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        self.remove_layer_at_multi_index(mi)

    def remove_layer_at_multi_index(self, at_multi_index):
        layer = self.remove_layer_recursive(at_multi_index, self.layers)

    def remove_layer_recursive(self, at_multi_index, tree):
        index = at_multi_index[0]
        if (len(at_multi_index) == 1):
            layer = tree[index]
            del tree[index]
            return layer
        else:
            sublist = tree[index]
            return self.remove_layer_recursive(at_multi_index[1:], sublist)

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

    def get_multi_index_of_layer(self, layer):
        return self.get_multi_index_of_layer_recursive(layer, self.layers)

    def get_layer_by_pick_index(self, pick_index):
        try:
            layer = self.get_layer_by_flattened_index(self.pick_layer_index_map[pick_index])
        except:
            log.error("Invalid pick_index: %s" % pick_index)
            layer = None
        return layer

    def get_layer_by_name(self, name):
        layers = self.flatten()
        for layer in layers:
            if layer.name == name:
                return layer
        return None

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
    
    def find_dependent_layer(self, layer, dependent_type):
        for child in self.flatten():
            if child.dependent_of == layer.invariant:
                return child
        return None
    
    ## fixme -- why wouldn't is_raisable, etc be an attribute of the layer???    
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

    # returns a list of the child layers of a root or folder layer
    def get_layer_children(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        l = self.get_layer_by_multi_index(mi)

        ret = []
        for item in l[1:]:
            i = item
            # a list means the first element in the list is the folder layer containing the other elements in the list
            if (isinstance(item, list)):
                i = item[0]
            ret.append(i)

        return ret

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

    def get_layer_multi_index_from_file_path(self, file_path):
        for layer in self.flatten():
            if (layer.file_path == file_path):
                return self.get_multi_index_of_layer(layer)
        #
        return None

    def get_layer_by_flattened_index(self, index):
        flattened = self.flatten()
        if index < len(flattened):
            return flattened[index]

        return None

    def count_layers(self):
        n = 0
        for layer in self.flatten():
            if not layer.is_root():
                n += 1
        #
        return n

    def count_raster_layers(self):
        ## fixme -- what  in the world are these used for?
        ## and if there is a need, maybe it should be more  like 
        ## count_layer_of_type(self, layer_type="")
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
                    layer.polygons is not None)):
                n += 1
        #
        return n

    def accumulate_layer_rects(self, layer_visibility, only_visible_layers=True):
        result = rect.NONE_RECT

        if (len(self.layers) == 0):
            return result

        layers = []
        for layer in self.flatten():
            if (only_visible_layers and not layer_visibility[layer]["layer"]):
                continue
            layers.append(layer)

        return self.accumulate_layer_bounds_from_list(layers)

    def accumulate_layer_bounds_from_list(self, layers):
        result = rect.NONE_RECT

        for layer in layers:
            if (result == rect.NONE_RECT):
                result = layer.bounds
            else:
                result = rect.accumulate_rect(result, layer.bounds)

        return result

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
            GUI.invoke_later(editor.layer_tree_control.select_layer, layer)
        return layer

    def get_mergeable_layers(self):
        layers = [layer for layer in self.flatten() if layer.has_points()]
        layers.reverse()
        return layers


    def set_control_point_link(self, dep_or_layer, truth_or_cp, truth_layer=None, truth_cp=None):
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
        log.debug("control_point_links: adding %s from %s" % (entry, truth))
        self.control_point_links[entry] = truth
    
    def remove_control_point_links(self, layer, remove_cp=-1):
        """Remove links to truth layer control points from the specified
        dependent layer.
        
        If a remove_cp is specified, only remove that control point's
        reference, otherwise remove all control points links that are on the
        dependent layer.
        """
        to_remove = []
        for dep, truth in self.control_point_links.iteritems():
            #print "checking", self.control_point_links[dep]
            dep_layer_invariant, dep_cp = dep[0], dep[1]
            if dep_layer_invariant == layer.invariant and (remove_cp < 0 or remove_cp == dep_cp):
                to_remove.append((dep, truth))
        for dep, truth in to_remove:
            log.debug("control_point_links: removing %s from %s" % (dep, truth))
            del self.control_point_links[dep]
        return to_remove
    
    def update_linked_control_points(self):
        """Update control points in depedent layers from the truth layers.
        
        The truth_layer is the layer that control point values are taken from
        and propagated to the dependent layer
        """
        layers = []
        for dep, truth in self.control_point_links.iteritems():
            truth_layer, truth_cp = self.get_layer_by_invariant(truth[0]), truth[1]
            dep_layer, dep_cp = self.get_layer_by_invariant(dep[0]), dep[1]
            dep_layer.copy_control_point_from(dep_cp, truth_layer, truth_cp)
            layers.append(dep_layer)
        return layers

    def destroy_recursive(self, layer):
        if (layer.is_folder()):
            for item in self.get_layer_children(layer):
                self.destroy_recursive(item)
        self.delete_undo_operations_for_layer(layer)


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
