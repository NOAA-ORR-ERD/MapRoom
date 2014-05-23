import os
import shutil
import tempfile
import time
import traceback
import numpy as np
import wx
import library.formats.verdat as verdat
import library.rect as rect
from library.accumulator import flatten

from layer_undo import LayerUndo
from layers import Layer, RootLayer, Grid, LineLayer, TriangleLayer, RasterLayer, constants, loaders

# Enthought library imports.
from traits.api import HasTraits, Int, Any, List, Set, Bool, Event, Dict
from pyface.api import YES, NO, GUI


class LayerManager(LayerUndo):

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
    
    layers = List(Any)
    
    # A mapping of a layer to all layers that are built from the layer,
    # as if it were a parent/child relationship.  dependents[layer]
    # yields another mapping of dependent layer type to actual layer,
    # so dependents[layer]["triangles"] is another layer that is the
    # triangulation of layer
    dependents = Dict(Any)
    
    batch = Bool
    
    events = List(Any)
    
    layer_loaded = Event
    
    layers_changed = Event
    
    layer_contents_changed = Event
    
    # when points are deleted from a layer the indexes of the points in the
    # merge dialog box become invalid; so this event will trigger the user to
    # re-find duplicates in order to create a valid list again
    layer_contents_deleted = Event
    
    layer_metadata_changed = Event
    
    projection_changed = Event
    
    refresh_needed = Event

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
        layer = RootLayer(manager=self)
        self.insert_layer([0], layer)
        grid = Grid(manager=self)
        self.insert_layer([1], grid)
        return self
    
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
    
    def get_default_visibility(self):
        layer_visibility = dict()
        for layer in self.flatten():
            layer_visibility[layer] = layer.get_visibility_dict()
        return layer_visibility

    def destroy(self):
        for layer in self.flatten():
            layer.destroy()
        self.layers = []

    def load_layer_from_metadata(self, metadata, editor=None):
        # FIXME: load all layer types, not just vector!
        layer = loaders.load_layer(metadata, manager=self)
        if layer is None:
            print "LAYER LOAD ERROR: %s" % "Unknown file type %s for %s" % (metadata.mime, metadata.uri)
            return None
            
        if layer.load_error_string != "":
            print "LAYER LOAD ERROR: %s" % layer.load_error_string
            return None
        
        layer.check_projection(editor.window)
        if layer.load_error_string != "":
            print "LAYER LOAD ERROR: %s" % layer.load_error_string
            return None
        
        self.insert_loaded_layer(layer, editor)
        return layer
    
    def check_layer(self, layer, window):
        if layer is not None:
            try:
                message = loaders.check_layer(layer)
                window.information(message, "No Errors Found")
            except Exception, e:
                if hasattr(e, "points") and e.points != None:
                    layer.clear_all_selections(constants.STATE_FLAGGED)
                    for p in e.points:
                        layer.select_point(p, constants.STATE_FLAGGED)
                self.dispatch_event('refresh_needed')
                window.error(e.message, "File Contains Errors")
        return "No selected layer."
    
    def save_layer(self, layer, file_path):
        if layer is not None:
            return loaders.save_layer(layer, file_path)
        return "No selected layer."
    
    def insert_loaded_layer(self, layer, editor=None, before=None, after=None):
        self.dispatch_event('layer_loaded', layer)
        mi = self.get_insertion_multi_index(before, after)
        self.insert_layer(mi, layer)
        if editor is not None:
            GUI.invoke_later(editor.layer_tree_control.select_layer, layer)
    
    def find_default_insert_layer(self):
        # By default, lat/lon layers stay at the top and other layers will
        # be inserted below them.  If the lat/lon layer has been moved down,
        # other layers will be inserted in the first position.
        pos = 0
        for layer in self.layers:
            if layer.skip_on_insert:
                pos += 1
            else:
                break
        return [pos]
    
    def get_insertion_multi_index(self, before=None, after=None):
        if before is not None:
            mi = self.get_multi_index_of_layer(before)
            print mi
            mi[-1] -= 1
        elif after is not None:
            mi = self.get_multi_index_of_layer(after)
            print mi
            mi[-1] += 1
        else:
            mi = None
        print mi
        return mi

    def insert_layer(self, at_multi_index, layer):
        if (at_multi_index == None or at_multi_index == []):
            at_multi_index = self.find_default_insert_layer()

        print "layers are " + str(self.layers)
        print "inserting layer " + str(layer) + " using multi_index = " + str(at_multi_index)
        if (not isinstance(layer, list)):
            if (layer.type == "folder"):
                layer = [layer]
        self.insert_layer_recursive(at_multi_index, layer, self.layers)
        self.dispatch_event('layers_changed')

    def insert_layer_recursive(self, at_multi_index, layer, tree):
        if (len(at_multi_index) == 1):
            tree.insert(at_multi_index[0], layer)
        else:
            item = tree[at_multi_index[0]]
            self.insert_layer_recursive(at_multi_index[1:], layer, item)

    def remove_layer(self, layer):
        mi = self.get_multi_index_of_layer(layer)
        self.remove_layer_at_multi_index(mi)

    def remove_layer_at_multi_index(self, at_multi_index):
        layer = self.remove_layer_recursive(at_multi_index, self.layers)
        if layer in self.dependents:
            # remove the parent/child relationship of dependent layers
            del self.dependents[layer]
        self.dispatch_event('layers_changed')

    def remove_layer_recursive(self, at_multi_index, tree):
        index = at_multi_index[0]
        if (len(at_multi_index) == 1):
            layer = tree[index]
            del tree[index]
            return layer
        else:
            sublist = tree[index]
            return self.remove_layer_recursive(at_multi_index[1:], sublist)

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

    def get_multi_index_of_layer_recursive(self, layer, tree):
        for i, item in enumerate(tree):
            if (isinstance(item, Layer)):
                if (item == layer):
                    # in the case of folders, we return the multi-index to the parent,
                    # since the folder "layer" itself is just a pseudo-layer
                    if (layer.type == "folder" or layer.type == "root"):
                        return []
                    else:
                        return [i]
            else:
                result = self.get_multi_index_of_layer_recursive(layer, item)
                if (result != None):
                    r = [i]
                    r.extend(result)

                    return r

        return None
    
    def find_dependent_layer(self, layer, dependent_type):
        if layer in self.dependents:
            if dependent_type in self.dependents[layer]:
                return self.dependents[layer][dependent_type]
        return None
    
    def set_dependent_layer(self, layer, dependent_type, dependent_layer):
        d = self.find_dependent_layer(layer, dependent_type)
        if d is not None:
            self.remove_layer(d)
        if layer not in self.dependents:
            self.dependents[layer] = {}
        self.dependents[layer][dependent_type] = dependent_layer

    def layer_is_folder(self, layer):
        return layer.type == "root" or layer.type == "folder"

    def is_raisable(self, layer):
        if layer.type != "root":
            mi = self.get_multi_index_of_layer(layer)
            if mi is not None:
                return mi[len(mi) - 1] >= 2
        return False

    def is_lowerable(self, layer):
        if layer.type != "root":
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
        n = 0
        for layer in self.flatten():
            if (hasattr(layer, "images") and layer.images != None):
                n += 1
        #
        return n

    def count_vector_layers(self):
        n = 0
        for layer in self.flatten():
            if (hasattr(layer, "points") and (layer.points != None or
                    layer.polygons != None)):
                n += 1
        #
        return n

    def reproject_all(self, projection, projection_is_identity):
        for layer in self.flatten():
            layer.reproject(projection, projection_is_identity)

    def accumulate_layer_rects(self, layer_visibility, only_visible_layers=True):
        result = rect.NONE_RECT

        if (len(self.layers) == 0):
            return result

        for layer in self.flatten():
            if (only_visible_layers and not layer_visibility[layer]["layer"]):
                continue

            if (result == rect.NONE_RECT):
                result = layer.bounds
            else:
                result = rect.accumulate_rect(result, layer.bounds)

        return result

    def render(self, render_window, pick_mode=False):
        list = self.flatten()
        length = len(list)
        for i, layer in enumerate(reversed(list)):
            layer.render(render_window, (length - 1 - i) * 10, pick_mode)

    def add_layer(self, type=None, editor=None, before=None, after=None):
        if type == "grid":
            layer = Grid(manager=self)
        elif type == "triangle":
            layer = TriangleLayer(manager=self)
        else:
            layer = LineLayer(manager=self)
        layer.new()
        self.insert_loaded_layer(layer, editor, before, after)
        return layer

    def add_folder(self, name="New Folder"):
        # FIXME: doesn't work, so menu/toolbar items are disabled
        folder = Layer()
        folder.type = "folder"
        folder.name = name
        self.insert_layer(None, folder)

    def delete_selected_layer(self, layer=None):
        if (layer == None):
            layer = self.project.layer_tree_control.get_selected_layer()
        window = self.project.window
        if (layer == None):
            window.status_bar.message = "Selected layer to delete!."
            return

        if (layer.type == "root"):
            m = "The root node of the layer tree is selected. This will delete all layers in the tree."
        elif (layer.type == "folder"):
            m = "A folder in the layer tree is selected. This will delete the entire sub-tree of layers."
        else:
            m = "Are you sure you want to delete " + layer.name + "?"

        if window.confirm(m) != YES:
            return

        self.destroy_recursive(layer)

        mi = self.get_multi_index_of_layer(layer)
        if (mi == []):
            self.layers = self.layers[0: 1]
        else:
            l = self.layers
            for index in mi[0: -1]:
                l = l[index]
            del l[mi[-1]]
        
        self.project.control.remove_renderer_for_layer(layer)

        self.dispatch_event('layers_changed')
    
    def get_mergeable_layers(self):
        layers = [layer for layer in self.flatten() if layer.has_points()]
        layers.reverse()
        return layers

    def merge_layers(self, layer_a, layer_b):
        layer = LineLayer(manager=self)
        layer.type = "line"
        layer.name = "Merged"
        layer.merge_from_source_layers(layer_a, layer_b)
        self.dispatch_event('layer_loaded', layer)
        self.insert_layer(None, layer)
        self.project.layer_tree_control.select_layer(layer)

    def destroy_recursive(self, layer):
        if (self.layer_is_folder(layer)):
            for item in self.get_layer_children(layer):
                self.destroy_recursive(item)
        self.delete_undo_operations_for_layer(layer)


def test():
    """
    a = Layer()
    a.name = "a"
    b = Layer()
    b.name = "b"
    c = Layer()
    c.name = "c"
    d = Layer()
    d.name = "d"
    e = Layer()
    e.name = "e"
    f = Layer()
    f.name = "f"
    g = Layer()
    g.name = "g"
    h = Layer()
    h.name = "h"

    tree = [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ]
    print a
    print tree

    lm = LayerManager()
    lm.layers = tree

    print "lm.flatten()"
    print lm.flatten()

    for layer in [ a, b, c, d, e, f, g, h ]:
        print "lm.get_multi_index_of_layer( {0} )".format( layer )
        mi = lm.get_multi_index_of_layer( layer )
        print mi
        print lm.get_layer_by_multi_index( mi )
    """

    """
    print "removing a"
    lm.remove_layer_at_multi_index( [ 0, 0 ] )
    print lm.layers
    print "removing d"
    lm.remove_layer_at_multi_index( [ 1, 1, 0 ] )
    print lm.layers
    print "removing e"
    lm.remove_layer_at_multi_index( [ 1, 1, 0 ] )
    print lm.layers
    """

    """
    print "inserting a"
    lm.insert_layer( [ 1, 1, 1 ], a )
    print lm.layers
    """
