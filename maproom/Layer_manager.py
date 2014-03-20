import os
import shutil
import tempfile
import time
import traceback
import numpy as np
import wx
import Layer
import library.formats.verdat as verdat
import library.rect as rect
from library.accumulator import flatten

from wx.lib.pubsub import pub


class LayerUndo(object):
    def __init__(self):
        # the undo stack is a list of objects, where each object is of the form:
        # { "n" : <int operation number>, "op" : <int operation enum>, "l" : <layer>, "i" : <point or line index in layer>, "p" : ( params depending on op enum ) }
        self.undo_stack = []
        self.undo_stack_next_index = 0
        self.undo_stack_next_operation_number = 0
    
    def refresh(self, **kwargs):
        pass

    def add_undo_operation_to_operation_batch(self, op, layer, index, params):
        self.clear_undo_stack_forward()
        self.undo_stack.append({"n": self.undo_stack_next_operation_number, "op": op, "l": layer, "i": index, "p": params})
        self.undo_stack_next_index = len(self.undo_stack)

    def end_operation_batch(self):
        self.show_undo_redo_debug_dump("end_operation_batch()")
        self.undo_stack_next_operation_number += 1
        self.refresh()

    def delete_undo_operations_for_layer(self, layer):
        self.clear_undo_stack_forward()
        new_stack = []
        for o in self.undo_stack:
            if (o["l"] != layer):
                new_stack.append(o)
        self.undo_stack = new_stack
        self.undo_stack_next_index = len(self.undo_stack)

    def clear_undo_stack_forward(self):
        if (len(self.undo_stack) > self.undo_stack_next_index):
            self.undo_stack = self.undo_stack[0: self.undo_stack_next_index]

    def get_current_undoable_operation_text(self):
        if (self.undo_stack_next_index == 0):
            return ""

        op = self.undo_stack[self.undo_stack_next_index - 1]["op"]

        return self.get_undo_redo_operation_text(op)

    def get_current_redoable_operation_text(self):
        if (self.undo_stack_next_index >= len(self.undo_stack)):
            return ""

        op = self.undo_stack[self.undo_stack_next_index]["op"]

        return self.get_undo_redo_operation_text(op)

    def get_undo_redo_operation_text(self, op):
        if (op == OP_ADD_POINT or op == OP_ADD_LINE):
            return "Add"
        elif (op == OP_DELETE_POINT or op == OP_DELETE_LINE):
            return "Delete"
        elif (op == OP_MOVE_POINT):
            return "Move"
        elif (op == OP_CHANGE_POINT_DEPTH):
            return "Depth Change"

        return ""

    def undo(self):
        if (self.undo_stack_next_index == 0):
            return
        operation_number = self.undo_stack[self.undo_stack_next_index - 1]["n"]
        # here we assume that all operations in the batch are actually from the same layer
        # we also assume that as point and line deletions are undone, the objects come back already in selected state,
        # which is true because they had to be in selected state at the time they were deleted
        layer = self.undo_stack[self.undo_stack_next_index - 1]["l"]
        layer.clear_all_selections()
        # walk backward until we get to a different operation number or hit the start of the stack
        affected_layers = set()
        while (True):
            if (self.undo_stack_next_index == 0 or self.undo_stack[self.undo_stack_next_index - 1]["n"] != operation_number):
                break
            self.undo_operation(self.undo_stack[self.undo_stack_next_index - 1], affected_layers)
            self.undo_stack_next_index -= 1
        for layer in affected_layers:
            pub.sendMessage(('layer', 'points', 'changed'), layer=layer)
        self.show_undo_redo_debug_dump("undo() done")
        self.refresh()

    def undo_operation(self, o, affected_layers):
        operation_number = o["n"]
        op = o["op"]
        layer = o["l"]
        index = o["i"]
        params = o["p"]

        affected_layers.add(layer)

        if (op == OP_ADD_POINT):
            layer.delete_point(index, False)
        elif (op == OP_ADD_LINE):
            layer.delete_line_segment(index, False)
        elif (op == OP_DELETE_POINT):
            ((x, y), z, color, state) = params
            layer.insert_point_at_index(index, (x, y), z, color, state, False)
        elif (op == OP_DELETE_LINE):
            (point_index_1, point_index_2, color, state) = params
            layer.insert_line_segment_at_index(index, point_index_1, point_index_2, color, state, False)
        elif (op == OP_MOVE_POINT):
            (world_d_x, world_d_y) = params
            layer.offset_point(index, world_d_x, world_d_y, False)
        elif (op == OP_CHANGE_POINT_DEPTH):
            (old_depth, new_depth) = params
            layer.points.z[index] = old_depth

    def redo(self):
        if (self.undo_stack_next_index >= len(self.undo_stack)):
            return
        operation_number = self.undo_stack[self.undo_stack_next_index]["n"]
        # walk forward until we get to a different operation number or hit the end of the stack
        affected_layers = set()
        while (True):
            if (self.undo_stack_next_index == len(self.undo_stack) or self.undo_stack[self.undo_stack_next_index]["n"] != operation_number):
                break
            self.redo_operation(self.undo_stack[self.undo_stack_next_index], affected_layers)
            self.undo_stack_next_index += 1
        for layer in affected_layers:
            pub.sendMessage(('layer', 'points', 'changed'), layer=layer)
        self.show_undo_redo_debug_dump("redo() done")
        self.refresh()

    def redo_operation(self, o, affected_layers):
        operation_number = o["n"]
        op = o["op"]
        layer = o["l"]
        index = o["i"]
        params = o["p"]

        affected_layers.add(layer)

        if (op == OP_ADD_POINT):
            ((x, y), z, color, state) = params
            layer.insert_point_at_index(index, (x, y), z, color, state, False)
        elif (op == OP_ADD_LINE):
            (point_index_1, point_index_2, color, state) = params
            layer.insert_line_segment_at_index(index, point_index_1, point_index_2, color, state, False)
        elif (op == OP_DELETE_POINT):
            layer.delete_point(index, False)
        elif (op == OP_DELETE_LINE):
            layer.delete_line_segment(index, False)
        elif (op == OP_MOVE_POINT):
            (world_d_x, world_d_y) = params
            layer.offset_point(index, -world_d_x, -world_d_y, False)
        elif (op == OP_CHANGE_POINT_DEPTH):
            (old_depth, new_depth) = params
            layer.points.z[index] = new_depth

    def show_undo_redo_debug_dump(self, location_message):
        print location_message
        print "the undo_stack is now: "
        if (len(self.undo_stack) <= 100):
            for item in self.undo_stack:
                print "    " + str(item)
        else:
            print "    longer than 100 items"
        print "undo_stack_next_index is now: " + str(self.undo_stack_next_index)


class Layer_manager(LayerUndo):

    """
    Manages the layers (a tree of Layer).

    A "multi_index" is a set of indexes taken in order, that give the location
    of a sub-tree or leaf within a tree. For instance,
    in the tree [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ], the multi_index [ 0 ] refers to subtree [ a, b ],
    the multi_index [ 1, 1, 1 ] refers to the leaf e, and the multi_index [ 3, 0 ] refers to the leaf g.

    The first layer in the overall list and in each sublist is assumed to be a "folder" layer, whose only
    purpose at present is to hold the folder name.
    """

    def __init__(self, project):
        LayerUndo.__init__(self)
        self.project = project
        self.layers = []
        layer = Layer.Layer(self)
        layer.name = "Layers"
        layer.type = "root"
        self.insert_layer([0], layer)
    
    def refresh(self, **kwargs):
        self.project.refresh(**kwargs)

    def flatten(self):
        return self.flatten_recursive(self.layers)

    def flatten_recursive(self, tree):
        result = []

        for item in tree:
            if (isinstance(item, Layer.Layer)):
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

    def load_layer_from_uri(self, uri):
        layer = Layer.Layer(self)
        layer.read_from_file(uri)
        if layer.load_error_string != "":
            print "LAYER LOAD ERROR: %s" % layer.load_error_string
            return None
        index = None
        self.insert_layer(index, layer)
        return layer

    def insert_layer(self, at_multi_index, layer):
        if (at_multi_index == None or at_multi_index == []):
            at_multi_index = [1]

        print "layers are " + str(self.layers)
        print "inserting layer " + str(layer) + " using multi_index = " + str(at_multi_index)
        if (not isinstance(layer, list)):
            if (layer.type == "folder"):
                layer = [layer]
        self.insert_layer_recursive(at_multi_index, layer, self.layers)
        pub.sendMessage(('layer', 'inserted'), manager=self, layer=layer)

    def insert_layer_recursive(self, at_multi_index, layer, tree):
        if (len(at_multi_index) == 1):
            tree.insert(at_multi_index[0], layer)
        else:
            item = tree[at_multi_index[0]]
            self.insert_layer_recursive(at_multi_index[1:], layer, item)

    def remove_layer(self, at_multi_index):
        self.remove_layer_recursive(at_multi_index, self.layers)
        self.refresh(rebuild_tree=True)

    def remove_layer_recursive(self, at_multi_index, tree):
        index = at_multi_index[0]
        if (len(at_multi_index) == 1):
            del tree[index]
        else:
            sublist = tree[index]
            self.remove_layer_recursive(at_multi_index[1:], sublist)

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
            if (isinstance(item, Layer.Layer)):
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

    def layer_is_folder(self, layer):
        return layer.type == "root" or layer.type == "folder"

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

    def count_raster_layers(self):
        n = 0
        for layer in self.flatten():
            if (layer.images != None):
                n += 1
        #
        return n

    def count_vector_layers(self):
        n = 0
        for layer in self.flatten():
            if (layer.points != None or
                    layer.polygons != None):
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

    def save_layer(self, layer, path):
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(path))

        f = open(temp_file, "w")
        had_error = False
        try:
            verdat.write_layer_as_verdat(f, layer)
        except Exception as e:
            had_error = True
            print traceback.format_exc(e)
            if hasattr(e, "points") and e.points != None:
                layer.clear_all_selections(Layer.STATE_FLAGGED)
                for p in e.points:
                    layer.select_point(p, Layer.STATE_FLAGGED)
                self.refresh()
            wx.MessageDialog(
                self.project.control,
                message=e.message,
                caption="Error Saving Verdat File",
                style=wx.OK | wx.ICON_ERROR,
            ).ShowModal()
        finally:
            f.close()
        if (not had_error and temp_file and os.path.exists(temp_file)):
            try:
                shutil.copy(temp_file, path)
            except Exception as e:
                print traceback.format_exc(e)
                wx.MessageDialog(self.project.control,
                                 message="Unable to save file to disk. Make sure you have write permissions to the file.",
                                 caption="Error Saving Verdat File",
                                 style=wx.OK | wx.ICON_ERROR).ShowModal()

    def add_layer(self, name="New Layer"):
        layer = Layer.Layer()
        layer.new()
        self.insert_layer(None, layer)

    def add_folder(self, name="New Folder"):
        # FIXME: doesn't work, so menu/toolbar items are disabled
        folder = Layer.Layer()
        folder.type = "folder"
        folder.name = name
        self.insert_layer(None, folder)

    def delete_selected_layer(self, layer=None):
        if (layer == None):
            layer = self.project.layer_tree_control.get_selected_layer()
            if (layer == None):
                wx.MessageDialog(
                    self.project.control,
                    message="You must select an item in the tree (either a single layer or a tree of layers) in the tree control before attempting to delete.",
                    style=wx.OK | wx.ICON_ERROR
                ).ShowModal()

                return

            if (layer.type == "root"):
                m = "The root node of the layer tree is selected. This will delete all layers in the tree."
            elif (layer.type == "folder"):
                m = "A folder in the layer tree is selected. This will delete the entire sub-tree of layers."
            else:
                m = "An individual layer in the layer tree is selected. This will delete the selected layer, " + layer.name + "."

            dialog = wx.MessageDialog(
                self.project.control,
                caption="Delete",
                message=m,
                style=wx.OK | wx.CANCEL
            )
            if (dialog.ShowModal() != wx.ID_OK):
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

        self.refresh(rebuild_tree=True)

    def merge_layers(self, layer_a, layer_b):
        layer = Layer.Layer()
        layer.type = ".verdat"
        layer.name = "Merged"
        layer.merge_from_source_layers(layer_a, layer_b)
        self.insert_layer(None, layer)
        self.project.layer_tree_control.select_layer(layer)

    def destroy_recursive(self, layer):
        if (self.layer_is_folder(layer)):
            for item in self.get_layer_children(layer):
                self.destroy_recursive(item)
        layer.destroy()


def test():
    """
    a = Layer.Layer()
    a.name = "a"
    b = Layer.Layer()
    b.name = "b"
    c = Layer.Layer()
    c.name = "c"
    d = Layer.Layer()
    d.name = "d"
    e = Layer.Layer()
    e.name = "e"
    f = Layer.Layer()
    f.name = "f"
    g = Layer.Layer()
    g.name = "g"
    h = Layer.Layer()
    h.name = "h"

    tree = [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ]
    print a
    print tree

    lm = Layer_manager()
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
    lm.remove_layer( [ 0, 0 ] )
    print lm.layers
    print "removing d"
    lm.remove_layer( [ 1, 1, 0 ] )
    print lm.layers
    print "removing e"
    lm.remove_layer( [ 1, 1, 0 ] )
    print lm.layers
    """

    """
    print "inserting a"
    lm.insert_layer( [ 1, 1, 1 ], a )
    print lm.layers
    """
