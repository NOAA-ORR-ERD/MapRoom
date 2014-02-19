import os
import shutil
import tempfile
import time
import traceback
import numpy as np
import wx
from xml.etree.ElementTree import ElementTree
import Layer
import library.formats.verdat as verdat
import library.rect as rect
from library.accumulator import flatten
import app_globals

from wx.lib.pubsub import pub


class Layer_manager():

    """
    Manages the layers (a tree of Layer).

    A "multi_index" is a set of indexes taken in order, that give the location
    of a sub-tree or leaf within a tree. For instance,
    in the tree [ [ a, b ], [c, [ d, e ] ], f, [ g, h ] ], the multi_index [ 0 ] refers to subtree [ a, b ],
    the multi_index [ 1, 1, 1 ] refers to the leaf e, and the multi_index [ 3, 0 ] refers to the leaf g.

    The first layer in the overall list and in each sublist is assumed to be a "folder" layer, whose only
    purpose at present is to hold the folder name.
    """

    def __init__(self):
        self.layers = []
        layer = Layer.Layer()
        layer.name = "Layers"
        layer.type = "root"
        self.insert_layer([0], layer)
        self.current_layer = None
        #self.add_folder( name = "folder_a" )
        #self.add_folder( name = "folder_b" )
        #self.add_folder( name = "folder_c" )

        pub.subscribe(self.on_layer_selection_changed, ('layer', 'selection', 'changed'))

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

    def destroy(self):
        for layer in self.flatten():
            layer.destroy()

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
        app_globals.application.refresh(rebuild_tree=True)

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

    def on_layer_selection_changed(self, manager, layer):
        if manager == self:
            print "Setting current layer to %r" % layer
            self.current_layer = layer

    def get_selected_layer(self):
        return self.current_layer

    def is_layer_selected(self, layer):
        return app_globals.application.layer_tree_control.get_selected_layer() == layer

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

    def accumulate_layer_rects(self, only_visible_layers=True):
        result = rect.NONE_RECT

        if (len(self.layers) == 0):
            return result

        for layer in self.flatten():
            if (only_visible_layers and not layer.is_visible):
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

    #

    def write_layer_as_xml_element(self, direcotry_path, file_name, f, indent, layer):
        i0 = " " * indent
        i2 = i0 + "  "
        i4 = i2 + "  "
        f.write(i0 +
                "<layer name=\"{0}\" type=\"{1}\" is_expanded=\"{2}\" is_visible=\"{3}\" file_path=\"{4}\" depth_unit=\"{5}\" default_depth=\"{6}\">\n".format(
                    layer.name, layer.type, str(layer.is_expanded), str(layer.is_visible), layer.file_path, layer.depth_unit, str(layer.default_depth)))

        if (not self.layer_is_folder(layer)):
            (f_name, ext) = os.path.splitext(file_name)
            c = 0
            while (True):
                data_file_name = f_name + "_" + str(c) + ext
                data_file_path = os.path.join(direcotry_path, data_file_name)
                if (not os.path.exists(data_file_path)):
                    break
                c += 1

            f.write(i2 + "<data_file file_name=\"{0}\" />\n".format(data_file_name))

            f_sub = open(data_file_path, "w")
            f_sub.write("<?xml version=\"1.0\" ?>\n")
            f_sub.write("<data>\n")

            if (layer.points != None):
                t0 = time.clock()
                xs = list(layer.points.x)
                ys = list(layer.points.y)
                zs = list(layer.points.z)
                cs = list(layer.points.color)
                ss = list(layer.points.state)
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "retrieved point coordinates in {0} seconds".format(t)
                t0 = time.clock()
                f_sub.write("  <points points_visible=\"{0}\" labels_visible=\"{1}\">\n".format(layer.points_visible, layer.labels_visible))
                for i in xrange(len(xs)):
                    f_sub.write("    <p x=\"{0}\" y=\"{1}\" z=\"{2}\" c=\"{3}\" s=\"{4}\" />\n".format(
                        str(xs[i]), str(ys[i]), str(zs[i]), str(cs[i]), str(ss[i])))
                f_sub.write("  </points>\n")
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "saved points in {0} seconds".format(t)

            if (layer.line_segment_indexes != None):
                t0 = time.clock()
                p1s = list(layer.line_segment_indexes.point1)
                p2s = list(layer.line_segment_indexes.point2)
                cs = list(layer.line_segment_indexes.color)
                ss = list(layer.line_segment_indexes.state)
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "retrieved point coordinates in {0} seconds".format(t)
                t0 = time.clock()
                f_sub.write("  <line_segment_indexes line_segments_visible=\"{0}\">\n".format(layer.line_segments_visible))
                for i in xrange(len(p1s)):
                    f_sub.write("    <l p1=\"{0}\" p2=\"{1}\" c=\"{2}\" s=\"{3}\" />\n".format(
                        str(p1s[i]), str(p2s[i]), str(cs[i]), str(ss[i])))
                f_sub.write("  </line_segment_indexes>\n")
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "saved line segment indexes in {0} seconds".format(t)

            if (layer.triangle_points != None):
                t0 = time.clock()
                xs = list(layer.triangle_points.x)
                ys = list(layer.triangle_points.y)
                zs = list(layer.triangle_points.z)
                cs = list(layer.triangle_points.color)
                ss = list(layer.triangle_points.state)
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "retrieved triangle point coordinates in {0} seconds".format(t)
                t0 = time.clock()
                f_sub.write("  <triangle_points triangles_visible=\"{0}\">\n".format(layer.triangles_visible))
                for i in xrange(len(xs)):
                    f_sub.write("    <p x=\"{0}\" y=\"{1}\" z=\"{2}\" c=\"{3}\" s=\"{4}\" />\n".format(
                        str(xs[i]), str(ys[i]), str(zs[i]), str(cs[i]), str(ss[i])))
                f_sub.write("  </triangle_points>\n")
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "saved triangle points in {0} seconds".format(t)

            if (layer.triangles != None):
                t0 = time.clock()
                p1s = list(layer.triangles.point1)
                p2s = list(layer.triangles.point2)
                p3s = list(layer.triangles.point3)
                cs = list(layer.triangles.color)
                ss = list(layer.triangles.state)
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "retrieved triangle indexes in {0} seconds".format(t)
                t0 = time.clock()
                f_sub.write("  <triangle_indexes>\n")
                for i in xrange(len(p1s)):
                    f_sub.write("    <l p1=\"{0}\" p2=\"{1}\" p3=\"{2}\" c=\"{3}\" s=\"{4}\" />\n".format(
                        str(p1s[i]), str(p2s[i]), str(p3s[i]), str(cs[i]), str(ss[i])))
                f_sub.write("  </triangle_indexes>\n")
                t = time.clock() - t0  # t is wall seconds elapsed (floating point)
                print "saved triangle indexes in {0} seconds".format(t)

            if (layer.polygons != None):
                pass

            if (layer.images != None):
                pass

            f_sub.write("</data>\n")
            f_sub.close()

        else:
            for item in self.get_layer_children(layer):
                self.write_layer_as_xml_element(direcotry_path, file_name, f, indent + 2, item)

        f.write(i0 + "</layer>\n")

    def save_layer(self, layer, path):
        if (path.endswith(".verdat")):
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
                    app_globals.application.refresh()
                wx.MessageDialog(
                    app_globals.application.frame,
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
                    wx.MessageDialog(app_globals.application.frame,
                                     message="Unable to save file to disk. Make sure you have write permissions to the file.",
                                     caption="Error Saving Verdat File",
                                     style=wx.OK | wx.ICON_ERROR).ShowModal()

        if (path.endswith(".xml")):
            # xml.etree.ElementTree doesn't format the output, so it's useless for our purposes; we write the xml by hand instead
            (p, f_name) = os.path.split(path)
            f = open(path, "w")
            f.write("<?xml version=\"1.0\" ?>\n")
            self.write_layer_as_xml_element(p, f_name, f, 0, layer)
            f.close()
            print "done saving"

    def load_xml_layer(self, path):
        tree = ElementTree()
        tree.parse(path)
        (directory_path, file_name) = os.path.split(path)
        self.load_xml_layer_recursive(tree.getroot(), [], directory_path)

    def load_xml_layer_recursive(self, element, multi_index, directory_path):
        print "load_xml_layer_recursive called " + multi_index.__repr__()
        layer = Layer.Layer()
        layer.name = element.get("name")
        layer.type = element.get("type")
        layer.is_expanded = element.get("is_expanded") == "True"
        layer.is_visible = element.get("is_visible") == "True"
        layer.file_path = element.get("file_path")
        layer.default_depth = float(element.get("default_depth"))
        layer.determine_layer_color()
        for i, child in enumerate(list(element)):
            if (child.tag == "data_file"):
                self.load_xml_layer_data(layer, os.path.join(directory_path, child.get("file_name")))
            elif (child.tag == "layer"):
                mi = multi_index[:]
                # remember that multi-indexes are 1-based because the folder "layer" itself is always in position 0
                mi.append(i + 1)
                self.load_xml_layer_recursive(child, mi, directory_path)
        if (layer.type != "root"):
            print "inserting layer " + layer.name + " at " + multi_index.__repr__()
            self.insert_layer(multi_index, layer)
            if (multi_index == []):
                multi_index = [1]

    def load_xml_layer_data(self, layer, path):
        print "load_xml_layer_data called for layer " + layer.name + ", path = " + path
        tree = ElementTree()
        tree.parse(path)
        e = tree.getroot()
        for child in list(e):
            if (child.tag == "points"):
                layer.points_visible = child.get("points_visible") == "True"
                layer.labels_visible = child.get("labels_visible") == "True"
                layer.points = layer.make_points(len(list(child)))
                xs = []
                ys = []
                zs = []
                cs = []
                ss = []
                for c in list(child):
                    xs.append(float(c.get("x")))
                    ys.append(float(c.get("y")))
                    zs.append(float(c.get("z")))
                    cs.append(int(c.get("c")))
                    ss.append(int(c.get("s")))
                layer.points.x = xs
                layer.points.y = ys
                layer.points.z = zs
                layer.points.color = cs
                layer.points.state = ss

            elif (child.tag == "line_segment_indexes"):
                layer.line_segments_visible = child.get("line_segments_visible") == "True"
                layer.line_segment_indexes = layer.make_line_segment_indexes(len(list(child)))
                p1s = []
                p2s = []
                cs = []
                ss = []
                for c in list(child):
                    p1s.append(int(c.get("p1")))
                    p2s.append(int(c.get("p2")))
                    cs.append(int(c.get("c")))
                    ss.append(int(c.get("s")))
                layer.line_segment_indexes.point1 = p1s
                layer.line_segment_indexes.point2 = p2s
                layer.line_segment_indexes.color = cs
                layer.line_segment_indexes.state = ss

            if (child.tag == "triangle_points"):
                layer.triangles_visible = child.get("triangles_visible") == "True"
                layer.triangle_points = layer.make_points(len(list(child)))
                xs = []
                ys = []
                zs = []
                cs = []
                ss = []
                for c in list(child):
                    xs.append(float(c.get("x")))
                    ys.append(float(c.get("y")))
                    zs.append(float(c.get("z")))
                    cs.append(int(c.get("c")))
                    ss.append(int(c.get("s")))
                layer.triangle_points.x = xs
                layer.triangle_points.y = ys
                layer.triangle_points.z = zs
                layer.triangle_points.color = cs
                layer.triangle_points.state = ss

            elif (child.tag == "triangle_indexes"):
                layer.triangles = layer.make_triangles(len(list(child)))
                p1s = []
                p2s = []
                p3s = []
                cs = []
                ss = []
                for c in list(child):
                    p1s.append(int(c.get("p1")))
                    p2s.append(int(c.get("p2")))
                    p3s.append(int(c.get("p3")))
                    cs.append(int(c.get("c")))
                    ss.append(int(c.get("s")))
                layer.triangles.point1 = p1s
                layer.triangles.point2 = p2s
                layer.triangles.point3 = p3s
                layer.triangles.color = cs
                layer.triangles.state = ss

        layer.bounds = layer.compute_bounding_rect()
        pub.sendMessage(('layer', 'loaded'), layer=layer)

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
            layer = app_globals.application.layer_tree_control.get_selected_layer()
            if (layer == None):
                wx.MessageDialog(
                    app_globals.application.frame,
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
                app_globals.application.frame,
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

        app_globals.application.refresh(rebuild_tree=True)

    def merge_layers(self, layer_a, layer_b):
        layer = Layer.Layer()
        layer.type = ".verdat"
        layer.name = "Merged"
        layer.merge_from_source_layers(layer_a, layer_b)
        self.insert_layer(None, layer)
        app_globals.application.layer_tree_control.select_layer(layer)

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