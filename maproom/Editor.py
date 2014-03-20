import os
import sys
import wx
import library.rect as rect
import Layer
import Layer_manager
import library.Opengl_renderer.Point_and_line_set_renderer as Point_and_line_set_renderer
import library.Opengl_renderer.Polygon_set_renderer as Polygon_set_renderer

from wx.lib.pubsub import pub

"""
    editing rules:
        - click on point tool deselects any selected lines (DONE)
        - click on line tool deselects all selected points, unless there is zero or one selected point (DONE)
        - Esc key clears all selections (DONE)
        - click on Clear Selection button clears all selections (DONE)
        - selecting a different line in the tree clears all selections (DONE)
        - moving a layer up or down in the tree clears all selections (DONE)
        - Delete key or click on Delete button deletes all selected points and lines (and any lines using the deleted points) (DONE)
        
        - with point tool
            - mouse-over point changes cursor to hand (DONE)
            - mouse-over line changes cursor to bull's eye (DONE)
            - click on line:
                - if either Shift or Control key is down, do nothing (DONE)
                - else insert point into line, select only the new point, and
                  transition into drag mode (DONE)
            - click on point:
                - if Control key down toggle selection of the point (DONE)
                - else if Shift key down:
                    - if the point is already selected, do nothing (DONE)
                    - else if the point is connected to other selected points via lines, select
                      all points on the shortest such path (and leave any other selections) (DONE)
                    - else select the point (and leave any other point selections) (DONE)
                - else if the point was already selected, do nothing (DONE)
                - else select only that point (DONE)
            - drag on point moves all selected points/lines (DONE)
            - click on empty space:
                - if either Shift or Control key is down, do nothing (DONE)
                - else add a new point, select only the new point, and do not transition
                  into drag mode (DONE)
        - with line tool
            - mouse-over point changes cursor to hand (DONE)
            - mouse-over line changes cursor to hand (DONE)
            - click on point:
                - if Shift or Control is down, do nothing (DONE)
                - else if a point is selected, if there
                  is not already a line from the selected point to the clicked point,
                  add the line; leave only the clicked point selected (DONE)
            - click on line:
                - if Control key down toggle selection of the line (DONE)
                - else if Shift key down
                    - if the line is already selected, do nothing (DONE)
                    - else if the line is connected to other selected lines via lines, select
                      all line segments along the shortest such path (and leave any other selections) (DONE)
                - else if line segment was not yet selected, select only this line segment (DONE)
                - else do nothing (DONE)
            - drag on point or line moves all selected points/lines (DONE)
            - click on empty space:
                - if either Shift or Control key is down, do nothing (DONE)
                - else add a new point and connect the selected point (if any)
                  to the new point; leave only the new point selected (DONE)
        - properties panel
            - update the properties panel when points or lines area added or deleted, or when
              the selection changes:
                - if points are selected show the list of points specified, and their depth
                  in the "Point depth" field if they all have the same depth, else blank in
                  the "Point depth" field (DONE)
                - else show the properties panel for the layer, including the "Default depth" field (DONE)
            - when the user changes the "Point depth" field to a valid number:
                - set the depth of selected point(s) and update their labels (DONE)
    
    undoable operations:
        - add points and/or lines
        - delete points and/or lines
        - drag points and/or lines
        - change depth of points
    non-undoable operations:
        - load layer
        - delete layer (deleting a layer deletes undo operations for that layer from the undo stack)
        - rename layer
        - triangulate
"""

OP_ADD_POINT = 1  # params = None
OP_DELETE_POINT = 2  # params = ( ( x, y ), z, color, state )
OP_ADD_LINE = 3  # params = None
OP_DELETE_LINE = 4  # params = ( point_index_1, point_index_2, color, state )
OP_MOVE_POINT = 5  # params = ( d_lon, d_lat )
OP_CHANGE_POINT_DEPTH = 6  # params = ( old_depth, new_depth )


class Editor():

    """
    A class for handling maproom point/line/polygon edit operations.
    """

    def __init__(self, project):
        self.project = project
        self.lm = project.layer_manager
        self.clickable_object_mouse_is_over = None

    def point_tool_selected(self):
        for layer in self.lm.flatten():
            layer.clear_all_line_segment_selections()
        self.project.refresh()

    def point_tool_deselected(self):
        pass

    def line_tool_selected(self):
        n = 0
        for layer in self.lm.flatten():
            n += layer.get_num_points_selected()
        if (n > 1):
            for layer in self.lm.flatten():
                layer.clear_all_point_selections()
            self.project.refresh()

    def line_tool_deselected(self):
        pass

    def esc_key_pressed(self):
        for layer in self.lm.flatten():
            layer.clear_all_selections()
        self.project.refresh()

    def delete_key_pressed(self):
        if (self.project.control.mode == self.project.control.MODE_EDIT_POINTS or self.project.control.mode == self.project.control.MODE_EDIT_LINES):
            layer = self.project.layer_tree_control.get_selected_layer()
            if (layer != None):
                layer.delete_all_selected_objects()
                self.end_operation_batch()
                self.project.refresh()

    def clicked_on_point(self, event, layer, point_index):
        act_like_point_tool = False

        if (self.project.control.mode == self.project.control.MODE_EDIT_LINES):
            if (event.ControlDown() or event.ShiftDown()):
                act_like_point_tool = True
                pass
            else:
                point_indexes = layer.get_selected_point_indexes()
                if (len(point_indexes == 1) and not layer.are_points_connected(point_index, point_indexes[0])):
                    layer.insert_line_segment(point_index, point_indexes[0])
                    self.end_operation_batch()
                    layer.clear_all_point_selections()
                    layer.select_point(point_index)
                elif len(point_indexes) == 0:  # no currently selected point
                    # select this point
                    layer.select_point(point_index)

        if (self.project.control.mode == self.project.control.MODE_EDIT_POINTS or act_like_point_tool):
            if (event.ControlDown()):
                if (layer.is_point_selected(point_index)):
                    layer.deselect_point(point_index)
                else:
                    layer.select_point(point_index)
            elif (layer.is_point_selected(point_index)):
                pass
            elif (event.ShiftDown()):
                path = layer.find_points_on_shortest_path_from_point_to_selected_point(point_index)
                if (path != []):
                    for p_index in path:
                        layer.select_point(p_index)
                else:
                    layer.select_point(point_index)
            else:
                layer.clear_all_selections()
                layer.select_point(point_index)

        self.project.refresh()

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        if (self.project.control.mode == self.project.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                self.esc_key_pressed()
                layer.insert_point_in_line(world_point, line_segment_index)
                self.end_operation_batch()
                self.project.control.forced_cursor = wx.StockCursor(wx.CURSOR_HAND)

        if (self.project.control.mode == self.project.control.MODE_EDIT_LINES):
            if (event.ControlDown()):
                if (layer.is_line_segment_selected(line_segment_index)):
                    layer.deselect_line_segment(line_segment_index)
                else:
                    layer.select_line_segment(line_segment_index)
            elif (layer.is_line_segment_selected(line_segment_index)):
                pass
            elif (event.ShiftDown()):
                path = layer.find_lines_on_shortest_path_from_line_to_selected_line(line_segment_index)
                if (path != []):
                    for l_s_i in path:
                        layer.select_line_segment(l_s_i)
                else:
                    layer.select_line_segment(line_segment_index)
            else:
                layer.clear_all_selections()
                layer.select_line_segment(line_segment_index)

        self.project.refresh()

    def clicked_on_polygon(self, layer, polygon_index):
        pass

    def clicked_on_empty_space(self, event, layer, world_point):
        if (self.project.control.mode == self.project.control.MODE_EDIT_POINTS or self.project.control.mode == self.project.control.MODE_EDIT_LINES):
            if (layer.type == "root" or layer.type == "folder"):
                wx.MessageDialog(
                    wx.GetApp().GetTopWindow(),
                    caption="Cannot Edit",
                    message="You cannot add points or lines to folder layers.",
                    style=wx.OK | wx.ICON_ERROR
                ).ShowModal()

                return

        if (self.project.control.mode == self.project.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                self.esc_key_pressed()
                # we release the focus because we don't want to immediately drag the new object (if any)
                # self.project.control.release_mouse() # shouldn't be captured now anyway
                layer.insert_point(world_point)
                layer.update_bounds()
                self.end_operation_batch()
                self.project.refresh()

        if (self.project.control.mode == self.project.control.MODE_EDIT_LINES):
            if (not event.ControlDown() and not event.ShiftDown()):
                point_indexes = layer.get_selected_point_indexes()
                if (len(point_indexes == 1)):
                    self.esc_key_pressed()
                    # we release the focus because we don't want to immediately drag the new object (if any)
                    # self.project.control.release_mouse()
                    point_index = layer.insert_point(world_point)
                    layer.update_bounds()
                    layer.insert_line_segment(point_index, point_indexes[0])
                    self.end_operation_batch()
                    layer.clear_all_point_selections()
                    layer.select_point(point_index)
                self.project.refresh()

    def dragged(self, world_d_x, world_d_y):
        if (self.clickable_object_mouse_is_over == None):
            return

        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.lm.get_layer_by_flattened_index(layer_index)
        layer.offset_selected_objects(world_d_x, world_d_y)
        # self.end_operation_batch()
        self.project.refresh()

    def finished_drag(self, mouse_down_position, mouse_move_position):
        if (self.clickable_object_mouse_is_over == None):
            return

        d_x = mouse_move_position[0] - mouse_down_position[0]
        d_y = mouse_down_position[1] - mouse_move_position[1]

        if (d_x == 0 and d_y == 0):
            return

        w_p0 = self.project.control.get_world_point_from_screen_point(mouse_down_position)
        w_p1 = self.project.control.get_world_point_from_screen_point(mouse_move_position)
        world_d_x = w_p1[0] - w_p0[0]
        world_d_y = w_p1[1] - w_p0[1]

        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.lm.get_layer_by_flattened_index(layer_index)

        s_p_i_s = layer.get_selected_point_plus_line_point_indexes()
        for point_index in s_p_i_s:
            params = (world_d_x, world_d_y)
            self.add_undo_operation_to_operation_batch(OP_MOVE_POINT, layer, point_index, params)

        self.end_operation_batch()

    def is_ugrid_point(self, obj):
        (layer_index, type, subtype, object_index) = self.parse_clickable_object(obj)
        #
        return type == Layer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.POINTS_SUB_LAYER_PICKER_OFFSET

    def clickable_object_is_ugrid_point(self):
        return self.is_ugrid_point(self.clickable_object_mouse_is_over)

    def clickable_object_is_ugrid_line(self):
        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        #
        return type == Layer.POINTS_AND_LINES_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.LINES_SUB_LAYER_PICKER_OFFSET

    def clickable_object_is_polyon_fill(self):
        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        #
        return type == Layer_manager.POLYGONS_SUB_LAYER_PICKER_OFFSET and subtype == Polygon_set_renderer.FILL_SUB_LAYER_PICKER_OFFSET

    def clickable_object_is_polyon_point(self):
        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        #
        return type == Layer_manager.POLYGONS_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.POINTS_SUB_LAYER_PICKER_OFFSET

    def clickable_object_is_polyon_point(self):
        (layer_index, type, subtype, object_index) = self.parse_clickable_object(self.clickable_object_mouse_is_over)
        #
        return type == Layer_manager.POLYGONS_SUB_LAYER_PICKER_OFFSET and subtype == Point_and_line_set_renderer.LINES_SUB_LAYER_PICKER_OFFSET

    def parse_clickable_object(self, o):
        if (o == None):
            return (None, None, None, None)

        # see Layer.py for layer types
        # see Point_and_line_set_renderer.py and Polygon_set_renderer.py for subtypes
        (layer_index, object_index) = o
        type_and_subtype = layer_index % 10
        type = (type_and_subtype / 5) * 5
        subtype = type_and_subtype % 5
        layer_index = layer_index / 10
        # print str( self.clickable_object_mouse_is_over ) + "," + str( ( layer_index, type, subtype ) )
        #
        return (layer_index, type, subtype, object_index)

    #


import unittest

# imports needed only for tests
import Layer_manager

import numpy as np


class EditorTests(unittest.TestCase):

    def setUp(self):
        self.layer_manager = Layer_manager.LayerManager()
        self.editor = Editor(self.layer_manager)

    def tearDown(self):
        pass

    def testPoints(self):
        points = [(10, 0), (20, 10), (30, 12)]
        for point in points:
            self.assertTrue(self.editor.is_ugrid_point(point), msg="Point %s failed" % str(point))
            self.editor.clickable_object_mouse_is_over = point
            self.assertTrue(self.editor.clickable_object_is_ugrid_point(), msg="Point %s failed" % str(point))

def getTestSuite():
    return unittest.makeSuite(EditorTests)
