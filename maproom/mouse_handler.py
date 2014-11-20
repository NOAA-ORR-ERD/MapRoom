import os
import time

import wx
import wx.glcanvas as glcanvas
import pyproj

import OpenGL
import OpenGL.GL as gl

import library.coordinates as coordinates
import renderer
import library.rect as rect
import app_globals
from mouse_commands import *


"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)

class MouseHandler(object):
    """
    Processing of mouse events, separate from the rendering window
    
    This is an object-based control system of mouse modes
    """
    icon = "help.png"
    menu_item_name = "Generic Mouse Handler"
    menu_item_tooltip = "Tooltip for generic mouse handler"
    editor_trait_for_enabled = ""

    def __init__(self, layer_control):
        self.layer_control = layer_control
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_ARROW)

    def process_mouse_down(self, event):
        return

    def process_mouse_motion_up(self, event):
        c = self.layer_control
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        prefs = c.project.task.get_preferences()
        status_text = coordinates.format_coords_for_display(
            proj_p[0], proj_p[1], prefs.coordinate_display_format)

        c.release_mouse()
        # print "mouse is not down"
        o = None
        if c.opengl_renderer is not None:
            o = c.opengl_renderer.picker.get_object_at_mouse_position(event.GetPosition())
        if (o != None):
            (layer_index, type, subtype, object_index) = renderer.parse_clickable_object(o)
            layer = c.layer_manager.get_layer_by_pick_index(layer_index)
            if (c.project.layer_tree_control.is_selected_layer(layer)):
                c.editor.clickable_object_mouse_is_over = o
            else:
                c.editor.clickable_object_mouse_is_over = None
            if renderer.is_ugrid_point(o):
                status_text += "  Point %s on %s" % (object_index + 1, str(layer))

        else:
            c.editor.clickable_object_mouse_is_over = None
        mouselog.debug("object under mouse: %s, on current layer: %s" % (o, c.editor.clickable_object_mouse_is_over is not None))

        c.project.task.status_bar.message = status_text

    def process_mouse_motion_down(self, event):
        c = self.layer_control
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        print "nop: d_x = " + str( d_x ) + ", d_y = " + str( d_x )

    def process_mouse_up(self, event):
        c = self.layer_control
        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False

    def process_mouse_wheel_scroll(self, event):
        c = self.layer_control
        rotation = event.GetWheelRotation()
        delta = event.GetWheelDelta()
        window = event.GetEventObject()
        mouselog.debug("on_mouse_wheel_scroll. delta=%d win=%s" % (delta, window))
        if (delta == 0):
            return

        amount = rotation / delta

        screen_point = event.GetPosition()
        
        # On windows, the mouse wheel events are only passed to controls with
        # the text focus.  So we are forced to grab mouse wheel events on the
        # Frame because the events are propagated up from the control with
        # the focus to the Frame.  The coordinates will be relative to the
        # control, not the map, so we have to transate them here.
        if window != self:
            monitor_point = window.ClientToScreen(screen_point)
            screen_point = c.ScreenToClient(monitor_point)
            mouselog.debug("Mouse over other window at screen pos %s, window pos %s!" % (monitor_point, screen_point))
            size = c.GetSize()
            if screen_point.x < 0 or screen_point.y < 0 or screen_point.x > size.x or screen_point.y > size.y:
                mouselog.debug("Mouse not over RenderWindow: skipping!")
                return
            
        world_point = c.get_world_point_from_screen_point(screen_point)

        prefs = c.project.task.get_preferences()

        zoom = 1.2
        zoom_speed = prefs.zoom_speed
        if zoom_speed == "Slow":
            zoom = 1.2
        elif zoom_speed == "Medium":
            zoom = 1.6
        elif zoom_speed == "Fast":
            zoom = 2.0

        if (amount < 0):
            c.projected_units_per_pixel *= zoom
        else:
            c.projected_units_per_pixel /= zoom
        c.constrain_zoom()

        projected_point = c.get_projected_point_from_screen_point(screen_point)
        new_projected_point = c.get_projected_point_from_world_point(world_point)

        delta = (new_projected_point[0] - projected_point[0], new_projected_point[1] - projected_point[1])

        c.projected_point_center = (c.projected_point_center[0] + delta[0],
                                       c.projected_point_center[1] + delta[1])

        c.render()

    def process_mouse_leave(self, event):
        # this messes up object dragging when the mouse goes outside the window
        # c.editor.clickable_object_mouse_is_over = None
        pass

    def process_key_down(self, event):
        pass

    def process_key_up(self, event):
        pass

    def process_key_char(self, event):
        c = self.layer_control
        if c.mouse_is_down:
            effective_mode = c.get_effective_tool_mode(event)
            if effective_mode == c.MODE_ZOOM_RECT or effective_mode == c.MODE_CROP:
                if (event.GetKeyCode() == wx.WXK_ESCAPE):
                    c.mouse_is_down = False
                    c.ReleaseMouse()
                    c.render()
        else:
            if (event.GetKeyCode() == wx.WXK_ESCAPE):
                self.esc_key_pressed()
            if (event.GetKeyCode() == wx.WXK_BACK):
                self.delete_key_pressed()
    
    def esc_key_pressed(self):
        self.layer_control.editor.clear_all_selections()
    
    def delete_key_pressed(self):
        pass

    def render_overlay(self):
        """Render additional graphics on canvas
        
        
        """
        pass

class PanMode(MouseHandler):
    """Mouse mode to pan the viewport
    """
    icon = "pan.png"
    menu_item_name = "Pan Mode"
    menu_item_tooltip = "Scroll the viewport by holding down the mouse"
    editor_trait_for_enabled = ""

    def get_cursor(self):
        c = self.layer_control
        if c.mouse_is_down:
            return self.layer_control.hand_closed_cursor
        return self.layer_control.hand_cursor

    def process_mouse_down(self, event):
        return

    def process_mouse_motion_down(self, event):
        c = self.layer_control
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
        if (d_x != 0 or d_y != 0):
            # the user has panned the map
            d_x_p = d_x * c.projected_units_per_pixel
            d_y_p = d_y * c.projected_units_per_pixel
            c.projected_point_center = (c.projected_point_center[0] - d_x_p,
                                           c.projected_point_center[1] - d_y_p)
            c.mouse_down_position = p
            c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_control
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False


class ObjectSelectionMode(MouseHandler):
    """Processing of mouse events, separate from the rendering window
    
    This is a precursor to an object-based control system of mouse modes
    """

    def __init__(self, layer_control):
        self.layer_control = layer_control

    def process_mouse_down(self, event):
        c = self.layer_control
        e = c.project
        lm = c.layer_manager

        if (e.clickable_object_mouse_is_over != None):  # the mouse is on a clickable object
            (layer_index, type, subtype, object_index) = renderer.parse_clickable_object(e.clickable_object_mouse_is_over)
            layer = lm.get_layer_by_pick_index(layer_index)
            if (e.layer_tree_control.is_selected_layer(layer)):
                if (e.clickable_object_is_ugrid_point()):
                    self.clicked_on_point(event, layer, object_index)
                if (e.clickable_object_is_ugrid_line()):
                    world_point = c.get_world_point_from_screen_point(event.GetPosition())
                    self.clicked_on_line_segment(event, layer, object_index, world_point)
        else:  # the mouse is not on a clickable object
            # fixme: there should be a reference to the layer manager in the RenderWindow
            # and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = e.layer_tree_control.get_selected_layer()
            if (layer != None):
                if (event.ControlDown() or event.ShiftDown()):
                    c.selection_box_is_being_defined = True
                    c.CaptureMouse()
                else:
                    world_point = c.get_world_point_from_screen_point(event.GetPosition())
                    self.clicked_on_empty_space(event, layer, world_point)

    def process_mouse_motion_down(self, event):
        c = self.layer_control
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        effective_mode = c.get_effective_tool_mode(event)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
        if (d_x != 0 or d_y != 0):
            w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
            w_p1 = c.get_world_point_from_screen_point(p)
            if not c.HasCapture():
                c.CaptureMouse()
            c.editor.dragged(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
            c.mouse_down_position = p
            print "move: %s" % str(c.mouse_move_position)
            print "down: %s" % str(c.mouse_down_position)
            c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_control
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        effective_mode = c.get_effective_tool_mode(event)

        if (c.selection_box_is_being_defined):
            c.mouse_move_position = event.GetPosition()
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
            w_r = c.get_world_rect_from_projected_rect(p_r)
            layer = c.layer_tree_control.get_selected_layer()
            if (layer != None):
                self.select_objects_in_rect(event, w_r, layer)
            c.selection_box_is_being_defined = False
            c.render()
        else:
            p = event.GetPosition()
            w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
            w_p1 = c.get_world_point_from_screen_point(p)
            print "move: %s" % str(c.mouse_move_position)
            print "down: %s" % str(c.mouse_down_position)
            c.editor.finished_drag(c.mouse_down_position, c.mouse_move_position, w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
        c.selection_box_is_being_defined = False
    
    def delete_key_pressed(self):
        self.layer_control.project.delete_selection()
        
    def clicked_on_point(self, event, layer, point_index):
        pass

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        pass

    def clicked_on_polygon(self, layer, polygon_index):
        pass

    def clicked_on_empty_space(self, event, layer, world_point):
        pass

    def select_objects_in_rect(self, event, rect, layer):
        raise RuntimeError("Abstract method")

class PointSelectionMode(ObjectSelectionMode):
    icon = "add_points.png"
    menu_item_name = "Point Edit Mode"
    menu_item_tooltip = "Edit and add points in the current layer"
    editor_trait_for_enabled = "layer_has_points"

    def get_cursor(self):
        e = self.layer_control.editor
        if e.clickable_object_mouse_is_over != None:
            if e.clickable_object_is_ugrid_line():
                return wx.StockCursor(wx.CURSOR_BULLSEYE)
            else:
                return wx.StockCursor(wx.CURSOR_HAND)
        return wx.StockCursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_control
        e = c.project
        vis = e.layer_visibility[layer]['layer']

        if (event.ControlDown()):
            if (layer.is_point_selected(point_index)):
                layer.deselect_point(point_index)
            else:
                layer.select_point(point_index)
        elif (layer.is_point_selected(point_index)):
            layer.clear_all_selections()
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

        e.refresh()

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        c = self.layer_control
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']

        if (not event.ControlDown() and not event.ShiftDown()):
            e.clear_all_selections(False)
            point_index = layer.insert_point_in_line(world_point, line_segment_index)
            lm.end_operation_batch()
            c.forced_cursor = wx.StockCursor(wx.CURSOR_HAND)
            if not vis:
                e.task.status_bar.message = "Split line in hidden layer %s" % layer.name
            else:
                layer.select_point(point_index)

    def clicked_on_empty_space(self, event, layer, world_point):
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)) )
        c = self.layer_control
        e = c.project
        vis = e.layer_visibility[layer]['layer']
        if (layer.type == "root" or layer.type == "folder"):
            e.window.error("You cannot add points to folder layers.", "Cannot Edit")
            return

        if (not event.ControlDown() and not event.ShiftDown()):
            e.clear_all_selections(False)
            
            # FIXME: this comment is from pre maproom3. Is it still applicable?
            # we release the focus because we don't want to immediately drag the new object (if any)
            # self.control.release_mouse() # shouldn't be captured now anyway
            
            cmd = InsertPointCommand(layer, world_point)
            e.process_command(cmd)

    def select_objects_in_rect(self, event, rect, layer):
        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), rect)

class LineSelectionMode(PointSelectionMode):
    icon = "add_lines.png"
    menu_item_name = "Line Edit Mode"
    menu_item_tooltip = "Edit and add lines in the current layer"
    editor_trait_for_enabled = "layer_has_points"

    def get_cursor(self):
        e = self.layer_control.editor
        if e.clickable_object_mouse_is_over != None:
            return wx.StockCursor(wx.CURSOR_HAND)
        return wx.StockCursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_control
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']
        message = ""

        if (event.ControlDown() or event.ShiftDown()):
            return PointSelectionMode.clicked_on_point(self, event, layer, point_index)

        point_indexes = layer.get_selected_point_indexes()
        if len(point_indexes == 1):
            if not layer.are_points_connected(point_index, point_indexes[0]):
                layer.insert_line_segment(point_index, point_indexes[0])
                lm.end_operation_batch()
                if not vis:
                    message = "Added line to hidden layer %s" % layer.name
            layer.clear_all_point_selections()
            if point_indexes[0] != point_index:
                # allow for deselecting points by clicking them again.
                # Only if the old selected point is not the same
                # as the clicked point will the clicked point be
                # highlighted.
                layer.select_point(point_index)
        elif len(point_indexes) == 0:  # no currently selected point
            layer.clear_all_selections()
            # select this point
            layer.select_point(point_index)

        e.refresh()
        if message:
            e.task.status_bar.message = message

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        c = self.layer_control
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']

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

        e.refresh()
        
    def clicked_on_empty_space(self, event, layer, world_point):
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)) )
        c = self.layer_control
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']
        if (layer.type == "root" or layer.type == "folder"):
            e.window.error("You cannot add lines to folder layers.", "Cannot Edit")
            return
        
        if (not event.ControlDown() and not event.ShiftDown()):
            point_indexes = layer.get_selected_point_indexes()
            if (len(point_indexes == 1)):
                e.clear_all_selections(False)
                # we release the focus because we don't want to immediately drag the new object (if any)
                # self.control.release_mouse()
                cmd = InsertLineCommand(layer, point_indexes[0], world_point)
                e.process_command(cmd)

    def select_objects_in_rect(self, event, rect, layer):
        layer.select_line_segments_in_rect(event.ControlDown(), event.ShiftDown(), rect)

class RectSelectMode(MouseHandler):
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_CROSS)
    
    def process_mouse_motion_down(self, event):
        c = self.layer_control
        p = event.GetPosition()
        c.mouse_move_position = event.GetPosition()
        c.render(event)

    def render_overlay(self):
        c = self.layer_control
        if c.mouse_is_down:
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            # self.opengl_renderer.draw_screen_rect( ( ( 20, 50 ), ( 300, 200 ) ), 1.0, 1.0, 0.0, alpha = 0.25 )
            rects = c.get_surrounding_screen_rects(((x1, y1), (x2, y2)))
            for r in rects:
                if (r != rect.EMPTY_RECT):
                    c.opengl_renderer.draw_screen_rect(r, 0.0, 0.0, 0.0, 0.25)
            # small adjustments to make stipple overlap gray rects perfectly
            y1 -= 1
            x2 += 1
            c.opengl_renderer.draw_screen_line((x1, y1), (x2, y1), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
            c.opengl_renderer.draw_screen_line((x1, y1), (x1, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
            c.opengl_renderer.draw_screen_line((x2, y1), (x2, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
            c.opengl_renderer.draw_screen_line((x1, y2), (x2, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)


class ZoomRectMode(RectSelectMode):
    icon = "zoom_box.png"
    menu_item_name = "Zoom Mode"
    menu_item_tooltip = "Zoom in to increase magnification of the current layer"

    def process_mouse_up(self, event):
        c = self.layer_control
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.mouse_move_position = event.GetPosition()
        (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                           c.mouse_move_position)
        d_x = x2 - x1
        d_y = y2 - y1
        if (d_x >= 5 and d_y >= 5):
            p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
            c.projected_point_center = rect.center(p_r)
            s_r = c.get_screen_rect()
            ratio_h = float(d_x) / float(rect.width(s_r))
            ratio_v = float(d_y) / float(rect.height(s_r))
            c.projected_units_per_pixel *= max(ratio_h, ratio_v)
            c.constrain_zoom()
            c.render()

class CropRectMode(RectSelectMode):
    icon = "crop.png"
    menu_item_name = "Crop Mode"
    menu_item_tooltip = "Crop the current layer"

    def process_mouse_up(self, event):
        c = self.layer_control
        e = c.project
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.mouse_move_position = event.GetPosition()
        (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                           c.mouse_move_position)
        p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
        w_r = c.get_world_rect_from_projected_rect(p_r)
        print "CROPPING!!!!  ", w_r
        layer = c.project.layer_tree_control.get_selected_layer()
        if (layer != None and layer.can_crop()):
            cmd = CropRectCommand(layer, w_r)
            e.process_command(cmd)
