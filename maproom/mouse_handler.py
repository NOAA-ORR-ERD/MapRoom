import os
import time

import wx
import wx.glcanvas as glcanvas
import pyproj

import OpenGL
import OpenGL.GL as gl

import library.coordinates as coordinates
import library.rect as rect
from mouse_commands import *
from vector_object_commands import *


"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)
mouselog = logging.getLogger("mouse")

class MouseHandler(object):
    """
    Processing of mouse events, separate from the rendering window
    
    This is an object-based control system of mouse modes
    """
    icon = "help.png"
    menu_item_name = "Generic Mouse Handler"
    menu_item_tooltip = "Tooltip for generic mouse handler"
    editor_trait_for_enabled = ""
    
    mouse_too_close_pixel_tolerance = 5

    def __init__(self, layer_canvas):
        self.layer_canvas = layer_canvas
        self.layer_canvas.hide_from_picker(None)
        self.snapped_point = None, 0
        self.first_mouse_down_position = 0, 0
        self.after_first_mouse_up = False
        self.mouse_up_too_close = False
        self.can_snap = False
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_ARROW)

    def process_mouse_down(self, event):
        return

    def get_world_point(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        cp = c.get_world_point_from_screen_point(p)
        return cp
    
    def get_position(self, event):
        return self.get_snap_position(event.GetPosition())
    
    def get_snap_position(self, position):
        if self.can_snap:
            c = self.layer_canvas
            o = c.get_object_at_mouse_position(position)
            self.snapped_point = None, 0
            if (o is not None):
                (layer_index, type, subtype, object_index) = c.picker.parse_clickable_object(o)
                layer = c.layer_manager.get_layer_by_pick_index(layer_index)
                before = tuple(position)
                if self.is_snappable_to_layer(layer) and c.picker.is_ugrid_point_type(type):
                    wp = (layer.points.x[object_index], layer.points.y[object_index])
                    position = c.get_screen_point_from_world_point(wp)
                    self.snapped_point = layer, object_index
                    log.debug("snapping to layer_index %s type %s oi %s %s %s" % (layer_index, type, object_index, before, position))
        return position

    def process_mouse_motion_up(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        prefs = c.project.task.get_preferences()
        status_text = coordinates.format_coords_for_display(
            proj_p[0], proj_p[1], prefs.coordinate_display_format)

        c.release_mouse()
        # print "mouse is not down"
        o = c.get_object_at_mouse_position(event.GetPosition())
        if (o is not None):
            (layer_index, type, subtype, object_index) = c.picker.parse_clickable_object(o)
            layer = c.layer_manager.get_layer_by_pick_index(layer_index)
            c.project.clickable_object_in_layer = layer
            if (c.project.layer_tree_control.is_selected_layer(layer)):
                c.project.clickable_object_mouse_is_over = o
            else:
                c.project.clickable_object_mouse_is_over = None
            if c.picker.is_ugrid_point(o):
                status_text += "  Point %s on %s" % (object_index + 1, str(layer))

        else:
            c.project.clickable_object_mouse_is_over = None
            c.project.clickable_object_in_layer = None
        mouselog.debug("object under mouse: %s, on current layer: %s" % (o, c.project.clickable_object_mouse_is_over is not None))

        c.project.task.status_bar.message = status_text

    def process_mouse_motion_down(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        #print "nop: d_x = " + str( d_x ) + ", d_y = " + str( d_x )
    
    def reset_early_mouse_params(self):
        self.mouse_up_too_close = False
        self.after_first_mouse_up = False
    
    def check_early_mouse_release(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        dx = p[0] - self.first_mouse_down_position[0]
        dy = p[1] - self.first_mouse_down_position[1]
        tol = self.mouse_too_close_pixel_tolerance
        if abs(dx) < tol and abs(dy) < tol:
            return True
        return False

    def process_mouse_up(self, event):
        c = self.layer_canvas
        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False

    def process_right_mouse_down(self, event):
        event.Skip()

    def process_right_mouse_up(self, event):
        event.Skip()

    def process_mouse_wheel_scroll(self, event):
        c = self.layer_canvas
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
        # c.project.clickable_object_mouse_is_over = None
        pass

    def process_key_down(self, event):
        pass

    def process_key_up(self, event):
        pass

    def process_key_char(self, event):
        c = self.layer_canvas
        if c.mouse_is_down:
            effective_mode = c.get_effective_tool_mode(event)
            if isinstance(effective_mode, RectSelectMode):
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
        self.layer_canvas.project.clear_all_selections()
    
    def delete_key_pressed(self):
        pass

    def render_overlay(self, renderer):
        """Render additional graphics on canvas
        
        
        """
        self.render_snapped_point(renderer)

    def render_snapped_point(self, renderer):
        """Highlight snapped point when applicable
        
        """
        layer, index = self.snapped_point
        if layer is not None:
            c = self.layer_canvas
            x = layer.points.x[index]
            y = layer.points.y[index]
            s = c.get_numpy_screen_point_from_world_point((x, y))
            x, y = s[0], s[1]
            (x1, y1, x2, y2) = rect.get_normalized_coordinates((x-5, y-5), (x+5, y+5))
            r = ((x1, y1), (x2, y2))
            renderer.draw_screen_box(r, 0.0, 0.0, 0.0, 1.0)

class PanMode(MouseHandler):
    """Mouse mode to pan the viewport
    """
    icon = "pan.png"
    menu_item_name = "Pan Mode"
    menu_item_tooltip = "Scroll the viewport by holding down the mouse"
    editor_trait_for_enabled = ""

    def get_cursor(self):
        c = self.layer_canvas
        if c.mouse_is_down:
            return self.layer_canvas.hand_closed_cursor
        return self.layer_canvas.hand_cursor

    def process_mouse_down(self, event):
        return

    def process_mouse_motion_down(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        #print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
        if (d_x != 0 or d_y != 0):
            # the user has panned the map
            d_x_p = d_x * c.projected_units_per_pixel
            d_y_p = d_y * c.projected_units_per_pixel
            c.projected_point_center = (c.projected_point_center[0] - d_x_p,
                                           c.projected_point_center[1] - d_y_p)
            c.mouse_down_position = p
            c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_canvas
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
    def process_mouse_down(self, event):
        c = self.layer_canvas
        e = c.project
        lm = c.layer_manager

        if (e.clickable_object_mouse_is_over is not None):  # the mouse is on a clickable object
            (layer_index, type, subtype, object_index) = c.picker.parse_clickable_object(e.clickable_object_mouse_is_over)
            layer = lm.get_layer_by_pick_index(layer_index)
            if (e.clickable_object_is_ugrid_point()):
                self.clicked_on_point(event, layer, object_index)
            elif (e.clickable_object_is_ugrid_line()):
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.clicked_on_line_segment(event, layer, object_index, world_point)
            elif (e.clickable_object_is_polygon_fill()):
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.clicked_on_polygon_fill(event, layer, object_index, world_point)
        elif (e.clickable_object_in_layer is not None):
            # clicked on something in different layer.
            self.clicked_on_different_layer(event, e.clickable_object_in_layer)
        else:  # the mouse is not on a clickable object
            # fixme: there should be a reference to the layer manager in the RenderWindow
            # and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = e.layer_tree_control.get_selected_layer()
            if (layer is not None):
                if (event.ControlDown() or event.ShiftDown()):
                    c.selection_box_is_being_defined = True
                    c.CaptureMouse()
                else:
                    world_point = c.get_world_point_from_screen_point(event.GetPosition())
                    self.clicked_on_empty_space(event, layer, world_point)

    def process_mouse_motion_down(self, event):
        c = self.layer_canvas
        p = self.get_position(event)
        proj_p = c.get_world_point_from_screen_point(p)
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        #print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
        if (d_x != 0 or d_y != 0):
            w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
            w_p1 = c.get_world_point_from_screen_point(p)
            if not c.HasCapture():
                c.CaptureMouse()
            c.project.dragged(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1], *self.snapped_point)
            c.mouse_down_position = p
            #print "move: %s" % str(c.mouse_move_position)
            #print "down: %s" % str(c.mouse_down_position)
            c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_canvas
        c.hide_from_picker(None)
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured

        if (c.selection_box_is_being_defined):
            c.mouse_move_position = event.GetPosition()
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
            w_r = c.get_world_rect_from_projected_rect(p_r)
            layer = c.layer_tree_control.get_selected_layer()
            if (layer is not None):
                self.select_objects_in_rect(event, w_r, layer)
            c.selection_box_is_being_defined = False
        else:
            p = self.get_position(event)
            w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
            w_p1 = c.get_world_point_from_screen_point(p)
            #print "move: %s" % str(c.mouse_move_position)
            #print "down: %s" % str(c.mouse_down_position)
            c.project.finished_drag(c.mouse_down_position, c.mouse_move_position, w_p1[0] - w_p0[0], w_p1[1] - w_p0[1], *self.snapped_point)
        c.selection_box_is_being_defined = False
        
        # This render is needed to update the picker buffer because the
        # rendered lines may have only been drawn in the overlay layer.  (Might
        # possibly render twice if the finished_drag renders because the final
        # drag position is different from the last rendered drag position)
        c.render()
    
    def delete_key_pressed(self):
        self.layer_canvas.project.delete_selection()
        
    def clicked_on_point(self, event, layer, point_index):
        pass

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        pass

    def clicked_on_polygon_fill(self, event, layer, polygon_index, world_point):
        pass

    def clicked_on_empty_space(self, event, layer, world_point):
        pass

    def clicked_on_different_layer(self, event, layer):
        c = self.layer_canvas
        e = c.project
        e.layer_tree_control.select_layer(layer)

    def select_objects_in_rect(self, event, rect, layer):
        raise RuntimeError("Abstract method")

class PointSelectionMode(ObjectSelectionMode):
    icon = "add_points.png"
    menu_item_name = "Point Edit Mode"
    menu_item_tooltip = "Edit and add points in the current layer"
    editor_trait_for_enabled = "layer_has_points"

    def get_cursor(self):
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            if e.clickable_object_is_ugrid_line():
                return wx.StockCursor(wx.CURSOR_BULLSEYE)
            else:
                return wx.StockCursor(wx.CURSOR_HAND)
        return wx.StockCursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
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
        c = self.layer_canvas
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']

        if (not event.ControlDown() and not event.ShiftDown()):
            e.clear_all_selections(False)
            cmd = SplitLineCommand(layer, line_segment_index, world_point)
            e.process_command(cmd)
            c.forced_cursor = wx.StockCursor(wx.CURSOR_HAND)
#            if not vis:
#                e.task.status_bar.message = "Split line in hidden layer %s" % layer.name
#            else:
#                layer.select_point(point_index)

    def clicked_on_empty_space(self, event, layer, world_point):
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)) )
        c = self.layer_canvas
        e = c.project
        vis = e.layer_visibility[layer]['layer']
        if layer.is_folder():
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
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            return wx.StockCursor(wx.CURSOR_HAND)
        return wx.StockCursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']
        message = ""

        if (event.ControlDown() or event.ShiftDown()):
            return PointSelectionMode.clicked_on_point(self, event, layer, point_index)

        point_indexes = layer.get_selected_point_indexes()
        if len(point_indexes == 1):
            if not layer.are_points_connected(point_index, point_indexes[0]):
                cmd = ConnectPointsCommand(layer, point_index, point_indexes[0])
                e.process_command(cmd)
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
        c = self.layer_canvas
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
        c = self.layer_canvas
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']
        if layer.is_folder():
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
    dim_background_outside_selection = True
    normalize_mouse_coordinates = True
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_CROSS)
    
    def is_snappable_to_layer(self, layer):
        return False
    
    def process_mouse_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        # unless it is released too soon.
        if not self.after_first_mouse_up:
            self.first_mouse_down_position = event.GetPosition()
        else:
            # reset mouse down position because the on_mouse_down event handler
            # in base_canvas sets it every time the mouse is pressed.  Without
            # this here it would move the start of the rectangle to this most
            # recent mouse press which is not what we want.
            c = self.layer_canvas
            c.mouse_down_position = self.first_mouse_down_position
        self.can_snap = True

    def process_mouse_motion_down(self, event):
        c = self.layer_canvas
        p = self.get_position(event)
        c.mouse_move_position = p
        c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_canvas
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return
        
        if not self.after_first_mouse_up and self.check_early_mouse_release(event):
            self.mouse_up_too_close = True
            self.after_first_mouse_up = True
            return
        self.after_first_mouse_up = True

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.mouse_move_position = self.get_position(event)
        if self.normalize_mouse_coordinates:
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
        else:
            x1, y1 = c.mouse_down_position
            x2, y2 = c.mouse_move_position
        self.process_rect_select(x1, y1, x2, y2)
        self.reset_early_mouse_params()
    
    def process_rect_select(self, x1, y1, x2, y2):
        raise RuntimeError("Abstract method")

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            if self.dim_background_outside_selection:
                rects = c.get_surrounding_screen_rects(((x1, y1), (x2, y2)))
                for r in rects:
                    if (r != rect.EMPTY_RECT):
                        renderer.draw_screen_rect(r, 0.0, 0.0, 0.0, 0.25)
                # small adjustments to make stipple overlap gray rects perfectly
                y1 -= 1
                x2 += 1
            sp = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            renderer.draw_screen_lines(sp, 1.0, 0, 1.0, 1.0, xor=True)
        self.render_snapped_point(renderer)


class ZoomRectMode(RectSelectMode):
    icon = "zoom_box.png"
    menu_item_name = "Zoom Mode"
    menu_item_tooltip = "Zoom in to increase magnification of the current layer"

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
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

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
        e = c.project
        p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
        w_r = c.get_world_rect_from_projected_rect(p_r)
        #print "CROPPING!!!!  ", w_r
        layer = c.project.layer_tree_control.get_selected_layer()
        if (layer is not None and layer.can_crop()):
            cmd = CropRectCommand(layer, w_r)
            e.process_command(cmd)

class ControlPointSelectionMode(ObjectSelectionMode):
    icon = "select.png"
    menu_item_name = "Control Point Edit Mode"
    menu_item_tooltip = "Select objects and move control points in the current layer"
    
    def is_snappable_to_layer(self, layer):
        return hasattr(layer, "center_point_index")

    def get_cursor(self):
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            return wx.StockCursor(wx.CURSOR_HAND)
        return wx.StockCursor(wx.CURSOR_ARROW)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
        e = c.project
        vis = e.layer_visibility[layer]['layer']

        if (event.ControlDown()):
            maintain_aspect = True
        else:
            maintain_aspect = False
        layer.set_anchor_point(point_index, maintain_aspect=maintain_aspect)
        if layer.type == "line_obj":
            self.can_snap = True
            c.hide_from_picker(layer)
            c.render() # force re-rendering to hide the current layer from the picker
        else:
            self.can_snap = False
            c.hide_from_picker(None)

    def clicked_on_polygon_fill(self, event, layer, ignored_index, world_point):
        # Clicking on filled portion of polygon corresponds to clicking on the
        # center point: rigid body translation
        print "center point", layer.center_point_index
        self.clicked_on_point(event, layer, layer.center_point_index)

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        c = self.layer_canvas
        e = c.project
        lm = c.layer_manager
        vis = e.layer_visibility[layer]['layer']

        layer.set_anchor_point(layer.center_point_index, maintain_aspect=True)

    def select_objects_in_rect(self, event, rect, layer):
        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), rect)


class AddVectorObjectByBoundingBoxMode(RectSelectMode):
    dim_background_outside_selection = False
    normalize_mouse_coordinates = False
    vector_object_command = None

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
        e = c.project
        p1 = c.get_projected_point_from_screen_point((x1, y1))
        p2 = c.get_projected_point_from_screen_point((x2, y2))
        cp1 = c.get_world_point_from_projected_point(p1)
        cp2 = c.get_world_point_from_projected_point(p2)
        layer = c.project.layer_tree_control.get_selected_layer()
        if (layer is not None):
            cmd = self.get_vector_object_command(layer, cp1, cp2, layer.manager.default_style)
            e.process_command(cmd)
    
    def get_vector_object_command(self, layer, cp1, cp2, style):
        return self.vector_object_command(layer, cp1, cp2, style)


class AddRectangleMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_square.png"
    menu_item_name = "Add Rectangle"
    menu_item_tooltip = "Add a new rectangles or squares"
    vector_object_command = DrawRectangleCommand


class AddEllipseMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_circle.png"
    menu_item_name = "Add Ellipse"
    menu_item_tooltip = "Add a new ellipses or circles"
    vector_object_command = DrawEllipseCommand


class AddLineMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_line.png"
    menu_item_name = "Add Line"
    menu_item_tooltip = "Add a new lines"
    vector_object_command = DrawLineCommand
    
    def is_snappable_to_layer(self, layer):
        return hasattr(layer, "center_point_index")

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            x1, y1 = c.mouse_down_position
            x2, y2 = c.mouse_move_position
            renderer.draw_screen_line((x1, y1), (x2, y2), 1.0, 0, 1.0, 1.0, xor=True)
        self.render_snapped_point(renderer)
    
    def get_vector_object_command(self, layer, cp1, cp2, style):
        return self.vector_object_command(layer, cp1, cp2, style, *self.snapped_point)


class AddPolylineMode(MouseHandler):
    icon = "shape_polyline.png"
    menu_item_name = "Add Polyline"
    menu_item_tooltip = "Add a new polyline"
    vector_object_command = DrawPolylineCommand

    def __init__(self, *args, **kwargs):
        MouseHandler.__init__(self, *args, **kwargs)
        self.points = []
        self.cursor_point = None
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_CROSS)
    
    def process_mouse_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        c = self.layer_canvas
        if len(self.points) == 0:
            self.reset_early_mouse_params()
            self.first_mouse_down_position = event.GetPosition()
            cp = self.get_world_point(event)
            self.points.append(cp)
            c.render(event)

    def process_mouse_motion_up(self, event):
        c = self.layer_canvas
        self.cursor_point = self.get_world_point(event)
        c.render(event)
    
    def process_mouse_motion_down(self, event):
        self.process_mouse_motion_up(event)

    def process_mouse_up(self, event):
        # After the first point, mouse up events add points
        c = self.layer_canvas
        
        if not self.after_first_mouse_up and self.check_early_mouse_release(event):
            self.mouse_up_too_close = True
            self.after_first_mouse_up = True
            return
        self.after_first_mouse_up = True
        
        cp = self.get_world_point(event)
        self.points.append(cp)
        c.render(event)
    
    def process_right_mouse_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        c = self.layer_canvas
        e = c.project
        if len(self.points) > 1:
            layer = e.layer_tree_control.get_selected_layer()
            if (layer is not None):
                cmd = self.vector_object_command(layer, self.points, layer.manager.default_style)
                e.process_command(cmd)

    def render_overlay(self, renderer):
        c = self.layer_canvas
        sp = [c.get_screen_point_from_world_point(p) for p in self.points]
        renderer.draw_screen_lines(sp, 1.0, 1.0, 0, 1.0, xor=True)
        if self.cursor_point is not None and len(sp) > 0:
            cp = c.get_screen_point_from_world_point(self.cursor_point)
            renderer.draw_screen_line(sp[-1], cp, 1.0, 0, 1.0, 1.0, xor=True)
        self.render_snapped_point(renderer)


class AddPolygonMode(AddPolylineMode):
    icon = "shape_polygon.png"
    menu_item_name = "Add Polygon"
    menu_item_tooltip = "Add a new polygon"
    vector_object_command = DrawPolygonCommand

    def render_overlay(self, renderer):
        c = self.layer_canvas
        sp = [c.get_screen_point_from_world_point(p) for p in self.points]
        renderer.draw_screen_lines(sp, 1.0, 1.0, 0, 1.0, xor=True)
        if self.cursor_point is not None and len(sp) > 0:
            cp = c.get_screen_point_from_world_point(self.cursor_point)
            renderer.draw_screen_line(sp[-1], cp, 1.0, 0, 1.0, 1.0, xor=True)
            if len(sp) > 2:
                renderer.draw_screen_line(sp[0], sp[-1], 1.0, 1.0, 0, 1.0, xor=True, stipple_pattern=0xf0f0)
        self.render_snapped_point(renderer)


class AddOverlayTextMode(MouseHandler):
    icon = "shape_text.png"
    menu_item_name = "Add Text"
    menu_item_tooltip = "Add a new text overlay"

    def __init__(self, *args, **kwargs):
        MouseHandler.__init__(self, *args, **kwargs)
        self.points = []
        self.cursor_point = None
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_CROSS)

    def process_mouse_up(self, event):
        # After the first point, mouse up events add points
        c = self.layer_canvas
        e = c.project
        layer = e.layer_tree_control.get_selected_layer()
        if (layer is not None):
            cp = self.get_world_point(event)
            cmd = AddTextCommand(layer, cp, layer.manager.default_style)
            e.process_command(cmd)


class AddOverlayIconMode(MouseHandler):
    icon = "shape_icon.png"
    menu_item_name = "Add Icon"
    menu_item_tooltip = "Add a new Marplot icon"

    def __init__(self, *args, **kwargs):
        MouseHandler.__init__(self, *args, **kwargs)
        self.points = []
        self.cursor_point = None
    
    def get_cursor(self):
        return wx.StockCursor(wx.CURSOR_CROSS)

    def process_mouse_up(self, event):
        # After the first point, mouse up events add points
        c = self.layer_canvas
        e = c.project
        layer = e.layer_tree_control.get_selected_layer()
        if (layer is not None):
            cp = self.get_world_point(event)
            cmd = AddIconCommand(layer, cp, layer.manager.default_style)
            e.process_command(cmd)
