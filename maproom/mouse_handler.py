import os
import time

import wx
import wx.glcanvas as glcanvas
import pyproj

import library.coordinates as coordinates
import renderer
import library.rect as rect
import app_globals

import OpenGL
import OpenGL.GL as gl

from library.Projection import Projection

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)

class MouseHandler(object):
    """Processing of mouse events, separate from the rendering window
    
    This is a precursor to an object-based control system of mouse modes
    """

    def __init__(self, layer_control):
        self.layer_control = layer_control

    def process_mouse_down(self, event):
        c = self.layer_control
        if (c.get_effective_tool_mode(event) == c.MODE_PAN):
            return

        c.select_object(event)

    def process_mouse_motion(self, event):
        c = self.layer_control
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        effective_mode = c.get_effective_tool_mode(event)
        if (not c.mouse_is_down):
            prefs = c.project.task.get_preferences()
            status_text = coordinates.format_coords_for_display(
                proj_p[0], proj_p[1], prefs.coordinate_display_format)

            c.release_mouse()
            # print "mouse is not down"
            o = None
            if c.opengl_renderer is not None:
                o = c.opengl_renderer.picker.get_object_at_mouse_position(event.GetPosition())
            if (o is not None):
                (pick_index, type, subtype, object_index) = renderer.parse_clickable_object(o)
                layer = c.layer_manager.get_layer_by_pick_index(pick_index)
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

        if (c.mouse_is_down):
            d_x = p[0] - c.mouse_down_position[0]
            d_y = c.mouse_down_position[1] - p[1]
            # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
            if (effective_mode == c.MODE_PAN):
                if (d_x != 0 or d_y != 0):
                    # the user has panned the map
                    d_x_p = d_x * c.projected_units_per_pixel
                    d_y_p = d_y * c.projected_units_per_pixel
                    c.projected_point_center = (c.projected_point_center[0] - d_x_p,
                                                   c.projected_point_center[1] - d_y_p)
                    c.mouse_down_position = p
                    c.render(event)
            elif (effective_mode == c.MODE_ZOOM_RECT or effective_mode == c.MODE_CROP or c.selection_box_is_being_defined):
                c.mouse_move_position = event.GetPosition()
                c.render(event)
            else:
                if (d_x != 0 or d_y != 0):
                    w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
                    w_p1 = c.get_world_point_from_screen_point(p)
                    if not c.HasCapture():
                        c.CaptureMouse()
                    c.editor.dragged(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
                    c.mouse_down_position = p
                    c.render(event)

    def process_mouse_up(self, event):
        c = self.layer_control
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        effective_mode = c.get_effective_tool_mode(event)

        if (effective_mode == c.MODE_ZOOM_RECT):
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
        elif (effective_mode == c.MODE_CROP):
            c.mouse_move_position = event.GetPosition()
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
            w_r = c.get_world_rect_from_projected_rect(p_r)
            print "CROPPING!!!!  ", w_r
            layer = c.project.layer_tree_control.get_selected_layer()
            if (layer != None and layer.can_crop()):
                layer.crop_rectangle(w_r)
                layer.manager.end_operation_batch()
                c.project.layer_manager.renderer_rebuild_event = True
            c.render()
        elif (effective_mode == c.MODE_EDIT_POINTS or effective_mode == c.MODE_EDIT_LINES):
            if (c.selection_box_is_being_defined):
                c.mouse_move_position = event.GetPosition()
                (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                                   c.mouse_move_position)
                p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
                w_r = c.get_world_rect_from_projected_rect(p_r)
                layer = c.layer_tree_control.get_selected_layer()
                if (layer != None):
                    if (effective_mode == c.MODE_EDIT_POINTS):
                        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), w_r)
                    else:
                        layer.select_line_segments_in_rect(event.ControlDown(), event.ShiftDown(), w_r)
                c.selection_box_is_being_defined = False
                c.render()
            else:
                c.editor.finished_drag(c.mouse_down_position, c.mouse_move_position)
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
                c.editor.esc_key_pressed()
            if (event.GetKeyCode() == wx.WXK_BACK):
                c.editor.delete_key_pressed()
