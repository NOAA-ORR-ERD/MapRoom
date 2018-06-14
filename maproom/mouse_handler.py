import sys
import math

import wx


from .library.coordinates import haversine, distance_bearing, format_coords_for_display, km_to_rounded_string, mi_to_rounded_string
from . library import rect
from .mouse_commands import ViewportCommand, SetAnchorCommand, CropRectCommand, InsertPointCommand, InsertLineCommand, SplitLineCommand, ConnectPointsCommand
from .vector_object_commands import DrawCircleCommand, DrawEllipseCommand, DrawLineCommand, DrawPolygonCommand, DrawPolylineCommand, DrawRectangleCommand, DrawArrowTextBoxCommand, DrawArrowTextIconCommand, AddTextCommand, AddIconCommand, UnlinkControlPointCommand
from .menu_commands import PolygonEditLayerCommand
from .actions import EditLayerAction

class NoObjectError(RuntimeError):
    pass


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
    toolbar_group = "other"

    mouse_too_close_pixel_tolerance = 8

    def __init__(self, layer_canvas):
        self.layer_canvas = layer_canvas
        self.layer_canvas.hide_from_picker(None)
        self.snapped_point = None, 0
        self.first_mouse_down_position = 0, 0
        self.after_first_mouse_up = False
        self.last_modifier_state = None
        self.mouse_up_too_close = False
        self.can_snap = False
        self.is_over_object = False
        self.current_object_under_mouse = None
        self.last_object_under_mouse = None

        # Optional (only OS X at this point) mouse wheel event filter
        self.wheel_scroll_count = 0
        self.use_every_nth_wheel_scroll = 5

    def get_cursor(self):
        return wx.Cursor(wx.CURSOR_ARROW)

    def get_help_text(self):
        return ""

    def get_long_help_text(self):
        return ""

    def process_mouse_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        c = self.layer_canvas
        self.reset_early_mouse_params()
        self.first_mouse_down_position = event.GetPosition()
        effective_mode = c.get_effective_tool_mode(event)
        log.debug("process_mouse_down: %s" % effective_mode)
        if effective_mode.__class__ == PanMode:
            self.pending_selection = None
            self.is_panning = True
        else:
            self.pending_selection = self.current_object_under_mouse
            self.is_panning = False
            self.process_left_down(event)

    def process_left_down(self, event):
        pass

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
                (layer, object_type, object_index) = o
                before = tuple(position)
                if self.is_snappable_to_layer(layer) and c.picker.is_ugrid_point_type(object_type):
                    wp = (layer.points.x[object_index], layer.points.y[object_index])
                    position = c.get_screen_point_from_world_point(wp)
                    self.snapped_point = layer, object_index
                    log.debug("snapping to layer %s type %s oi %s %s %s" % (layer, object_type, object_index, before, position))
        return position

    def update_status_text(self, proj_p=None, obj=None, zoom=False, instructions=""):
        c = self.layer_canvas
        e = c.project
        prefs = e.task.preferences
        items = []
        if proj_p is not None:
            items.append(format_coords_for_display(proj_p[0], proj_p[1], prefs.coordinate_display_format))
        if zoom:
            items.append("Zoom level=%.2f" % c.zoom_level)
        if instructions:
            items.append(instructions)
        e.status_message = " ".join(items)

        obj_text = ""
        long_text = ""
        if obj is not None:
            (layer, object_type, object_index) = obj
            obj_text, long_text = layer.clickable_object_info(c.picker, object_type, object_index)
            if not long_text:
                long_text = self.get_long_help_text()

        e.debug_message = obj_text
        e.long_status.show_status_text(long_text, multiline=True)

    def process_mouse_motion_up(self, event):
        c = self.layer_canvas
        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)

        c.release_mouse()
        # print "mouse is not down"
        self.current_object_under_mouse = c.get_object_at_mouse_position(event.GetPosition())
        log.debug("process_mouse_motion_up: object under mouse: %s" % (str(self.current_object_under_mouse)))
        obj = None
        if (self.current_object_under_mouse is not None):
            (layer, object_type, object_index) = self.current_object_under_mouse
            c.project.clickable_object_in_layer = layer
            sel = c.project.layer_tree_control.get_edit_layer()
            if (layer == sel):
                obj = c.project.clickable_object_mouse_is_over = self.current_object_under_mouse
            elif sel is not None and sel.show_unselected_layer_info_for(layer):
                obj = self.current_object_under_mouse
            else:
                c.project.clickable_object_mouse_is_over = None
            self.is_over_object = True
        else:
            c.project.clickable_object_mouse_is_over = None
            c.project.clickable_object_in_layer = None
            self.is_over_object = False
        mouselog.debug("object under mouse: %s, on current layer: %s" % (self.current_object_under_mouse, c.project.clickable_object_mouse_is_over is not None))

        self.update_status_text(proj_p, obj, True, self.get_help_text())

    def process_mouse_motion_down(self, event):
        c = self.layer_canvas
        effective_mode = c.get_effective_tool_mode(event)
        log.debug("process_mouse_motion_down: panning=%s mode=%s" % (self.is_panning, effective_mode))
        if self.is_panning:
            e = c.project
            p = event.GetPosition()
            d_x = p[0] - c.mouse_down_position[0]
            d_y = c.mouse_down_position[1] - p[1]
            # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
            if (d_x != 0 or d_y != 0):
                # the user has panned the map
                d_x_p = d_x * c.projected_units_per_pixel
                d_y_p = d_y * c.projected_units_per_pixel
                center = (c.projected_point_center[0] - d_x_p,
                          c.projected_point_center[1] - d_y_p)
                c.mouse_down_position = p

                cmd = ViewportCommand(None, center, c.projected_units_per_pixel)
                e.process_command(cmd)
                event.Skip()
        else:
            if self.pending_selection is not None:
                self.process_mouse_motion_with_selection(event)
            if effective_mode.__class__ == PanMode and not self.is_panning:
                if not self.check_early_mouse_release(event):
                    self.is_panning = True
                return

    def process_mouse_motion_with_selection(self, event):
        event.Skip()

    def dragged_on_empty_space(self, event):
        pass

    def reset_early_mouse_params(self):
        self.mouse_up_too_close = False
        self.after_first_mouse_up = False

    def check_early_mouse_release(self, event):
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
        c = self.layer_canvas
        e = c.project

        if (e.clickable_object_mouse_is_over is not None):  # the mouse is on a clickable object
            (layer, object_type, object_index) = e.clickable_object_mouse_is_over
            print(f"right mouse down: {self} {layer} {object_type} {object_index}")
            if (e.clickable_object_is_ugrid_point()):
                self.right_clicked_on_point(event, layer, object_index)
            elif (e.clickable_object_is_ugrid_line()):
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.right_clicked_on_line_segment(event, layer, object_index, world_point)
            else:  # anything else is the interior of the layer
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.right_clicked_on_interior(event, layer, object_index, world_point)
        elif (e.clickable_object_in_layer is not None):
            # clicked on something in different layer.
            self.right_clicked_on_different_layer(event, e.clickable_object_in_layer)
        else:  # the mouse is not on a clickable object
            # fixme: there should be a reference to the layer manager in the RenderWindow
            # and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = e.layer_tree_control.get_edit_layer()
            world_point = c.get_world_point_from_screen_point(event.GetPosition())
            self.right_clicked_on_empty_space(event, layer, world_point)

    def right_clicked_on_point(self, event, layer, point_index):
        pass

    def right_clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        pass

    def right_clicked_on_interior(self, event, layer, object_index, world_point):
        print(f"Right clicked on {layer}")
        menu = [EditLayerAction]
        c = self.layer_canvas
        e = c.project
        e.popup_context_menu_from_actions(self.layer_canvas, menu, popup_data={'layer':layer, 'object_index':object_index})

    def right_clicked_on_empty_space(self, event, layer, world_point):
        pass

    def right_clicked_on_different_layer(self, event, layer):
        pass

    def process_right_mouse_up(self, event):
        event.Skip()

    def process_middle_mouse_down(self, event):
        c = self.layer_canvas
        self.reset_early_mouse_params()
        self.first_mouse_down_position = event.GetPosition()
        effective_mode = c.get_effective_tool_mode(event)
        if effective_mode.__class__ == PanMode:
            self.pending_selection = None
            self.is_panning = True

    def process_middle_mouse_up(self, event):
        event.Skip()

    def process_mouse_wheel_scroll(self, event):
        c = self.layer_canvas
        e = c.project
        rotation = event.GetWheelRotation()
        delta = event.GetWheelDelta()
        window = event.GetEventObject()
        mouselog.debug("on_mouse_wheel_scroll. rot=%s delta=%d win=%s" % (rotation, delta, window))
        if rotation == 0 or delta == 0:
            return

        if sys.platform == "darwin":
            # OS X mouse wheel handling is not the same as other platform.
            # The delta value is 10 while the other platforms are 120,
            # and the rotation amount varies, seemingly due to the speed at
            # which the wheel is rotated (or how fast the trackpad is swiped)
            # while other platforms are either 120 or -120.  When mouse wheel
            # handling is performed in the usual manner on OS X it produces a
            # strange back-and-forth zooming in/zooming out.  So, this extra
            # hack is needed to operate like the other platforms.

            # add extra to the rotation so the minimum amount is 1 or -1
            extra = delta if rotation > 0 else -delta
            amount = (rotation + extra) // delta
            self.wheel_scroll_count -= abs(amount)
            if self.wheel_scroll_count > 0:
                return
            self.wheel_scroll_count = self.use_every_nth_wheel_scroll
        else:
            amount = rotation // delta

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

        prefs = e.task.preferences

        zoom = 1.25
        zoom_speed = prefs.zoom_speed
        if zoom_speed == "Slow":
            zoom = 1.25
        elif zoom_speed == "Medium":
            zoom = 1.5
        elif zoom_speed == "Fast":
            zoom = 2.0

        units_per_pixel = c.zoom(amount, zoom)
        # Store the units per pixel value so the projection calculations will
        # take them into account to correctly calculate the delta
        c.projected_units_per_pixel = units_per_pixel

        projected_point = c.get_projected_point_from_screen_point(screen_point)
        new_projected_point = c.get_projected_point_from_world_point(world_point)

        delta = (new_projected_point[0] - projected_point[0], new_projected_point[1] - projected_point[1])

        center = (c.projected_point_center[0] + delta[0],
                  c.projected_point_center[1] + delta[1])

        cmd = ViewportCommand(None, center, units_per_pixel)
        e.process_command(cmd)

        p = event.GetPosition()
        proj_p = c.get_world_point_from_screen_point(p)
        self.update_status_text(proj_p, self.current_object_under_mouse, True)

    def process_mouse_leave(self, event):
        # this messes up object dragging when the mouse goes outside the window
        # c.project.clickable_object_mouse_is_over = None
        pass

    def process_key_down(self, event):
        pass

    def process_key_up(self, event):
        pass

    def process_key_char(self, event):
        keycode = event.GetKeyCode()
        text = event.GetUnicodeKey()
        log.debug("process_key_char: char=%s, key=%s, modifiers=%s" % (text, keycode, bin(event.GetModifiers())))
        handled = False
        c = self.layer_canvas
        if c.mouse_is_down:
            effective_mode = c.get_effective_tool_mode(event)
            if isinstance(effective_mode, RectSelectMode):
                if (keycode == wx.WXK_ESCAPE):
                    c.mouse_is_down = False
                    if c.HasCapture():
                        c.ReleaseMouse()
                    c.render()
                    handled = True
        if not handled:
            text = chr(text)
            if keycode == wx.WXK_ESCAPE:
                self.esc_key_pressed()
                handled = True
            elif keycode == wx.WXK_DELETE or keycode == wx.WXK_BACK:
                self.delete_key_pressed()
                handled = True
            elif text in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':  # string.printable minus whitespace
                self.process_text_input(event, text)
                handled = True
        return handled

    def esc_key_pressed(self):
        self.layer_canvas.project.clear_all_selections()

    def delete_key_pressed(self):
        pass

    def process_text_input(self, event, text):
        self.layer_canvas.project.process_info_panel_keystroke(event, text)

    def render_overlay(self, renderer):
        """Render additional graphics on canvas


        """
        self.render_snapped_point(renderer)

    def get_current_object_info(self):
        c = self.layer_canvas
        e = c.project
        if (e.clickable_object_mouse_is_over is None):
            raise NoObjectError

        (layer, object_type, object_index) = e.clickable_object_mouse_is_over
        return layer, object_type, object_index

    def dragged(self, world_d_x, world_d_y, snapped_layer, snapped_cp, about_center=False):
        pass

    def finished_drag(self, mouse_down_position, mouse_move_position, world_d_x, world_d_y, snapped_layer, snapped_cp):
        pass

    def rotated(self, world_d_x, world_d_y):
        pass

    def finished_rotate(self, world_d_x, world_d_y):
        pass

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
            (x1, y1, x2, y2) = rect.get_normalized_coordinates((x - 5, y - 5), (x + 5, y + 5))
            r = ((x1, y1), (x2, y2))
            renderer.draw_screen_box(r, 0.0, 0.0, 0.0, 1.0)

    def distance_between_screen_points(self, sp1, sp2):
        c = self.layer_canvas
        p1 = c.get_world_point_from_screen_point(sp1)
        p2 = c.get_world_point_from_screen_point(sp2)
        km = haversine(p1, p2)
        return km

    def show_distance_between_screen_points(self, text, sp1, sp2, extra=0):
        km = self.distance_between_screen_points(sp1, sp2)
        self.show_distance(text, km + extra)

    def show_cumulative_distance_between_screen_points(self, cum_text, text, sp1, sp2, extra=0):
        km = self.distance_between_screen_points(sp1, sp2)
        cum_km = km + extra
        c = self.layer_canvas
        s = "%s: %s, %s; %s: %s, %s" % (cum_text, km_to_rounded_string(cum_km), mi_to_rounded_string(cum_km * .621371), text, km_to_rounded_string(km), mi_to_rounded_string(km * .621371))
        c.project.status_message = s

    def show_distance(self, text, km):
        c = self.layer_canvas
        s = "%s: %s, %s" % (text, km_to_rounded_string(km), mi_to_rounded_string(km * .621371))
        c.project.status_message = s

    def show_width_height(self, sp, spx, spy):
        c = self.layer_canvas
        p1 = c.get_world_point_from_screen_point(sp)
        p2 = c.get_world_point_from_screen_point(spx)
        wkm = haversine(p1, p2)
        p3 = c.get_world_point_from_screen_point(spy)
        hkm = haversine(p1, p3)
        s = "Width: %s, %s  Height: %s, %s" % (km_to_rounded_string(wkm), mi_to_rounded_string(wkm * .621371), km_to_rounded_string(hkm), mi_to_rounded_string(hkm * .621371))
        c.project.status_message = s


class PanMode(MouseHandler):
    """Mouse mode to pan the viewport
    """
    icon = "pan.png"
    menu_item_name = "Pan Mode"
    menu_item_tooltip = "Scroll the viewport by holding down the mouse"
    editor_trait_for_enabled = ""
    toolbar_group = "view"

    def __init__(self, layer_canvas):
        MouseHandler.__init__(self, layer_canvas)
        self.is_panning = False
        self.pending_selection = None

    def get_cursor(self):
        c = self.layer_canvas
        if c.mouse_is_down:
            return self.layer_canvas.hand_closed_cursor
        return self.layer_canvas.hand_cursor

    def process_mouse_up(self, event):
        c = self.layer_canvas
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False
        if self.pending_selection is not None:
            layer, object_type, object_index = self.pending_selection
            c.project.layer_tree_control.set_edit_layer(layer)
        self.is_panning = False
        self.pending_selection = None


class RNCSelectionMode(PanMode):
    """Mouse mode to pan the viewport
    """
    icon = "select.png"
    menu_item_name = "RNC Chart Selection Mode"
    menu_item_tooltip = "Select an RNC chart to download"
    editor_trait_for_enabled = ""
    toolbar_group = "select"

    def get_rnc_object(self):
        c = self.layer_canvas
        e = c.project
        if e.clickable_object_mouse_is_over is not None:
            (layer, object_type, object_index) = e.clickable_object_mouse_is_over
            if layer.can_highlight_clickable_object(c, object_type, object_index):
                return layer, object_type, object_index
        return None

    def parse_rnc_object(self, rnc):
        layer, object_type, object_index = rnc
        name = layer.ring_identifiers[object_index]['name']
        if ";" in name:
            name, filename, url = name.split(";")
            num, _ = filename.split("_", 1)
        else:
            name = "Invalid RNC"
            filename = ""
            url = None
            num = "0"
        return name, num, filename, url

    def get_long_help_text(self):
        rnc = self.get_rnc_object()
        if rnc is not None:
            layer, object_type, object_index = rnc
            name, num, filename, url = self.parse_rnc_object(rnc)
            return "RNC #%s: %s" % (num, name)
        return ""

    def get_cursor(self):
        return wx.Cursor(wx.CURSOR_ARROW)

    def process_mouse_motion_up(self, event):
        MouseHandler.process_mouse_motion_up(self, event)
        over = self.get_rnc_object()
        self.is_over_object = over is not None
        if over != self.last_object_under_mouse:
            self.layer_canvas.project.refresh()
            self.last_object_under_mouse = over

    # def process_mouse_motion_down(self, event):
    #     if not self.is_panning:
    #         if self.check_early_mouse_release(event):
    #             return
    #         self.is_panning = True
    #     else:
    #         PanMode.process_mouse_motion_down(self, event)

    def process_mouse_up(self, event):
        c = self.layer_canvas
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False
        if not self.is_panning:
            rnc = self.get_rnc_object()
            if rnc is not None:
                layer, object_type, object_index = rnc
                name, num, filename, url = self.parse_rnc_object(rnc)
                if url:
                    p = self.get_world_point(event)
                    if p[0] > 0:
                        regime = 360
                    else:
                        regime = 0
                    # submit download to downloader!
                    log.info("LOADING RNC MAP #%s from %s in %d - %d" % (num, url, regime - 360, regime))
                    e = c.project
                    e.download_rnc(url, filename, num, regime)

        self.is_panning = False

    def render_overlay(self, renderer):
        # draw outline of polygon object that's currently being moused-over
        rnc = self.get_rnc_object()
        if rnc is not None:
            c = self.layer_canvas
            layer, object_type, object_index = rnc
            wp_list = layer.get_highlight_lines(object_type, object_index)
            for wp in wp_list:
                sp = [c.get_screen_point_from_world_point(w) for w in wp]
                renderer.draw_screen_lines(sp, 1.0, 0, 1.0, 1.0, xor=True)


class PolygonSelectionMode(RNCSelectionMode):
    """Mouse mode to select rings
    """
    icon = "select.png"
    menu_item_name = "Polygon Selection Mode"
    menu_item_tooltip = "Select a polygon"
    editor_trait_for_enabled = ""
    toolbar_group = "select"

    def get_help_text(self):
        rnc = self.get_rnc_object()
        if rnc is not None:
            layer, object_type, object_index = rnc
            ident = layer.ring_identifiers[object_index]
            geom = layer.geometry[ident['geom_index']]
            return "   Geom %s: %s" % (str(geom), layer.name)
        return ""

    def get_long_help_text(self):
        return ""

    def process_mouse_up(self, event):
        c = self.layer_canvas
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured
        c.selection_box_is_being_defined = False
        if not self.is_panning:
            rnc = self.get_rnc_object()
            if rnc is not None:
                layer, object_type, object_index = rnc
                cmd = PolygonEditLayerCommand(layer, object_type, object_index)
                c.project.process_command(cmd)
                c.render(event)

        self.is_panning = False


class SelectionMode(MouseHandler):
    """Processing of mouse events, separate from the rendering window

    This is a precursor to an object-based control system of mouse modes
    """
    icon = "select.png"
    toolbar_group = "select"

    def get_cursor(self):
        return wx.Cursor(wx.CURSOR_ARROW)

    def process_left_down(self, event):
        c = self.layer_canvas
        e = c.project
        proj_p = None

        self.last_modifier_state = None
        if (e.clickable_object_mouse_is_over is not None):  # the mouse is on a clickable object
            p = self.get_position(event)
            proj_p = c.get_world_point_from_screen_point(p)

            (layer, object_type, object_index) = e.clickable_object_mouse_is_over
            if (e.clickable_object_is_ugrid_point()):
                self.clicked_on_point(event, layer, object_index)
            elif (e.clickable_object_is_ugrid_line()):
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.clicked_on_line_segment(event, layer, object_index, world_point)
            else:  # anything else is the interior
                world_point = c.get_world_point_from_screen_point(event.GetPosition())
                self.clicked_on_interior(event, layer, object_index, world_point)
        elif (e.clickable_object_in_layer is not None):
            # clicked on something in different layer.
            self.clicked_on_different_layer(event, e.clickable_object_in_layer)
        else:  # the mouse is not on a clickable object
            # fixme: there should be a reference to the layer manager in the RenderWindow
            # and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = e.layer_tree_control.get_edit_layer()
            if (layer is not None):
                if (event.ControlDown() or event.ShiftDown()):
                    c.selection_box_is_being_defined = True
                    c.CaptureMouse()
                else:
                    world_point = c.get_world_point_from_screen_point(event.GetPosition())
                    self.clicked_on_empty_space(event, layer, world_point)

        self.update_status_text(proj_p, None, True, self.get_help_text())

    def process_mouse_motion_with_selection(self, event):
        c = self.layer_canvas
        e = c.project
        if (e.clickable_object_mouse_is_over is not None):  # the mouse is on a clickable object
            layer, object_type, object_index = e.clickable_object_mouse_is_over
            p = self.get_position(event)
            proj_p = c.get_world_point_from_screen_point(p)
            d_x = p[0] - c.mouse_down_position[0]
            d_y = c.mouse_down_position[1] - p[1]
            # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
            if (d_x != 0 or d_y != 0):
                modifiers = event.GetModifiers()
                rotate = modifiers & wx.MOD_CMD
                last = self.last_modifier_state
                if last is None:
                    self.last_modifier_state = modifiers
                elif last != modifiers:
                    if last & wx.MOD_CMD and not modifiers & wx.MOD_CMD:
                        # stopped rotation, pick up with dragging next time
                        c.mouse_down_position = p
                        self.last_modifier_state = modifiers
                        return
                    elif layer.can_rotate:
                        layer.set_initial_rotation()  # reset rotation when control is pressed again
                        c.mouse_down_position = p
                        self.last_modifier_state = modifiers
                        return

                self.last_modifier_state = modifiers
                w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
                w_p1 = c.get_world_point_from_screen_point(p)
                if not c.HasCapture():
                    c.CaptureMouse()
                if rotate and layer.can_rotate:
                    cmd = self.rotated(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
                else:
                    cmd = self.dragged(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1], *self.snapped_point, about_center=modifiers & wx.MOD_SHIFT)
                    c.mouse_down_position = p
                if cmd is not None:
                    c.project.process_command(cmd)
                    c.render(event)

            self.update_status_text(proj_p, None, True, self.get_help_text())
        else:
            self.dragged_on_empty_space(event)

    def process_mouse_up(self, event):
        c = self.layer_canvas
        c.hide_from_picker(None)
        if (not c.mouse_is_down):
            c.selection_box_is_being_defined = False
            return

        e = c.project
        c.mouse_is_down = False
        c.release_mouse()  # it's hard to know for sure when the mouse may be captured

        if (c.selection_box_is_being_defined):
            c.mouse_move_position = event.GetPosition()
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(c.mouse_down_position,
                                                               c.mouse_move_position)
            p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
            w_r = c.get_world_rect_from_projected_rect(p_r)
            layer = c.layer_tree_control.get_edit_layer()
            if (layer is not None):
                self.select_objects_in_rect(event, w_r, layer)
            c.selection_box_is_being_defined = False
        elif (e.clickable_object_mouse_is_over is not None):
            modifiers = event.GetModifiers()
            last = self.last_modifier_state
            if last is not None and last != modifiers:
                modifiers = last
            rotate = modifiers & wx.MOD_CMD
            p = self.get_position(event)
            w_p0 = c.get_world_point_from_screen_point(c.mouse_down_position)
            w_p1 = c.get_world_point_from_screen_point(p)
            # print "move: %s" % str(c.mouse_move_position)
            # print "down: %s" % str(c.mouse_down_position)
            if rotate:
                cmd = self.finished_rotate(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
            else:
                cmd = self.finished_drag(c.mouse_down_position, p, w_p1[0] - w_p0[0], w_p1[1] - w_p0[1], *self.snapped_point)
            if cmd is not None:
                e.process_command(cmd)
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

    def clicked_on_interior(self, event, layer, polygon_index, world_point):
        pass

    def clicked_on_empty_space(self, event, layer, world_point):
        pass

    def clicked_on_different_layer(self, event, layer):
        c = self.layer_canvas
        e = c.project
        e.layer_tree_control.set_edit_layer(layer)

    def select_objects_in_rect(self, event, rect, layer):
        raise RuntimeError("Abstract method")


class ObjectSelectionMode(SelectionMode):
    """Processing of mouse events, separate from the rendering window

    This is a precursor to an object-based control system of mouse modes
    """

    def get_help_text(self):
        text = "Click & drag to move control point, press Shift to resize about object center"
        if sys.platform == "darwin":
            text += ", Cmd for rotate"
        else:
            text += ", Ctrl for rotate"
        return text

    def dragged(self, world_d_x, world_d_y, snapped_layer, snapped_cp, about_center=False):
        (layer, object_type, object_index) = self.get_current_object_info()
        cmd = layer.dragging_selected_objects(world_d_x, world_d_y, snapped_layer, snapped_cp, about_center)
        return cmd

    def finished_drag(self, mouse_down_position, mouse_move_position, world_d_x, world_d_y, snapped_layer, snapped_cp):
        if world_d_x == 0 and world_d_y == 0:
            return
        (layer, object_type, object_index) = self.get_current_object_info()
        cmd = layer.dragging_selected_objects(world_d_x, world_d_y, snapped_layer, snapped_cp)
        return cmd

    def rotated(self, world_d_x, world_d_y):
        (layer, object_type, object_index) = self.get_current_object_info()
        cmd = layer.rotating_selected_objects(world_d_x, world_d_y)
        return cmd

    def finished_rotate(self, world_d_x, world_d_y):
        if world_d_x == 0 and world_d_y == 0:
            return
        (layer, object_type, object_index) = self.get_current_object_info()
        cmd = layer.rotating_selected_objects(world_d_x, world_d_y)
        return cmd


class PointSelectionMode(ObjectSelectionMode):
    """Combo of PanMode and PointEdit mode, but only allowing points/lines
    to be selected and moved, not added to or deleted.
    """
    icon = "select.png"
    menu_item_name = "Point Selection Mode"
    menu_item_tooltip = "Edit and add points in the current layer"
    editor_trait_for_enabled = "layer_has_points"

    def __init__(self, layer_canvas):
        ObjectSelectionMode.__init__(self, layer_canvas)
        self.is_panning = False
        self.pending_selection = None

    def get_cursor(self):
        c = self.layer_canvas
        e = c.project
        if (self.current_object_under_mouse is not None):
            if e.clickable_object_mouse_is_over is not None:
                if e.clickable_object_is_ugrid_line() or e.clickable_object_is_ugrid_point():
                    return wx.Cursor(wx.CURSOR_HAND)
        return wx.Cursor(wx.CURSOR_ARROW)

    def clicked_on_empty_space(self, event, layer, world_point):
        # Mouse down only sets the initial point, after that it is ignored
        self.reset_early_mouse_params()
        self.first_mouse_down_position = event.GetPosition()
        self.pending_selection = self.current_object_under_mouse
        self.is_panning = False

    def dragged_on_empty_space(self, event):
        if self.pending_selection is not None:
            return
        if not self.is_panning:
            if not self.check_early_mouse_release(event):
                self.is_panning = True
            return
        c = self.layer_canvas
        e = c.project
        p = event.GetPosition()
        d_x = p[0] - c.mouse_down_position[0]
        d_y = c.mouse_down_position[1] - p[1]
        # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
        if (d_x != 0 or d_y != 0):
            # the user has panned the map
            d_x_p = d_x * c.projected_units_per_pixel
            d_y_p = d_y * c.projected_units_per_pixel
            center = (c.projected_point_center[0] - d_x_p,
                      c.projected_point_center[1] - d_y_p)
            c.mouse_down_position = p

            cmd = ViewportCommand(None, center, c.projected_units_per_pixel)
            e.process_command(cmd)
            event.Skip()

    def process_mouse_up(self, event):
        ObjectSelectionMode.process_mouse_up(self, event)
        self.is_panning = False

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
        e = c.project

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

    def select_objects_in_rect(self, event, rect, layer):
        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), rect)


class PointEditMode(ObjectSelectionMode):
    icon = "add_points.png"
    menu_item_name = "Point Edit Mode"
    menu_item_tooltip = "Edit and add points in the current layer"
    editor_trait_for_enabled = "layer_has_points"
    toolbar_group = "edit"

    def get_cursor(self):
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            if e.clickable_object_is_ugrid_line():
                return wx.Cursor(wx.CURSOR_BULLSEYE)
            else:
                return wx.Cursor(wx.CURSOR_HAND)
        return wx.Cursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
        e = c.project

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

        if (not event.ControlDown() and not event.ShiftDown()):
            e.clear_all_selections(False)
            cmd = SplitLineCommand(layer, line_segment_index, world_point)
            e.process_command(cmd)
            c.forced_cursor = wx.Cursor(wx.CURSOR_HAND)
#            if not vis:
#                e.status_message = "Split line in hidden layer %s" % layer.name
#            else:
#                layer.select_point(point_index)

    def clicked_on_empty_space(self, event, layer, world_point):
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)))
        c = self.layer_canvas
        e = c.project
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


class LineEditMode(PointEditMode):
    icon = "add_lines.png"
    menu_item_name = "Line Edit Mode"
    menu_item_tooltip = "Edit and add lines in the current layer"
    editor_trait_for_enabled = "layer_has_points"
    toolbar_group = "edit"

    def get_cursor(self):
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            return wx.Cursor(wx.CURSOR_HAND)
        return wx.Cursor(wx.CURSOR_PENCIL)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas
        e = c.project
        vis = e.layer_visibility[layer]['layer']
        message = ""

        if (event.ControlDown() or event.ShiftDown()):
            return PointEditMode.clicked_on_point(self, event, layer, point_index)

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
            e.status_message = message

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        c = self.layer_canvas
        e = c.project

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
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)))
        c = self.layer_canvas
        e = c.project
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
        return wx.Cursor(wx.CURSOR_CROSS)

    def is_snappable_to_layer(self, layer):
        return False

    def process_mouse_down(self, event):
        if self.mouse_up_too_close:
            # process the press-release-move-press method of creating objects
            self.process_left_down(event)
        else:
            # process the default press-drag-press method of creating objects
            MouseHandler.process_mouse_down(self, event)

    def process_left_down(self, event):
        if self.after_first_mouse_up:
            # reset mouse_down_position because the on_mouse_down event handler
            # in base_canvas sets it every time the mouse is pressed.
            c = self.layer_canvas
            c.mouse_down_position = self.first_mouse_down_position
            if self.mouse_up_too_close:
                # User has pressed and released the mouse to set an initial
                # point, and this second mouse down will set the end.
                self.finish_mouse_event(event)
        else:
            # Mouse down sets the initial point, after that it is ignored
            # unless it is released too soon.
            self.first_mouse_down_position = event.GetPosition()
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
        self.finish_mouse_event(event)

    def finish_mouse_event(self, event):
        c = self.layer_canvas
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
            self.show_width_height((x1, y1), (x2, y1), (x1, y2))
        self.render_snapped_point(renderer)


class RulerMode(RectSelectMode):
    icon = "ruler.png"
    menu_item_name = "Measure Distance"
    menu_item_tooltip = "Measure the great-circle distance between two points"
    toolbar_group = "view"

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            x1, y1 = c.mouse_down_position
            x2, y2 = c.mouse_move_position
            renderer.draw_screen_line((x1, y1), (x2, y2), 1.0, 0, 1.0, 1.0, xor=True)
            self.show_distance_between_screen_points("Path length", c.mouse_down_position, c.mouse_move_position)

    def process_rect_select(self, x1, y1, x2, y2):
        # Clear ruler line by redrawing screen.  The ruler won't be drawn in
        # render_overlay because the mouse will not be down.
        self.layer_canvas.render()


class ZoomRectMode(RectSelectMode):
    icon = "zoom_box.png"
    menu_item_name = "Zoom Mode"
    menu_item_tooltip = "Zoom in to increase magnification of the current layer"
    toolbar_group = "view"

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
        e = c.project
        d_x = x2 - x1
        d_y = y2 - y1
        if (d_x >= 5 and d_y >= 5):
            w_r = c.get_world_rect_from_screen_rect(((x1, y1), (x2, y2)))
            center, units_per_pixel = c.calc_zoom_to_world_rect(w_r, False)
            cmd = ViewportCommand(None, center, units_per_pixel)
            e.process_command(cmd)


class CropRectMode(RectSelectMode):
    icon = "crop.png"
    menu_item_name = "Crop Mode"
    menu_item_tooltip = "Crop the current layer"
    toolbar_group = "view"

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
        e = c.project
        p_r = c.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
        w_r = c.get_world_rect_from_projected_rect(p_r)
        # print "CROPPING!!!!  ", w_r
        layer = c.project.layer_tree_control.get_edit_layer()
        if (layer is not None and layer.can_crop()):
            cmd = CropRectCommand(layer, w_r)
            e.process_command(cmd)


class ControlPointEditMode(ObjectSelectionMode):
    icon = "select.png"
    menu_item_name = "Control Point Edit Mode"
    menu_item_tooltip = "Select objects and move control points in the current layer"

    def right_clicked_on_point(self, event, layer, point_index):
        menu = []
        try:
            if layer.can_anchor_point_move():
                menu.append(("Set Anchor to Control Point %d" % point_index, SetAnchorCommand(layer, point_index)))
        except AttributeError:
            pass
        truth, truth_cp = layer.get_control_point_link(point_index)
        if truth is not None:
            menu.append(("Remove Control Point Link to %s" % truth.name, UnlinkControlPointCommand(layer, point_index)))
        if menu:
            self.layer_canvas.project.popup_context_menu_from_commands(self.layer_canvas, menu)

    def is_snappable_to_layer(self, layer):
        return hasattr(layer, "center_point_index")

    def get_cursor(self):
        e = self.layer_canvas.project
        if e.clickable_object_mouse_is_over is not None:
            return wx.Cursor(wx.CURSOR_HAND)
        return wx.Cursor(wx.CURSOR_ARROW)

    def clicked_on_point(self, event, layer, point_index):
        c = self.layer_canvas

        if (event.ControlDown()):
            maintain_aspect = True
        else:
            maintain_aspect = False
        layer.set_anchor_point(point_index, maintain_aspect=maintain_aspect)
        if layer.type == "line_obj":
            self.can_snap = True
            c.hide_from_picker(layer)
            c.render()  # force re-rendering to hide the current layer from the picker
        else:
            self.can_snap = False
            c.hide_from_picker(None)

    def clicked_on_interior(self, event, layer, ignored_index, world_point):
        # Clicking on filled portion of polygon corresponds to clicking on the
        # center point: rigid body translation
        log.debug("center point: %s" % layer.center_point_index)
        self.clicked_on_point(event, layer, layer.center_point_index)

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        layer.set_anchor_point(layer.center_point_index, maintain_aspect=True)

    def select_objects_in_rect(self, event, rect, layer):
        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), rect)


class AddVectorObjectByBoundingBoxMode(RectSelectMode):
    dim_background_outside_selection = False
    normalize_mouse_coordinates = False
    vector_object_command = None
    toolbar_group = "annotation"

    def process_rect_select(self, x1, y1, x2, y2):
        c = self.layer_canvas
        e = c.project
        p1 = c.get_projected_point_from_screen_point((x1, y1))
        p2 = c.get_projected_point_from_screen_point((x2, y2))
        cp1 = c.get_world_point_from_projected_point(p1)
        cp2 = c.get_world_point_from_projected_point(p2)
        layer = c.project.layer_tree_control.get_edit_layer()
        if (layer is not None):
            cmd = self.get_vector_object_command(layer, cp1, cp2)
            e.process_command(cmd, ControlPointEditMode)

    def get_vector_object_command(self, layer, cp1, cp2, style=None):
        return self.vector_object_command(layer, cp1, cp2, style)


class AddRectangleMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_square.png"
    menu_item_name = "Add Rectangle"
    menu_item_tooltip = "Add a new rectangle or square"
    vector_object_command = DrawRectangleCommand


class AddEllipseMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_ellipse.png"
    menu_item_name = "Add Ellipse"
    menu_item_tooltip = "Add a new ellipse or circle"
    vector_object_command = DrawEllipseCommand


class AddArrowTextMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_arrow_text.png"
    menu_item_name = "Add Arrow/Text Box"
    menu_item_tooltip = "Add a new arrow and text box combo object"
    vector_object_command = DrawArrowTextBoxCommand

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            (x1, y1) = c.mouse_down_position
            (x2, y2) = c.mouse_move_position
            (xh, yh) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            sp = [(x1, y1), (xh, yh), (x2, yh), (x2, y2), (xh, y2), (xh, yh)]
            renderer.draw_screen_lines(sp, 1.0, 0, 1.0, 1.0, xor=True)
            self.show_width_height((x1, y1), (x2, y1), (x1, y2))
        self.render_snapped_point(renderer)


class AddArrowTextIconMode(AddArrowTextMode):
    icon = "shape_arrow_text_icon.png"
    menu_item_name = "Add Arrow/Text/Icon Box"
    menu_item_tooltip = "Add a new arrow/text box/icon combo object"
    vector_object_command = DrawArrowTextIconCommand


class AddCircleMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_circle.png"
    menu_item_name = "Add Circle"
    menu_item_tooltip = "Add a new circle from center point"
    vector_object_command = DrawCircleCommand

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            lon1, lat1 = c.get_world_point_from_screen_point(c.mouse_down_position)
            lon2, lat2 = c.get_world_point_from_screen_point(c.mouse_move_position)
            rkm = haversine(lon1, lat1, lon2, lat2)
            # bearing = math.atan2(math.sin(lon2 - lon1) * math.cos(lat2), math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
            _, lat2 = distance_bearing(lon1, lat1, 0.0, rkm)
            lon2, _ = distance_bearing(lon1, lat1, 90.0, rkm)
            rx = lon2 - lon1
            ry = lat2 - lat1
            w = [(lon1 - rx, lat1 - ry), (lon1 + rx, lat1 - ry), (lon1 + rx, lat1 + ry), (lon1 - rx, lat1 + ry), (lon1 - rx, lat1 - ry)]

            sp = [c.get_screen_point_from_world_point(p) for p in w]
            renderer.draw_screen_lines(sp, 1.0, 0, 1.0, 1.0, xor=True)

            (x1, y1) = c.mouse_down_position
            (x2, y2) = c.mouse_move_position
            renderer.draw_screen_line((x1, y1), (x2, y2), 1.0, 0, 1.0, 1.0, xor=True)
            self.show_distance("Radius", rkm)
        self.render_snapped_point(renderer)


class AddLineMode(AddVectorObjectByBoundingBoxMode):
    icon = "shape_line.png"
    menu_item_name = "Add Line"
    menu_item_tooltip = "Add a new line"
    vector_object_command = DrawLineCommand

    def is_snappable_to_layer(self, layer):
        return hasattr(layer, "center_point_index")

    def render_overlay(self, renderer):
        c = self.layer_canvas
        if c.mouse_is_down:
            x1, y1 = c.mouse_down_position
            x2, y2 = c.mouse_move_position
            renderer.draw_screen_line((x1, y1), (x2, y2), 1.0, 0, 1.0, 1.0, xor=True)
            self.show_distance_between_screen_points("Path length", c.mouse_down_position, c.mouse_move_position)
        self.render_snapped_point(renderer)

    def get_vector_object_command(self, layer, cp1, cp2, style=None):
        return self.vector_object_command(layer, cp1, cp2, style, *self.snapped_point)


class AddPolylineMode(MouseHandler):
    icon = "shape_polyline.png"
    menu_item_name = "Add Polyline"
    menu_item_tooltip = "Add a new polyline"
    vector_object_command = DrawPolylineCommand
    toolbar_group = "annotation"

    def __init__(self, *args, **kwargs):
        MouseHandler.__init__(self, *args, **kwargs)
        self.points = []
        self.cursor_point = None
        self.cumulative_distance = 0
        self.mouse_moved_enough = False

    def get_cursor(self):
        return wx.Cursor(wx.CURSOR_CROSS)

    def process_left_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        c = self.layer_canvas
        if len(self.points) == 0:
            self.cumulative_distance = 0
            self.reset_early_mouse_params()
            self.first_mouse_down_position = event.GetPosition()
            cp = self.get_world_point(event)
            self.points.append(cp)
            c.render(event)

    def process_mouse_motion_up(self, event):
        c = self.layer_canvas
        self.cursor_point = self.get_world_point(event)
        c.render(event)
        if not self.mouse_moved_enough:
            self.mouse_moved_enough = not self.check_early_mouse_release(event)

    def process_mouse_motion_down(self, event):
        self.process_mouse_motion_up(event)
        if not self.mouse_moved_enough:
            self.mouse_moved_enough = not self.check_early_mouse_release(event)

    def process_mouse_up(self, event):
        if self.mouse_moved_enough:
            self.finish_mouse_event(event)
            self.mouse_moved_enough = False

    def finish_mouse_event(self, event):
        c = self.layer_canvas
        self.add_point_at_event(event)
        c.render(event)

    def add_point_at_event(self, event):
        cp = self.get_world_point(event)
        self.points.append(cp)
        if len(self.points) > 0:
            c = self.layer_canvas
            sp1 = c.get_screen_point_from_world_point(self.points[-1])
            sp2 = c.get_screen_point_from_world_point(self.points[-2])
            self.cumulative_distance += self.distance_between_screen_points(sp2, sp1)

    def process_right_mouse_down(self, event):
        # Mouse down only sets the initial point, after that it is ignored
        c = self.layer_canvas
        e = c.project
        if len(self.points) > 1:
            layer = e.layer_tree_control.get_edit_layer()
            if (layer is not None):
                cmd = self.vector_object_command(layer, self.points)
                e.process_command(cmd, ControlPointEditMode)

    def render_overlay(self, renderer):
        c = self.layer_canvas
        sp = [c.get_screen_point_from_world_point(p) for p in self.points]
        renderer.draw_screen_lines(sp, 1.0, 1.0, 0, 1.0, xor=True)
        if self.cursor_point is not None and len(sp) > 0:
            cp = c.get_screen_point_from_world_point(self.cursor_point)
            renderer.draw_screen_line(sp[-1], cp, 1.0, 0, 1.0, 1.0, xor=True)
            self.show_cumulative_distance_between_screen_points("Cumulative length", "Current segment length", sp[-1], cp, self.cumulative_distance)
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
            self.show_cumulative_distance_between_screen_points("Cumulative length", "Current segment length", sp[-1], cp, self.cumulative_distance)
            if len(sp) > 2:
                renderer.draw_screen_line(sp[0], sp[-1], 1.0, 1.0, 0, 1.0, xor=True, stipple_pattern=0xf0f0)
        self.render_snapped_point(renderer)


class AddOverlayMode(MouseHandler):
    vector_object_command = None
    toolbar_group = "annotation"

    def get_cursor(self):
        return wx.Cursor(wx.CURSOR_CROSS)

    def process_mouse_up(self, event):
        # After the first point, mouse up events add points
        c = self.layer_canvas
        e = c.project
        layer = e.layer_tree_control.get_edit_layer()
        if (layer is not None):
            cp = self.get_world_point(event)
            cmd = self.get_vector_object_command(layer, cp)
            e.process_command(cmd, ControlPointEditMode)

    def get_vector_object_command(self, layer, cp, style=None):
        return self.vector_object_command(layer, cp, style)


class AddOverlayTextMode(AddOverlayMode):
    icon = "shape_text.png"
    menu_item_name = "Add Text"
    menu_item_tooltip = "Add a new text overlay"
    vector_object_command = AddTextCommand

    def get_vector_object_command(self, layer, cp, style=None):
        return self.vector_object_command(layer, cp, style, 300, 250)


class AddOverlayIconMode(AddOverlayMode):
    icon = "shape_icon.png"
    menu_item_name = "Add Icon"
    menu_item_tooltip = "Add a new Marplot icon"
    vector_object_command = AddIconCommand
