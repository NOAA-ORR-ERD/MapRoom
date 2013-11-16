import os
import time

import wx
import wx.glcanvas as glcanvas
import pyproj

import lon_lat_grid
import library.coordinates as coordinates
import library.Opengl_renderer
import library.Opengl_renderer.LayerRendererOpenGL as LayerRendererOpenGL
import library.rect as rect
import app_globals

import OpenGL
import OpenGL.GL as gl

from wx.lib.pubsub import pub

from library.Projection import Projection

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""


def update_status(message):
    tlw = wx.GetApp().GetTopWindow()
    tlw.SetStatusText(message, 1)
    # for debugging purposes, so we can see a history of rendering
    try:
        f = open("output_render.txt", "a")
        f.write(message + '\n')
    except:
        pass


class RenderWindow(glcanvas.GLCanvas):

    """
    The core rendering class for MapRoom app.
    """

    # fixme: this should be in app_globals, or something like that
    IMAGE_PATH = "ui/images"

    MODE_PAN = 0
    MODE_ZOOM_RECT = 1
    MODE_EDIT_POINTS = 2
    MODE_EDIT_LINES = 3

    mode = MODE_PAN
    hand_cursor = None
    hand_closed_cursor = None
    forced_cursor = None

    opengl_renderer = None

    mouse_is_down = False
    is_alt_key_down = False
    selection_box_is_being_defined = False
    mouse_down_position = (0, 0)
    mouse_move_position = (0, 0)

    #is_initialized = False
    #is_closing = False

    def __init__(self, *args, **kwargs):
        self.layer_manager = kwargs.pop('layer_manager')
        self.editor = kwargs.pop('editor')
        self.layer_renderers = {}

        kwargs['attribList'] = (glcanvas.WX_GL_RGBA,
                                glcanvas.WX_GL_DOUBLEBUFFER,
                                glcanvas.WX_GL_MIN_ALPHA, 8, )
        glcanvas.GLCanvas.__init__(self, *args, **kwargs)

        self.context = glcanvas.GLContext(self)

        p = os.path.join(self.IMAGE_PATH, "cursors", "hand.ico")
        self.hand_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        p = os.path.join(self.IMAGE_PATH, "cursors", "hand_closed.ico")
        self.hand_closed_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)

        self.lon_lat_grid = lon_lat_grid.Lon_lat_grid()
        self.lon_lat_grid_shown = True

        # two variables keep track of what's visible on the screen:
        # (1) the projected point at the center of the screen
        self.projected_point_center = (0, 0)
        # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)
        self.projected_units_per_pixel = 10000
        self.projection = Projection("+proj=merc +units=m +over")
        # for longlat projection, apparently someone decided that since the projection
        # is the identity, it might as well do something and so it returns the coordinates as
        # radians instead of degrees; so here we use this variable to avoid using the longlat projection
        self.projection_is_identity = False

        #self.frame.Bind( wx.EVT_MOVE, self.refresh )
        #self.frame.Bind( wx.EVT_IDLE, self.on_idle )
        # self.frame.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )

        self.Bind(wx.EVT_IDLE, self.on_idle)  # not sure about this -- but it's where the cursors are set.
        self.Bind(wx.EVT_PAINT, self.render)
        self.Bind(wx.EVT_SIZE, self.resize_render_pane)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_CHAR, self.on_key_char)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_up)
        # Prevent flashing on Windows by doing nothing on an erase background event.
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
        #self.is_initialized = True

    def update_renderers(self):
        for layer in self.layer_manager.layers:
            if not layer in self.layer_renderers:
                self.layer_renderers[layer] = LayerRendererOpenGL.LayerRendererOpenGL(self, layer)
                self.layer_renderers[layer].create_necessary_renderers()

    def on_mouse_down(self, event):
        # self.SetFocus() # why would it not be focused?
        print "in on_mouse_down"
        self.get_effective_tool_mode(event)  # update alt key state
        self.forced_cursor = None
        self.mouse_is_down = True
        self.selection_box_is_being_defined = False
        self.mouse_down_position = event.GetPosition()
        self.mouse_move_position = self.mouse_down_position

        if (self.get_effective_tool_mode(event) == self.MODE_PAN):
            return

        e = self.editor
        lm = self.layer_manager

        if (e.clickable_object_mouse_is_over != None):  # the mouse is on a clickable object
            (layer_index, type, subtype, object_index) = e.parse_clickable_object(e.clickable_object_mouse_is_over)
            layer = lm.get_layer_by_flattened_index(layer_index)
            if (lm.is_layer_selected(layer)):
                if (e.clickable_object_is_ugrid_point()):
                    e.clicked_on_point(event, layer, object_index)
                if (e.clickable_object_is_ugrid_line()):
                    world_point = self.get_world_point_from_screen_point(event.GetPosition())
                    e.clicked_on_line_segment(event, layer, object_index, world_point)
        else:  # the mouse is not on a clickable object
            # fixme: there should be a reference to the layer manager in the RenderWindow
            # and we could get the selected layer from there -- or is selected purely a UI concept?
            layer = app_globals.application.layer_tree_control.get_selected_layer()
            if (layer != None):
                if (event.ControlDown() or event.ShiftDown()):
                    self.selection_box_is_being_defined = True
                    self.CaptureMouse()
                else:
                    world_point = self.get_world_point_from_screen_point(event.GetPosition())
                    e.clicked_on_empty_space(event, layer, world_point)

    def release_mouse(self):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        while self.HasCapture():
            self.ReleaseMouse()

    def on_mouse_motion(self, event):
        self.get_effective_tool_mode(event)  # update alt key state

        p = event.GetPosition()
        proj_p = self.get_world_point_from_screen_point(p)
        if (not self.mouse_is_down):
            status_text = coordinates.format_coords_for_display(proj_p[0], proj_p[1])

            self.release_mouse()
            # print "mouse is not down"
            o = None
            if self.opengl_renderer is not None:
                o = self.opengl_renderer.picker.get_object_at_mouse_position(event.GetPosition())
            # print "object that is under mouse:", o
            if (o != None):
                (layer_index, type, subtype, object_index) = self.editor.parse_clickable_object(o)
                layer = self.layer_manager.get_layer_by_flattened_index(layer_index)
                if (self.layer_manager.is_layer_selected(layer)):
                    self.editor.clickable_object_mouse_is_over = o
                if self.editor.is_ugrid_point(o):
                    status_text += "  Point %s on %s" % (object_index + 1, str(layer))

            else:
                self.editor.clickable_object_mouse_is_over = None

            tlw = wx.GetApp().GetTopWindow()
            tlw.SetStatusText(status_text)

        if (self.mouse_is_down):
            d_x = p[0] - self.mouse_down_position[0]
            d_y = self.mouse_down_position[1] - p[1]
            # print "d_x = " + str( d_x ) + ", d_y = " + str( d_x )
            if (self.get_effective_tool_mode(event) == self.MODE_PAN):
                if (d_x != 0 or d_y != 0):
                    # the user has panned the map
                    d_x_p = d_x * self.projected_units_per_pixel
                    d_y_p = d_y * self.projected_units_per_pixel
                    self.projected_point_center = (self.projected_point_center[0] - d_x_p,
                                                   self.projected_point_center[1] - d_y_p)
                    self.mouse_down_position = p
                    self.render(event)
            elif (self.get_effective_tool_mode(event) == self.MODE_ZOOM_RECT or self.selection_box_is_being_defined):
                self.mouse_move_position = event.GetPosition()
                self.render(event)
            else:
                if (d_x != 0 or d_y != 0):
                    w_p0 = self.get_world_point_from_screen_point(self.mouse_down_position)
                    w_p1 = self.get_world_point_from_screen_point(p)
                    if not self.HasCapture():
                        self.CaptureMouse()
                    self.editor.dragged(w_p1[0] - w_p0[0], w_p1[1] - w_p0[1])
                    self.mouse_down_position = p
                    self.render(event)

    def on_mouse_up(self, event):
        self.get_effective_tool_mode(event)  # update alt key state

        self.forced_cursor = None

        if (not self.mouse_is_down):
            self.selection_box_is_being_defined = False

            return

        self.mouse_is_down = False
        self.release_mouse()  # it's hard to know for sure when the mouse may be captured

        if (self.get_effective_tool_mode(event) == self.MODE_ZOOM_RECT):
            self.mouse_move_position = event.GetPosition()
            (x1, y1, x2, y2) = rect.get_normalized_coordinates(self.mouse_down_position,
                                                               self.mouse_move_position)
            d_x = x2 - x1
            d_y = y2 - y1
            if (d_x >= 5 and d_y >= 5):
                p_r = self.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
                self.projected_point_center = rect.center(p_r)
                s_r = self.get_screen_rect()
                ratio_h = float(d_x) / float(rect.width(s_r))
                ratio_v = float(d_y) / float(rect.height(s_r))
                self.projected_units_per_pixel *= max(ratio_h, ratio_v)
                self.constrain_zoom()
                self.render()
        elif (self.get_effective_tool_mode(event) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode(event) == self.MODE_EDIT_LINES):
            if (self.selection_box_is_being_defined):
                self.mouse_move_position = event.GetPosition()
                (x1, y1, x2, y2) = rect.get_normalized_coordinates(self.mouse_down_position,
                                                                   self.mouse_move_position)
                p_r = self.get_projected_rect_from_screen_rect(((x1, y1), (x2, y2)))
                w_r = self.get_world_rect_from_projected_rect(p_r)
                layer = self.layer_tree_control.get_selected_layer()
                if (layer != None):
                    if (self.get_effective_tool_mode(event) == self.MODE_EDIT_POINTS):
                        layer.select_points_in_rect(event.ControlDown(), event.ShiftDown(), w_r)
                    else:
                        layer.select_line_segments_in_rect(event.ControlDown(), event.ShiftDown(), w_r)
                self.selection_box_is_being_defined = False
                self.render()
            else:
                self.editor.finished_drag(self.mouse_down_position, self.mouse_move_position)
        self.selection_box_is_being_defined = False

    def on_mouse_wheel_scroll(self, event):
        self.get_effective_tool_mode(event)  # update alt key state

        rotation = event.GetWheelRotation()
        delta = event.GetWheelDelta()
        if (delta == 0):
            return

        amount = rotation / delta

        screen_point = event.GetPosition()
        world_point = self.get_world_point_from_screen_point(screen_point)

        prefs = app_globals.preferences

        zoom = 1.2
        zoom_speed = prefs["Scroll Zoom Speed"]
        if zoom_speed == "Slow":
            zoom = 1.2
        elif zoom_speed == "Medium":
            zoom = 1.6
        elif zoom_speed == "Fast":
            zoom = 2.0

        if (amount < 0):
            self.projected_units_per_pixel *= zoom

        else:
            self.projected_units_per_pixel /= zoom
        self.constrain_zoom()

        projected_point = self.get_projected_point_from_screen_point(screen_point)
        new_projected_point = self.get_projected_point_from_world_point(world_point)

        delta = (new_projected_point[0] - projected_point[0], new_projected_point[1] - projected_point[1])

        self.projected_point_center = (self.projected_point_center[0] + delta[0],
                                       self.projected_point_center[1] + delta[1])

        self.render()

    def on_mouse_leave(self, event):
        self.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        # this messes up object dragging when the mouse goes outside the window
        # self.editor.clickable_object_mouse_is_over = None

    def on_key_down(self, event):
        self.get_effective_tool_mode(event)
        event.Skip()

    def on_key_up(self, event):
        self.get_effective_tool_mode(event)
        event.Skip()

    def on_key_char(self, event):
        self.get_effective_tool_mode(event)
        self.set_cursor()

        if (self.mouse_is_down and self.get_effective_tool_mode(event) == self.MODE_ZOOM_RECT):
            if (event.GetKeyCode() == wx.WXK_ESCAPE):
                self.mouse_is_down = False
                self.ReleaseMouse()
                self.render()
        else:
            if (event.GetKeyCode() == wx.WXK_ESCAPE):
                self.editor.esc_key_pressed()
            if (event.GetKeyCode() == wx.WXK_BACK):
                self.editor.delete_key_pressed()

    def on_idle(self, event):
        # self.get_effective_tool_mode( event ) # update alt key state (not needed, it gets called in set_cursor anyway
        # print self.mouse_is_down
        self.set_cursor()

    def set_cursor(self):
        if (self.forced_cursor != None):
            self.SetCursor(self.forced_cursor)
            #
            return

        if (self.editor.clickable_object_mouse_is_over != None and
                (self.get_effective_tool_mode(None) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode(None) == self.MODE_EDIT_LINES)):
            if (self.get_effective_tool_mode(None) == self.MODE_EDIT_POINTS and self.editor.clickable_object_is_ugrid_line()):
                self.SetCursor(wx.StockCursor(wx.CURSOR_BULLSEYE))
            else:
                self.SetCursor(wx.StockCursor(wx.CURSOR_HAND))
            #
            return

        if (self.mouse_is_down):
            if (self.get_effective_tool_mode(None) == self.MODE_PAN):
                self.SetCursor(self.hand_closed_cursor)
            #
            return

        # w = wx.FindWindowAtPointer() is this needed?
        # if ( w == self.renderer ):
        c = wx.StockCursor(wx.CURSOR_ARROW)
        if (self.get_effective_tool_mode(None) == self.MODE_PAN):
            c = self.hand_cursor
        if (self.get_effective_tool_mode(None) == self.MODE_ZOOM_RECT):
            c = wx.StockCursor(wx.CURSOR_CROSS)
        if (self.get_effective_tool_mode(None) == self.MODE_EDIT_POINTS or self.get_effective_tool_mode(None) == self.MODE_EDIT_LINES):
            c = wx.StockCursor(wx.CURSOR_PENCIL)
        self.SetCursor(c)

    def get_effective_tool_mode(self, event):
        if (event != None):
            try:
                self.is_alt_key_down = event.AltDown()
                # print self.is_alt_key_down
            except:
                pass
        if (self.is_alt_key_down):
            return self.MODE_PAN
        return self.mode

    def render(self, event=None):
        """
        import traceback
        traceback.print_stack();
        import code; code.interact( local = locals() )
        """

        t0 = time.clock()
        self.SetCurrent(self.context)
        self.update_renderers()

        # this has to be here because the window has to exist before making the renderer
        if (self.opengl_renderer == None):
            self.opengl_renderer = library.Opengl_renderer.Opengl_renderer(True)

        s_r = self.get_screen_rect()
        # print "s_r = " + str( s_r )
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        # print "p_r = " + str( p_r )
        w_r = self.get_world_rect_from_projected_rect(p_r)
        # print "w_r = " + str( w_r )

        if (not self.opengl_renderer.prepare_to_render_projected_objects(p_r, s_r)):
            return

        """
        self.root_renderer.render()
        self.set_screen_projection_matrix()
        self.box_overlay.render()
        self.set_render_projection_matrix()
        """

        def render_layers(pick_mode=False):
            list = self.layer_manager.flatten()
            length = len(list)
            for i, layer in enumerate(reversed(list)):
                self.layer_renderers[layer].render(self, (length - 1 - i) * 10, pick_mode)

        render_layers()

        if (not self.opengl_renderer.prepare_to_render_screen_objects(s_r)):
            return

        # we use a try here since we must call done_rendering_screen_objects() below
        # to pop the gl stack
        try:
            if (self.lon_lat_grid_shown):
                self.lon_lat_grid.draw(self, w_r, p_r, s_r)
            if ((self.get_effective_tool_mode(event) == self.MODE_ZOOM_RECT or self.selection_box_is_being_defined) and self.mouse_is_down):
                (x1, y1, x2, y2) = rect.get_normalized_coordinates(self.mouse_down_position,
                                                                   self.mouse_move_position)
                # self.opengl_renderer.draw_screen_rect( ( ( 20, 50 ), ( 300, 200 ) ), 1.0, 1.0, 0.0, alpha = 0.25 )
                rects = self.get_surrounding_screen_rects(((x1, y1), (x2, y2)))
                for r in rects:
                    if (r != rect.EMPTY_RECT):
                        self.opengl_renderer.draw_screen_rect(r, 0.0, 0.0, 0.0, 0.25)
                # small adjustments to make stipple overlap gray rects perfectly
                y1 -= 1
                x2 += 1
                self.opengl_renderer.draw_screen_line((x1, y1), (x2, y1), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
                self.opengl_renderer.draw_screen_line((x1, y1), (x1, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
                self.opengl_renderer.draw_screen_line((x2, y1), (x2, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
                self.opengl_renderer.draw_screen_line((x1, y2), (x2, y2), 1.0, 0, 0, 0, 1.0, 1, 0x00FF)
        except Exception as inst:
            raise
            # print "error during rendering of screen objects: " + str( inst )

        self.opengl_renderer.done_rendering_screen_objects()

        self.SwapBuffers()

        self.opengl_renderer.prepare_to_render_picker(s_r)
        render_layers(pick_mode=True)
        self.opengl_renderer.done_rendering_picker()

        elapsed = time.clock() - t0
        wx.CallAfter(update_status, "Render complete, took %f seconds." % elapsed)

        if (event != None):
            event.Skip()

    def rebuild_points_and_lines_for_layer(self, layer):
        if layer in self.layer_renderers:
            self.layer_renderers[layer].rebuild_point_and_line_set_renderer()

    def rebuild_triangles_for_layer(self, layer):
        if layer in self.layer_renderers:
            self.layer_renderers[layer].rebuild_triangle_set_renderer()

    def resize_render_pane(self, event):
        if not self.GetContext():
            return

        event.Skip()
        self.render(event)

    # functions related to world coordinates, projected coordinates, and screen coordinates

    def get_screen_size(self):
        return self.GetClientSize()

    def get_screen_rect(self):
        size = self.get_screen_size()
        #
        return ((0, 0), (size[0], size[1]))

    def get_projected_point_from_screen_point(self, screen_point):
        c = rect.center(self.get_screen_rect())
        d = (screen_point[0] - c[0], screen_point[1] - c[1])
        d_p = (d[0] * self.projected_units_per_pixel, d[1] * self.projected_units_per_pixel)
        #
        return (self.projected_point_center[0] + d_p[0],
                self.projected_point_center[1] - d_p[1])

    def get_projected_rect_from_screen_rect(self, screen_rect):
        left_bottom = (screen_rect[0][0], screen_rect[1][1])
        right_top = (screen_rect[1][0], screen_rect[0][1])
        #
        return (self.get_projected_point_from_screen_point(left_bottom),
                self.get_projected_point_from_screen_point(right_top))

    def get_screen_point_from_projected_point(self, projected_point):
        d_p = (projected_point[0] - self.projected_point_center[0],
               projected_point[1] - self.projected_point_center[1])
        d = (d_p[0] / self.projected_units_per_pixel, d_p[1] / self.projected_units_per_pixel)
        r = self.get_screen_rect()
        c = rect.center(r)
        #
        return (c[0] + d[0], c[1] - d[1])

    def get_screen_rect_from_projected_rect(self, projected_rect):
        left_top = (projected_rect[0][0], projected_rect[1][1])
        right_bottom = (projected_rect[1][0], projected_rect[0][1])
        #
        return (self.get_screen_point_from_projected_point(left_top),
                self.get_screen_point_from_projected_point(right_bottom))

    def get_world_point_from_projected_point(self, projected_point):
        return self.projection(projected_point[0], projected_point[1], inverse=True)

    def get_world_rect_from_projected_rect(self, projected_rect):
        return (self.get_world_point_from_projected_point(projected_rect[0]),
                self.get_world_point_from_projected_point(projected_rect[1]))

    def get_projected_point_from_world_point(self, world_point):
        return self.projection(world_point[0], world_point[1])

    def get_projected_rect_from_world_rect(self, world_rect):
        return (self.get_projected_point_from_world_point(world_rect[0]),
                self.get_projected_point_from_world_point(world_rect[1]))

    def get_world_point_from_screen_point(self, screen_point):
        return self.get_world_point_from_projected_point(self.get_projected_point_from_screen_point(screen_point))

    def get_world_rect_from_screen_rect(self, screen_rect):
        return self.get_world_rect_from_projected_rect(self.get_projected_rect_from_screen_rect(screen_rect))

    def get_screen_point_from_world_point(self, world_point):
        screen_point = self.get_screen_point_from_projected_point(self.get_projected_point_from_world_point(world_point))
        # screen points are pixels, which should be int values
        return (round(screen_point[0]), round(screen_point[1]))

    def get_screen_rect_from_world_rect(self, world_rect):
        rect = self.get_screen_rect_from_projected_rect(self.get_projected_rect_from_world_rect(world_rect))
        return ((int(round(rect[0][0])), int(round(rect[0][1]))), (int(round(rect[1][0])), int(round(rect[1][1]))))

    def zoom(self, steps=1, ratio=2.0, focus_point_screen=None):
        if ratio > 0:
            self.projected_units_per_pixel /= ratio
        else:
            self.projected_units_per_pixel *= abs(ratio)
        self.constrain_zoom()
        self.render()

    def zoom_in(self):
        self.zoom(ratio=2.0)

    def zoom_out(self):
        self.zoom(ratio=-2.0)

    def zoom_to_fit(self):
        w_r = self.layer_manager.accumulate_layer_rects()
        if (w_r != rect.NONE_RECT):
            self.zoom_to_world_rect(w_r)

    def zoom_to_world_rect(self, w_r):
        p_r = self.get_projected_rect_from_world_rect(w_r)
        size = self.get_screen_size()
        # so that when we zoom, the points don't hit the very edge of the window
        EDGE_PADDING = 20
        size.x -= EDGE_PADDING * 2
        size.y -= EDGE_PADDING * 2
        pixels_h = rect.width(p_r) / self.projected_units_per_pixel
        pixels_v = rect.height(p_r) / self.projected_units_per_pixel
        print "pixels_h = {0}, pixels_v = {1}".format(pixels_h, pixels_v)
        ratio_h = float(pixels_h) / float(size[0])
        ratio_v = float(pixels_v) / float(size[1])
        print "size = %r" % (size,)
        print "ratio_h = %r, ratio_v = %r" % (ratio_h, ratio_v)
        ratio = max(ratio_h, ratio_v)

        print "ratio = %r" % ratio
        self.projected_point_center = rect.center(p_r)
        self.projected_units_per_pixel *= ratio
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str(self.projected_units_per_pixel)

        self.render()

    def zoom_to_include_world_rect(self, w_r):
        view_w_r = self.get_world_rect_from_screen_rect(self.get_screen_rect())
        if (not rect.contains_rect(view_w_r, w_r)):
            # first try just panning
            p_r = self.get_projected_rect_from_world_rect(w_r)
            self.projected_point_center = rect.center(p_r)
            view_w_r = self.get_world_rect_from_screen_rect(self.get_screen_rect())
            if (not rect.contains_rect(view_w_r, w_r)):
                # otherwise we have to zoom (i.e., zoom out because panning didn't work)
                self.zoom_to_world_rect(w_r)

    def reproject_all(self, srs):
        self.update_renderers()
        s_r = self.get_screen_rect()
        s_c = rect.center(s_r)
        w_c = self.get_world_point_from_screen_point(s_c)
        was_identity = self.projection_is_identity

        # print "self.projected_units_per_pixel A = " + str( self.projected_units_per_pixel )
        self.projection = Projection(srs)
        self.projection_is_identity = self.projection.srs.find("+proj=longlat") != -1

        for layer in self.layer_manager.flatten():
            self.layer_renderers[layer].reproject(self.projection, self.projection_is_identity)
        # print "self.projected_units_per_pixel B = " + str( self.projected_units_per_pixel )

        ratio = 1.0
        if (was_identity and not self.projection_is_identity):
            ratio = 40075016.6855801 / 360.0
        if (not was_identity and self.projection_is_identity):
            ratio = 360.0 / 40075016.6855801
        self.projected_units_per_pixel *= ratio
        self.constrain_zoom()
        print "self.projected_units_per_pixel = " + str(self.projected_units_per_pixel)
        # import code; code.interact( local = locals() )

        self.projected_point_center = self.get_projected_point_from_world_point(w_c)

        self.render()

    def get_canvas_as_image(self):
        window_size = self.GetClientSize()

        gl.glReadBuffer(gl.GL_FRONT)

        raw_data = gl.glReadPixels(
            x=0,
            y=0,
            width=window_size[0],
            height=window_size[1],
            format=gl.GL_RGB,
            type=gl.GL_UNSIGNED_BYTE,
            outputType=str,
        )

        bitmap = wx.BitmapFromBuffer(
            width=window_size[0],
            height=window_size[1],
            dataBuffer=raw_data,
        )

        image = wx.ImageFromBitmap(bitmap)

        # Flip the image vertically, because glReadPixel()'s y origin is at
        # the bottom and wxPython's y origin is at the top.
        screenshot = image.Mirror(horizontally=False)
        return screenshot

    def constrain_zoom(self):
        if (self.projection_is_identity):
            min_val = 0.00001
            max_val = 1
        else:
            min_val = 1
            max_val = 80000
        self.projected_units_per_pixel = max(self.projected_units_per_pixel, min_val)
        self.projected_units_per_pixel = min(self.projected_units_per_pixel, max_val)

    def get_surrounding_screen_rects(self, r):
        # return four disjoint rects surround r on the screen
        sr = self.get_screen_rect()

        if (r[0][1] <= sr[0][1]):
            above = rect.EMPTY_RECT
        else:
            above = (sr[0], (sr[1][0], r[0][1]))

        if (r[1][1] >= sr[1][1]):
            below = rect.EMPTY_RECT
        else:
            below = ((sr[0][0], r[1][1]), sr[1])

        if (r[0][0] <= sr[0][0]):
            left = rect.EMPTY_RECT
        else:
            left = ((sr[0][0], r[0][1]), (r[0][0], r[1][1]))

        if (r[1][0] >= sr[1][0]):
            right = rect.EMPTY_RECT
        else:
            right = ((r[1][0], r[0][1]), (sr[1][0], r[1][1]))

        return [above, below, left, right]

    """
    def get_degrees_lon_per_pixel( self, reference_latitude = None ):
        if ( reference_latitude == None ):
            reference_latitude = self.world_point_center[ 1 ]
        factor = math.cos( math.radians( reference_latitude ) )
        ###
        return self.degrees_lat_per_pixel * factor
    
    def get_lon_dist_from_screen_dist( self, screen_dist ):
        return self.get_degrees_lon_per_pixel() * screen_dist
    
    def get_lat_dist_from_screen_dist( self, screen_dist ):
        return self.degrees_lat_per_pixel * screen_dist
    """

import unittest

# imports needed only for tests
import Editor
import Layer
import Layer_manager
import RenderController

import numpy as np


class RenderWindowTests(unittest.TestCase):

    def setUp(self):
        self.layer_manager = Layer_manager.Layer_manager()
        self.editor = Editor.Editor(self.layer_manager)
        self.frame = wx.Frame(None, -1, "Test Frame", size=(400, 440))
        self.frame.CreateStatusBar()
        self.canvas = RenderWindow(self.frame, -1, size=(400, 400), layer_manager=self.layer_manager, editor=self.editor)
        self.render_controller = RenderController.RenderController(self.layer_manager, self.canvas)
        self.frame.Show()

    def tearDown(self):
        # NOTE: The frame cannot be immediately destroyed here, or else it gets messages
        # while being partially deleted. We need to give the thread a bit of time for proper shut down
        wx.CallAfter(self.frame.Destroy)

    def testLonLatGrid(self):
        screen_rect = self.canvas.get_screen_rect()
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(screen_rect)
        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)

        grid = self.canvas.lon_lat_grid
        grid.resize(world_rect, screen_rect)

        steps = np.arange(-10.0, 17.790560385793754, 10.0, dtype=np.float64)
        self.assertEquals(True, (grid.lat_steps == steps).all())
        self.assertEquals(True, (grid.lon_steps == steps).all())

        self.assertEquals(grid.lat_step, 10.0)
        self.assertEquals(grid.lon_step, 10.0)

        layer = Layer.Layer()
        layer.read_from_file(os.path.abspath("TestData/Verdat/000026pts.verdat"))
        self.layer_manager.insert_layer(None, layer)
        self.canvas.Update()
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(screen_rect)
        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)

        self.assertEquals(proj_rect, ((-9305004.516951878, -920746.4576416654), (-1316099.4471520581, 7068158.612158153)))
        self.assertEquals(world_rect, ((-83.58827776379056, -8.297405610026354), (-11.822722487979462, 53.64172954714474)))

        grid.resize(world_rect, screen_rect)
        self.assertEquals(grid.lat_step, 20.0)
        self.assertEquals(grid.lon_step, 20.0)
        print "lat_step = %r" % grid.lat_step
        print "lon_step = %r" % grid.lon_step

        image = self.canvas.get_canvas_as_image()
        imageName = "TestData/TestResults/zoom_to_world_rect1.png"
        if not os.path.exists(imageName):
            image.SaveFile(imageName, wx.BITMAP_TYPE_PNG)

    def testRenderLoadedFile(self):
        layer = Layer.Layer()
        layer.read_from_file(os.path.abspath("TestData/Verdat/000026pts.verdat"))
        self.layer_manager.insert_layer(None, layer)

        # use the actual window size to make a more real-world test.
        newSize = (793, 678)
        self.canvas.Size = newSize

        self.assertEquals(((0, 0), newSize), self.canvas.get_screen_rect())

        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        # print "proj_rect = %r" % (proj_rect,)
        self.assertEquals(proj_rect, ((-13229554.132491037, -3696890.9693971025), (2608450.168387103, 9844303.12391359)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        # print "world_rect = %r" % (world_rect,)
        self.assertEquals(world_rect, ((-118.84310679303275, -31.66425036668012), (23.432106541262737, 66.02389787475978)))

    def testResizeRenderView(self):
        # first make sure the defaults are correct
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-2000000.0, -2000000.0), (2000000.0, 2000000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        # the size the MapRoom app uses for this window.
        newSize = (793, 678)
        self.canvas.Size = newSize

        self.assertEquals(((0, 0), newSize), self.canvas.get_screen_rect())

        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        # print "proj_rect = %r" % (proj_rect,)
        self.assertEquals(proj_rect, ((-3965000.0, -3390000.0), (3965000.0, 3390000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        # print "world_rect = %r" % (world_rect,)
        self.assertEquals(world_rect, ((-35.618201015339025, -29.276545609179184), (35.618201015339025, 29.27654560917917)))

    def testCoordinateConversions(self):
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-2000000.0, -2000000.0), (2000000.0, 2000000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        proj_point = self.canvas.get_projected_point_from_screen_point((0, 0))
        self.assertEquals(proj_point, (-2000000.0, 2000000.0))
        proj_point = self.canvas.get_projected_point_from_screen_point((200, 200))
        self.assertEquals(proj_point, (0.0, 0.0))
        proj_point = self.canvas.get_projected_point_from_screen_point((400, 400))
        self.assertEquals(proj_point, (2000000.0, -2000000.0))

        # We get values like 19999999.99999, so we round the return values before running
        # comparisons.
        proj_point = self.canvas.get_projected_point_from_world_point((-17.966305682390427, 17.790560385793754))
        self.assertEquals((round(proj_point[0]), round(proj_point[1])), (-2000000.0, 2000000.0))
        proj_point = self.canvas.get_projected_point_from_world_point((0.0, 0.0))
        self.assertEquals((round(proj_point[0]), round(proj_point[1])), (0.0, 0.0))
        proj_point = self.canvas.get_projected_point_from_world_point((17.966305682390427, -17.790560385793764))
        self.assertEquals((round(proj_point[0]), round(proj_point[1])), (2000000.0, -2000000.0))

        world_point = self.canvas.get_world_point_from_screen_point((0, 0))
        self.assertEquals(world_point, (-17.966305682390427, 17.790560385793754))
        world_point = self.canvas.get_world_point_from_screen_point((200, 200))
        self.assertEquals(world_point, (0.0, 0.0))
        world_point = self.canvas.get_world_point_from_screen_point((400, 400))
        self.assertEquals(world_point, (17.966305682390427, -17.790560385793764))

        world_point = self.canvas.get_world_point_from_projected_point((-2000000.0, 2000000.0))
        self.assertEquals(world_point, (-17.966305682390427, 17.790560385793754))
        world_point = self.canvas.get_world_point_from_projected_point((0.0, 0.0))
        self.assertEquals(world_point, (0.0, 0.0))
        world_point = self.canvas.get_world_point_from_projected_point((2000000.0, -2000000.0))
        self.assertEquals(world_point, (17.966305682390427, -17.790560385793764))

        screen_point = self.canvas.get_screen_point_from_projected_point((-2000000.0, 2000000.0))
        self.assertEquals(screen_point, (0, 0))
        screen_point = self.canvas.get_screen_point_from_projected_point((0.0, 0.0))
        self.assertEquals(screen_point, (200, 200))
        screen_point = self.canvas.get_screen_point_from_projected_point((2000000.0, -2000000.0))
        self.assertEquals(screen_point, (400, 400))

        screen_point = self.canvas.get_screen_point_from_world_point((-17.966305682390427, 17.790560385793754))
        self.assertEquals(screen_point, (0, 0))
        screen_point = self.canvas.get_screen_point_from_world_point((0.0, 0.0))
        self.assertEquals(screen_point, (200, 200))
        screen_point = self.canvas.get_screen_point_from_world_point((17.966305682390427, -17.790560385793764))
        self.assertEquals(screen_point, (400, 400))

    def testZoom(self):
        screen_rect = self.canvas.get_screen_rect()
        print "screen rect = %r" % (screen_rect, )
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-2000000.0, -2000000.0), (2000000.0, 2000000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        screen_rect = self.canvas.get_screen_rect_from_world_rect(world_rect)
        self.assertEquals(screen_rect, self.canvas.get_screen_rect())

        self.canvas.zoom_in()
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-1000000.0, -1000000.0), (1000000.0, 1000000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        #self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        screen_rect = self.canvas.get_screen_rect_from_world_rect(world_rect)
        self.assertEquals(screen_rect, self.canvas.get_screen_rect())

        self.canvas.zoom_out()
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-2000000.0, -2000000.0), (2000000.0, 2000000.0)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        screen_rect = self.canvas.get_screen_rect_from_world_rect(world_rect)
        self.assertEquals(screen_rect, self.canvas.get_screen_rect())

        self.canvas.zoom(ratio=1.5)
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertEquals(proj_rect, ((-1333333.3333333335, -1333333.3333333335), (1333333.3333333335, 1333333.3333333335)))

        world_rect = self.canvas.get_world_rect_from_projected_rect(proj_rect)
        #self.assertEquals(world_rect, ((-17.966305682390427, -17.790560385793764), (17.966305682390427, 17.790560385793754)))

        screen_rect = self.canvas.get_screen_rect_from_world_rect(world_rect)
        self.assertEquals(screen_rect, self.canvas.get_screen_rect())

    def testZoomToWorldRect(self):
        proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        ratio = 1.234  # an uneven ratio so we don't get nice even numbers
        new_rect = ((proj_rect[0][0] * ratio, proj_rect[0][1] * ratio), (proj_rect[1][0] * ratio, proj_rect[1][1] * ratio))
        w_r = self.canvas.get_world_rect_from_projected_rect(new_rect)
        self.assertEqual(w_r, ((-22.17042121206979, -21.769222673475657), (22.17042121206979, 21.769222673475646)))

        print "world_rect = %r" % (w_r,)

        self.canvas.zoom_to_world_rect(w_r)

        new_proj_rect = self.canvas.get_projected_rect_from_screen_rect(self.canvas.get_screen_rect())
        self.assertNotEqual(proj_rect, new_proj_rect)
        self.assertEqual(new_proj_rect, ((-2742222.2222222225, -2742222.222222222), (2742222.2222222225, 2742222.222222223)))

        # print "new_proj_rect = %r" % (new_proj_rect,)

        new_world_rect = self.canvas.get_world_rect_from_projected_rect(new_proj_rect)
        self.assertNotEqual(w_r, new_world_rect)
        self.assertEqual(new_world_rect, ((-24.63380134674421, -24.051051010416202), (24.63380134674421, 24.051051010416202)))
        # print "new_world_rect = %r" % (new_world_rect,)


def getTestSuite():
    return unittest.makeSuite(RenderWindowTests)
