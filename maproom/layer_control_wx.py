import os
import time

import wx
import wx.glcanvas as glcanvas
import pyproj

import library.coordinates as coordinates
import renderer
import library.rect as rect
import app_globals
from mouse_handler import *

import OpenGL
import OpenGL.GL as gl

from library.projection import Projection, NullProjection

import preferences

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)
mouselog = logging.getLogger("mouse")
mouselog.setLevel(logging.INFO)

class LayerControl(glcanvas.GLCanvas):

    """
    The core rendering class for MapRoom app.
    """

    MODE_PAN = 0
    MODE_ZOOM_RECT = 1
    MODE_EDIT_POINTS = 2
    MODE_EDIT_LINES = 3
    MODE_CROP = 4
    
    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'default': [PanMode, ZoomRectMode],
        }

    opengl_renderer = None

    mouse_is_down = False
    is_alt_key_down = False
    selection_box_is_being_defined = False
    mouse_down_position = (0, 0)
    mouse_move_position = (0, 0)

    @classmethod
    def get_valid_mouse_mode(cls, mouse_mode, mode_mode_toolbar_name):
        """
        Return a valid mouse mode for the specified toolbar
        
        Used when switching modes to guarantee a valid mouse mode.
        """
        valid = cls.valid_mouse_modes.get(mode_mode_toolbar_name, cls.valid_mouse_modes['default'])
        if mouse_mode not in valid:
            return valid[0]
        return mouse_mode
    
    context = None
    
    @classmethod
    def init_context(cls, canvas):
        # Only one GLContext is needed for the entire application -- this way,
        # textures can be shared among views.
        if cls.context is None:
            cls.context = glcanvas.GLContext(canvas)

    def __init__(self, *args, **kwargs):
        self.project = kwargs.pop('project')
        self.layer_manager = kwargs.pop('layer_manager')
        self.editor = self.project
        self.layer_renderers = {}

        kwargs['attribList'] = (glcanvas.WX_GL_RGBA,
                                glcanvas.WX_GL_DOUBLEBUFFER,
                                glcanvas.WX_GL_MIN_ALPHA, 8, )

        glcanvas.GLCanvas.__init__(self, *args, **kwargs)

        self.init_context(self)

        p = os.path.join(app_globals.image_path, "cursors", "hand.ico")
        self.hand_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        p = os.path.join(app_globals.image_path, "cursors", "hand_closed.ico")
        self.hand_closed_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        self.forced_cursor = None
        
        self.bounding_boxes_shown = False

        # two variables keep track of what's visible on the screen:
        # (1) the projected point at the center of the screen
        self.projected_point_center = (0, 0)
        # (2) the number of projected units (starts as meters, or degrees; starts as meters) per pixel on the screen (i.e., the zoom level)        
        ## does this get re-set anyway? pretty arbitrary.
        self.projected_units_per_pixel = 10000

        self.projection = Projection(preferences.DEFAULT_PROJECTION_STRING)

        self.pick_layer_index_map = {} # provides mapping from pick_layer index to layer index.

        #self.frame.Bind( wx.EVT_MOVE, self.refresh )
        #self.frame.Bind( wx.EVT_IDLE, self.on_idle )
        # self.frame.Bind( wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll )

        self.Bind(wx.EVT_IDLE, self.on_idle)  # not sure about this -- but it's where the cursors are set.
        self.Bind(wx.EVT_PAINT, self.render)
        # Prevent flashing on Windows by doing nothing on an erase background event.
        ## fixme -- I think you can pass a flag to the Window instead...
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda event: None)
        self.Bind(wx.EVT_SIZE, self.resize_render_pane)
        
        # mouse handler events
        self.mouse_handler = PanMode(self)
        
        self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.Bind(wx.EVT_MOTION, self.on_mouse_motion)
        self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_mouse_wheel_scroll)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.on_mouse_leave)
        self.Bind(wx.EVT_CHAR, self.on_key_char)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_up)
    
    def change_view(self, layer_manager):
        self.layer_manager = layer_manager

    def update_renderers(self):
        for layer in self.layer_manager.layers:
            if not layer in self.layer_renderers:
                r = renderer.LayerRenderer(self)
                self.layer_renderers[layer] = r
                layer.create_renderer(r)
    
    def remove_renderer_for_layer(self, layer):
        if layer in self.layer_renderers:
            del self.layer_renderers[layer]

    def rebuild_renderers(self):
        for layer in self.layer_manager.layers:
            # Don't rebuild image layers because their numpy data has been
            # thrown away.  If editing of image data is allowed at some future
            # point, we'll have to rethink this.
            if not layer.type == "image":
                self.remove_renderer_for_layer(layer)
        self.update_renderers()

    def rebuild_renderer_for_layer(self, layer, in_place=False):
        if layer in self.layer_renderers:
            layer.rebuild_renderer(self.layer_renderers[layer], in_place)
            log.debug("renderer rebuilt")
        else:
            log.warning("layer %s isn't in layer_renderers!" % layer)
            for layer in self.layer_renderers.keys():
                log.warning("  layer: %s" % layer)
    
    def set_mouse_handler(self, mode):
        self.release_mouse()
        self.mouse_handler = mode(self)

    def on_mouse_down(self, event):
        # self.SetFocus() # why would it not be focused?
        mouselog.debug("in on_mouse_down: event=%s" % event)
        self.get_effective_tool_mode(event)  # update alt key state
        self.forced_cursor = None
        self.mouse_is_down = True
        self.selection_box_is_being_defined = False
        self.mouse_down_position = event.GetPosition()
        self.mouse_move_position = self.mouse_down_position

        self.mouse_handler.process_mouse_down(event)

    def release_mouse(self):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        while self.HasCapture():
            self.ReleaseMouse()

    def on_mouse_motion(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        if self.mouse_is_down:
            self.mouse_handler.process_mouse_motion_down(event)
        else:
            self.mouse_handler.process_mouse_motion_up(event)

    def on_mouse_up(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        self.forced_cursor = None
        
        self.mouse_handler.process_mouse_up(event)

    def on_mouse_wheel_scroll(self, event):
        self.get_effective_tool_mode(event)  # update alt key state
        
        self.mouse_handler.process_mouse_wheel_scroll(event)

    def on_mouse_leave(self, event):
        self.SetCursor(wx.StockCursor(wx.CURSOR_ARROW))
        
        self.mouse_handler.process_mouse_leave(event)

    def on_key_down(self, event):
        self.get_effective_tool_mode(event)
        
        self.mouse_handler.process_key_down(event)
        
        event.Skip()

    def on_key_up(self, event):
        self.get_effective_tool_mode(event)
        
        self.mouse_handler.process_key_up(event)
        
        event.Skip()

    def on_key_char(self, event):
        self.get_effective_tool_mode(event)
        self.set_cursor()
        
        self.mouse_handler.process_key_char(event)

    def on_idle(self, event):
        self.set_cursor()

    def set_cursor(self):
        if (self.forced_cursor != None):
            self.SetCursor(self.forced_cursor)
            #
            return

        c = self.mouse_handler.get_cursor()
        self.SetCursor(c)

    def get_effective_tool_mode(self, event):
        middle_down = False
        if (event != None):
            try:
                self.is_alt_key_down = event.AltDown()
                # print self.is_alt_key_down
            except:
                pass
            try:
                middle_down = event.MiddleIsDown()
            except:
                pass
        if self.is_alt_key_down or middle_down:
            mode = PanMode
        else:
            mode = self.project.mouse_mode
        return mode

    def render(self, event=None):
        # Get interactive console here:
#        import traceback
#        traceback.print_stack();
#        import code; code.interact( local = locals() )
        if not self.IsShownOnScreen():
            # log.debug("layer_control_wx.render: not shown yet, so skipping render")
            return

        t0 = time.clock()
        self.SetCurrent(self.context)
        # this has to be here because the window has to exist before making the renderer
        if (self.opengl_renderer == None):
            self.opengl_renderer = renderer.RendererDriver(True)
        self.update_renderers()

        s_r = self.get_screen_rect()
        p_r = self.get_projected_rect_from_screen_rect(s_r)
        w_r = self.get_world_rect_from_projected_rect(p_r)

        if not self.opengl_renderer.prepare_to_render(p_r, s_r):
            return

        """
        self.root_renderer.render()
        self.set_screen_projection_matrix()
        self.box_overlay.render()
        self.set_render_projection_matrix()
        """

        ## fixme -- why is this a function defined in here??
        ##   so that it can be called with and without pick-mode turned on
        ##   but it seems to be in the wrong place -- make it a regular  method?
        def render_layers(pick_mode=False):
            list = self.layer_manager.flatten()
            length = len(list)
            self.layer_manager.pick_layer_index_map = {} # make sure it's cleared
            pick_layer_index = -1
            for i, layer in enumerate(reversed(list)):
                if pick_mode:
                    if layer.pickable:
                        pick_layer_index += 1
                        self.layer_manager.pick_layer_index_map[pick_layer_index] = (length - 1 - i) # looping reversed...
                        renderer = self.layer_renderers[layer]
                        layer.render(self.opengl_renderer,
                                     renderer,
                                     w_r, p_r, s_r,
                                     self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                     pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                     pick_mode)
                else: # not in pick-mode
                    renderer = self.layer_renderers[layer]
                    layer.render(self.opengl_renderer,
                                 renderer,
                                 w_r, p_r, s_r,
                                 self.project.layer_visibility[layer], ##fixme couldn't this be a property of the layer???
                                 pick_layer_index * 10, ##fixme -- this 10 should not be hard-coded here!
                                 pick_mode)

        render_layers()

        self.opengl_renderer.prepare_to_render_screen_objects()
        if (self.bounding_boxes_shown):
            self.draw_bounding_boxes()
        
        self.mouse_handler.render_overlay()

        self.SwapBuffers()

        self.opengl_renderer.prepare_to_render_projected_objects()
        self.opengl_renderer.prepare_to_render_picker(s_r)
        render_layers(pick_mode=True)
        self.opengl_renderer.done_rendering_picker()

        elapsed = time.clock() - t0

        def update_status(message):
            self.project.task.status_bar.debug = message
        wx.CallAfter(update_status, "Render complete, took %f seconds." % elapsed)

        if (event != None):
            event.Skip()

    def draw_bounding_boxes(self):
        layers = self.layer_manager.flatten()
        for layer in layers:
            w_r = layer.bounds
            if (w_r != rect.EMPTY_RECT) and (w_r != rect.NONE_RECT):
                s_r = self.get_screen_rect_from_world_rect(w_r)
                r, g, b, a = renderer.int_to_color(layer.color)
                self.opengl_renderer.draw_screen_box(s_r, r, g, b, 0.5, stipple_pattern=0xf0f0)

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
        w_r = self.layer_manager.accumulate_layer_rects(self.project.layer_visibility)
        if (w_r != rect.NONE_RECT):
            self.zoom_to_world_rect(w_r)

    def zoom_to_world_rect(self, w_r):
        if (w_r == rect.NONE_RECT):
            return
        p_r = self.get_projected_rect_from_world_rect(w_r)
        size = self.get_screen_size()
        # so that when we zoom, the points don't hit the very edge of the window
        EDGE_PADDING = 20
        size.x -= EDGE_PADDING * 2
        size.y -= EDGE_PADDING * 2
        pixels_h = rect.width(p_r) / self.projected_units_per_pixel
        pixels_v = rect.height(p_r) / self.projected_units_per_pixel
        ratio_h = float(pixels_h) / float(size[0])
        ratio_v = float(pixels_v) / float(size[1])
        ratio = max(ratio_h, ratio_v)

        self.projected_point_center = rect.center(p_r)
        self.projected_units_per_pixel *= ratio
        self.constrain_zoom()

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

    # not used???
    # def reproject_all(self, srs):
    #     self.update_renderers()
    #     s_r = self.get_screen_rect()
    #     s_c = rect.center(s_r)
    #     w_c = self.get_world_point_from_screen_point(s_c)

    #     self.projection = Projection(srs)

    #     for layer in self.layer_manager.flatten():
    #         self.layer_renderers[layer].reproject(self.projection)

    #     ratio = 1.0
    #     if (was_identity and not self.projection_is_identity):
    #         ratio = 40075016.6855801 / 360.0
    #     if (not was_identity and self.projection_is_identity):
    #         ratio = 360.0 / 40075016.6855801
    #     self.projected_units_per_pixel *= ratio
    #     self.constrain_zoom()
    #     # log.debug("self.projected_units_per_pixel = " + str(self.projected_units_per_pixel))
    #     # import code; code.interact( local = locals() )

    #     self.projected_point_center = self.get_projected_point_from_world_point(w_c)

    #     self.render()

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
        ## fixme: this  should not be hard coded -- could scale to projection(90,90, inverse=True or ??)
        ## Also should be in some kind of app preferences location...
        min_val = .02
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

    def do_jump_coords(self):
        prefs = self.project.task.get_preferences()
        from ui.Jump_coords_dialog import JumpCoordsDialog
        dialog = JumpCoordsDialog(self, prefs.coordinate_display_format)
        if dialog.ShowModalWithFocus() == wx.ID_OK:
            lat_lon = coordinates.lat_lon_from_format_string(dialog.coords_text.Value)
            self.projected_point_center = self.get_projected_point_from_world_point(lat_lon)
            self.project.refresh()
        dialog.Destroy()
    
    def do_select_points(self, layer, indexes):
        if len(indexes) > 0 and layer.has_points():
            layer.clear_all_point_selections()
            layer.select_points(indexes)
            w_r = layer.compute_selected_bounding_rect()
            self.zoom_to_include_world_rect(w_r)
            self.project.update_layer_contents_ui()
            self.project.refresh()

    def do_find_points(self):
        from ui.Find_point_dialog import FindPointDialog
        dialog = FindPointDialog(self.project)
        if dialog.ShowModalWithFocus() == wx.ID_OK:
            try:
                values, error = dialog.get_values()
                layer = dialog.layer
                self.do_select_points(layer, values)
                if error:
                    tlw = wx.GetApp().GetTopWindow()
                    tlw.SetStatusText(error)
            except IndexError:
                tlw = wx.GetApp().GetTopWindow()
                tlw.SetStatusText(u"No point #%s in this layer" % values)
            except ValueError:
                tlw = wx.GetApp().GetTopWindow()
                tlw.SetStatusText(u"Point number must be an integer, not '%s'" % values)
            except:
                raise
        dialog.Destroy()
        
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
