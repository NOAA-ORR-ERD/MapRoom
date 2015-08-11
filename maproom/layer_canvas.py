import os
import time

import wx
import pyproj

from peppy2 import get_image_path

import library.coordinates as coordinates
import renderer
import library.rect as rect
from mouse_handler import *

from library.projection import Projection, NullProjection

import preferences

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)

class LayerCanvas(renderer.BaseCanvas):

    """
    The core rendering class for MapRoom app.
    """

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, CropRectMode],
        'AnnotationLayerToolBar': [PanMode, ZoomRectMode, ControlPointSelectionMode, AddLineMode, AddPolylineMode, AddRectangleMode, AddEllipseMode, AddPolygonMode, AddOverlayTextMode, AddOverlayIconMode],
        'default': [PanMode, ZoomRectMode],
        }

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
    
    def __init__(self, *args, **kwargs):
        self.project = kwargs.pop('project')
        self.layer_manager = kwargs.pop('layer_manager')
        self.editor = self.project
        
        renderer.BaseCanvas.__init__(self, *args, **kwargs)
        
        self.layer_renderers = {}

        p = get_image_path("icons/hand.ico", file=__name__)
        self.hand_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        p = get_image_path("icons/hand_closed.ico", file=__name__)
        self.hand_closed_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        self.forced_cursor = None
        self.set_mouse_handler(MouseHandler)  # dummy initial mouse handler
        self.default_pan_mode = PanMode(self)

        self.projection = Projection(preferences.DEFAULT_PROJECTION_STRING)

        self.pick_layer_index_map = {} # provides mapping from pick_layer index to layer index.

    def change_view(self, layer_manager):
        self.layer_manager = layer_manager

    def update_renderers(self):
        for layer in self.layer_manager.flatten():
            if not layer in self.layer_renderers:
                r = layer.create_renderer(self)
                self.layer_renderers[layer] = r
        pass
    
    def remove_renderer_for_layer(self, layer):
        if layer in self.layer_renderers:
            del self.layer_renderers[layer]
        pass

    def rebuild_renderers(self):
        for layer in self.layer_manager.flatten():
            # Don't rebuild image layers because their numpy data has been
            # thrown away.  If editing of image data is allowed at some future
            # point, we'll have to rethink this.
            if not layer.type == "image":
                self.remove_renderer_for_layer(layer)
        self.update_renderers()

    def rebuild_renderer_for_layer(self, layer, in_place=False):
        if layer in self.layer_renderers:
            layer.rebuild_renderer(in_place)
            log.debug("renderer rebuilt")
        else:
            log.warning("layer %s isn't in layer_renderers!" % layer)
            for layer in self.layer_renderers.keys():
                log.warning("  layer: %s" % layer)
    
    def get_selected_layer(self):
        return self.project.layer_tree_control.get_selected_layer()
    
    def set_mouse_handler(self, mode):
        self.release_mouse()
        self.mouse_handler = mode(self)

    def release_mouse(self):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        while self.native.HasCapture():
            self.ReleaseMouse()

    def set_cursor(self, mode=None):
        if (self.forced_cursor is not None):
            self.SetCursor(self.forced_cursor)
            #
            return

        if mode is None:
            mode = self.mouse_handler
        c = mode.get_cursor()
        self.SetCursor(c)

    def get_effective_tool_mode(self, event):
        middle_down = False
        if (event is not None):
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
            mode = self.default_pan_mode
        else:
            mode = self.mouse_handler
        return mode

    def draw_bounding_boxes(self):
        layers = self.layer_manager.flatten()
        for layer in layers:
            w_r = layer.bounds
            if (w_r != rect.EMPTY_RECT) and (w_r != rect.NONE_RECT):
                s_r = self.get_screen_rect_from_world_rect(w_r)
                r, g, b, a = renderer.int_to_color_floats(layer.style.line_color)
                self.overlay.draw_screen_box(s_r, r, g, b, 0.5, stipple_pattern=0xf0f0)

    # functions related to world coordinates, projected coordinates, and screen coordinates

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

    def get_numpy_world_point_from_screen_point(self, screen_point):
        world_point = self.get_world_point_from_projected_point(self.get_projected_point_from_screen_point(screen_point))
        w = np.empty_like(world_point)
        w[:] = world_point
        return w

    def get_world_rect_from_screen_rect(self, screen_rect):
        return self.get_world_rect_from_projected_rect(self.get_projected_rect_from_screen_rect(screen_rect))

    def get_screen_point_from_world_point(self, world_point):
        screen_point = self.get_screen_point_from_projected_point(self.get_projected_point_from_world_point(world_point))
        # screen points are pixels, which should be int values
        return (round(screen_point[0]), round(screen_point[1]))

    def get_numpy_screen_point_from_world_point(self, world_point):
        screen_point = self.get_screen_point_from_projected_point(self.get_projected_point_from_world_point(world_point))
        s = np.empty_like(world_point)
        s[:] = screen_point
        return s

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

    def zoom_to_world_rect(self, w_r, border=True):
        if (w_r == rect.NONE_RECT):
            return
        p_r = self.get_projected_rect_from_world_rect(w_r)
        size = self.get_screen_size()
        if border:
            # so that when we zoom, the points don't hit the very edge of the window
            EDGE_PADDING = 20
        else:
            EDGE_PADDING = 0
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

    def get_zoom_rect(self):
        return self.get_world_rect_from_screen_rect(self.get_screen_rect())

    def zoom_to_include_world_rect(self, w_r):
        view_w_r = self.get_zoom_rect()
        if (not rect.contains_rect(view_w_r, w_r)):
            # first try just panning
            p_r = self.get_projected_rect_from_world_rect(w_r)
            self.projected_point_center = rect.center(p_r)
            view_w_r = self.get_world_rect_from_screen_rect(self.get_screen_rect())
            if (not rect.contains_rect(view_w_r, w_r)):
                # otherwise we have to zoom (i.e., zoom out because panning didn't work)
                self.zoom_to_world_rect(w_r)

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
        from ui.dialogs import JumpCoordsDialog
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
        from ui.dialogs import FindPointDialog
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
        if ( reference_latitude is None ):
            reference_latitude = self.world_point_center[ 1 ]
        factor = math.cos( math.radians( reference_latitude ) )
        ###
        return self.degrees_lat_per_pixel * factor
    
    def get_lon_dist_from_screen_dist( self, screen_dist ):
        return self.get_degrees_lon_per_pixel() * screen_dist
    
    def get_lat_dist_from_screen_dist( self, screen_dist ):
        return self.degrees_lat_per_pixel * screen_dist
    """
