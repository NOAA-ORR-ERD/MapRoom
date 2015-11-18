import os
import time

import wx
import pyproj

from omnimon import get_image_path

import library.coordinates as coordinates
import renderer
import library.rect as rect
from mouse_handler import *

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)

class LayerCanvas(renderer.ScreenCanvas):

    """
    The core rendering class for MapRoom app.
    """

    valid_mouse_modes = {
        'VectorLayerToolBar': [PanMode, ZoomRectMode, RulerMode, PointSelectionMode, LineSelectionMode],
        'PolygonLayerToolBar': [PanMode, ZoomRectMode, RulerMode, CropRectMode],
        'AnnotationLayerToolBar': [PanMode, ZoomRectMode, RulerMode, ControlPointSelectionMode, AddLineMode, AddPolylineMode, AddRectangleMode, AddEllipseMode, AddCircleMode, AddPolygonMode, AddOverlayTextMode, AddOverlayIconMode],
        'default': [PanMode, ZoomRectMode, RulerMode],
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
        renderer.ScreenCanvas.__init__(self, *args, **kwargs)

        p = get_image_path("icons/hand.ico", file=__name__)
        self.hand_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        p = get_image_path("icons/hand_closed.ico", file=__name__)
        self.hand_closed_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        self.forced_cursor = None
        self.set_mouse_handler(MouseHandler)  # dummy initial mouse handler
        self.default_pan_mode = PanMode(self)

        self.pick_layer_index_map = {} # provides mapping from pick_layer index to layer index.

    def rebuild_renderers(self):
        for layer in self.layer_manager.flatten():
            # Don't rebuild image layers because their numpy data has been
            # thrown away.  If editing of image data is allowed at some future
            # point, we'll have to rethink this.
            if not layer.type == "image":
                self.remove_renderer_for_layer(layer)
        self.update_renderers()

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
