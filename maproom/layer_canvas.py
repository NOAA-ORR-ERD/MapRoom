import numpy as np

import wx

from sawx.filesystem import get_image_path

from . import renderer
from .mouse_handler import MouseHandler, PanMode

"""
The RenderWindow class -- where the opengl rendering really takes place.
"""

import logging
log = logging.getLogger(__name__)


class LayerCanvas(renderer.ScreenCanvas):

    """
    The core rendering class for MapRoom app.
    """

    mouse_is_down = False
    is_alt_key_down = False
    selection_box_is_being_defined = False
    mouse_down_position = (0, 0)
    mouse_move_position = (0, 0)

    def __init__(self, *args, **kwargs):
        renderer.ScreenCanvas.__init__(self, *args, **kwargs)
        self.SetName("Maproom Project")

        p = get_image_path("icons/hand.ico", file=__name__)
        self.hand_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        p = get_image_path("icons/hand_closed.ico", file=__name__)
        self.hand_closed_cursor = wx.Cursor(p, wx.BITMAP_TYPE_ICO, 16, 16)
        self.set_mouse_handler(MouseHandler)  # dummy initial mouse handler
        self.default_pan_mode = PanMode(self)

        self.pick_layer_index_map = {}  # provides mapping from pick_layer index to layer index.

    def rebuild_renderers(self):
        for layer in self.project.layer_manager.flatten():
            # Don't rebuild image layers because their numpy data has been
            # thrown away.  If editing of image data is allowed at some future
            # point, we'll have to rethink this.
            if not layer.type == "image":
                self.remove_renderer_for_layer(layer)
        self.update_renderers()

    def get_edit_layer(self):
        return self.project.current_layer

    def set_mouse_handler(self, mode):
        self.release_mouse()
        self.mouse_handler = mode(self)

    def release_mouse(self):
        self.mouse_is_down = False
        self.selection_box_is_being_defined = False
        while self.native.HasCapture():
            self.ReleaseMouse()

    def set_cursor(self, mode=None):
        if mode is None:
            mode = self.mouse_handler
        c = mode.get_cursor()
        self.SetCursor(c)

    def get_effective_tool_mode(self, event):
        middle_down = False
        if (event is not None):
            try:
                self.is_alt_key_down = event.AltDown()
            except:
                pass
            try:
                middle_down = event.MiddleIsDown()
            except:
                pass
        log.debug("alt key: %s middle down: %s evt %s" % (self.is_alt_key_down, middle_down, event))
        if self.is_alt_key_down or middle_down:
            mode = self.default_pan_mode
        else:
            mode = self.mouse_handler
        return mode

    def do_jump_coords(self):
        prefs = self.project.preferences
        from .ui.dialogs import JumpCoordsDialog
        dialog = JumpCoordsDialog(self, prefs.coordinate_display_format)
        if dialog.ShowModalWithFocus() == wx.ID_OK:
            self.projected_point_center = self.get_projected_point_from_world_point(dialog.lat_lon)
            self.project.refresh()
        dialog.Destroy()

    def do_center_on_point_index(self, layer, index):
        if layer.has_points():
            lat_lon = layer.points[index].x, layer.points[index].y
            self.projected_point_center = self.get_projected_point_from_world_point(lat_lon)
            self.project.refresh()

    def do_center_on_points(self, layer, start_index, count):
        if layer.has_points():
            last_index = start_index + count
            lat = np.sum(layer.points[start_index:last_index].x) / count
            lon = np.sum(layer.points[start_index:last_index].y) / count
            self.projected_point_center = self.get_projected_point_from_world_point((lat, lon))
            self.project.refresh()

    def do_select_points(self, layer, indexes):
        if len(indexes) > 0 and layer.has_points():
            layer.clear_all_point_selections()
            layer.select_points(indexes)
            w_r = layer.compute_selected_bounding_rect()
            self.zoom_to_include_world_rect(w_r)
            self.project.update_layer_contents_ui()
            self.project.refresh()

    def do_find_points(self):
        from .ui.dialogs import FindPointDialog
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
                tlw.SetStatusText("No point #%s in this layer" % values)
            except ValueError:
                tlw = wx.GetApp().GetTopWindow()
                tlw.SetStatusText("Point number must be an integer, not '%s'" % values)
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
