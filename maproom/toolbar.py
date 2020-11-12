import sys
import inspect

from maproom.app_framework.action import MafRadioAction

from . import mouse_handler as modes

valid_mouse_modes = {
    'VectorLayerToolBar': [
        "point_selection_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
        "point_edit_mode",
        "line_edit_mode",
    ],
    'PolygonLayerToolBar': [
        "polygon_selection_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
        "crop_rect_mode",
    ],
    'AnnotationLayerToolBar': [
        "control_point_edit_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
        "add_line_mode",
        "add_polyline_mode",
        "add_rectangle_mode",
        "add_ellipse_mode",
        "add_circle_mode",
        "add_polygon_mode",
        "add_overlay_text_mode",
        "add_overlay_icon_mode",
        "add_arrow_text_mode",
        "add_arrow_text_icon_mode",
    ],
    'BaseLayerToolBar': [
        "selection_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
    ],
    'StickyLayerToolBar': [
        "sticky_selection_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
    ],
    'RNCToolBar': [
        "rnc_selection_mode",
        "pan_mode",
        "zoom_rect_mode",
        "ruler_mode",
    ],
}


def get_valid_mouse_mode(mouse_mode, mode_mode_toolbar_name):
    """
    Return a valid mouse mode for the specified toolbar

    Used when switching modes to guarantee a valid mouse mode.
    """
    valid = valid_mouse_modes.get(mode_mode_toolbar_name, valid_mouse_modes['BaseLayerToolBar'])
    action_key = mode_to_action_key[mouse_mode]
    if action_key not in valid:
        group = mouse_mode.toolbar_group
        if group == "select":
            # find another select mode
            for action_key in valid:
                m = action_key_to_mode[action_key]
                if m.toolbar_group == "select":
                    return m
        return action_key_to_mode[valid[0]]
    return mouse_mode


class MouseModeTool(MafRadioAction):
    mouse_mode_cls = None

    def calc_name(self, action_key):
        return self.mouse_mode_cls.menu_item_tooltip

    def calc_icon_name(self, action_key):
        return self.mouse_mode_cls.icon

    def calc_enabled(self, action_key):
        return True

    def calc_checked(self, action_key):
        return self.editor.mouse_mode_factory == self.mouse_mode_cls

    def perform(self, action_key):
        self.editor.mouse_mode_factory = self.mouse_mode_cls
        self.editor.update_layer_selection_ui()

# Base layer tools

class selection_mode(MouseModeTool):
    mouse_mode_cls = modes.SelectionMode

class pan_mode(MouseModeTool):
    mouse_mode_cls = modes.PanMode

class zoom_rect_mode(MouseModeTool):
    mouse_mode_cls = modes.ZoomRectMode

class ruler_mode(MouseModeTool):
    mouse_mode_cls = modes.RulerMode

# Point layer tools

class point_selection_mode(MouseModeTool):
    mouse_mode_cls = modes.PointSelectionMode

class point_edit_mode(MouseModeTool):
    mouse_mode_cls = modes.PointEditMode

class line_edit_mode(MouseModeTool):
    mouse_mode_cls = modes.LineEditMode

# Polygon layer tools

class polygon_selection_mode(MouseModeTool):
    mouse_mode_cls = modes.PolygonSelectionMode

class crop_rect_mode(MouseModeTool):
    mouse_mode_cls = modes.CropRectMode

# Annotation layer tools
class control_point_edit_mode(MouseModeTool):
    mouse_mode_cls = modes.ControlPointEditMode

class add_line_mode(MouseModeTool):
    mouse_mode_cls = modes.AddLineMode

class add_polyline_mode(MouseModeTool):
    mouse_mode_cls = modes.AddPolylineMode

class add_rectangle_mode(MouseModeTool):
    mouse_mode_cls = modes.AddRectangleMode

class add_ellipse_mode(MouseModeTool):
    mouse_mode_cls = modes.AddEllipseMode

class add_circle_mode(MouseModeTool):
    mouse_mode_cls = modes.AddCircleMode

class add_polygon_mode(MouseModeTool):
    mouse_mode_cls = modes.AddPolygonMode

class add_overlay_text_mode(MouseModeTool):
    mouse_mode_cls = modes.AddOverlayTextMode

class add_overlay_icon_mode(MouseModeTool):
    mouse_mode_cls = modes.AddOverlayIconMode

class add_arrow_text_mode(MouseModeTool):
    mouse_mode_cls = modes.AddArrowTextMode

class add_arrow_text_icon_mode(MouseModeTool):
    mouse_mode_cls = modes.AddArrowTextIconMode

# RNC download layer tools

class rnc_selection_mode(MouseModeTool):
    mouse_mode_cls = modes.RNCSelectionMode

# Sticky screen layer tools

class sticky_selection_mode(MouseModeTool):
    mouse_mode_cls = modes.StickySelectionMode


# build lookup dicts between the action_Key and mouse mode class needed for the
# get_valid_mouse_mode function

def is_tool(obj):
    return inspect.isclass(obj) and MouseModeTool in obj.__mro__[1:]

tools = inspect.getmembers(sys.modules[__name__], is_tool)
action_key_to_mode = {action_key:mode.mouse_mode_cls for action_key, mode in tools}
mode_to_action_key = {mode.mouse_mode_cls:action_key for action_key, mode in tools}
