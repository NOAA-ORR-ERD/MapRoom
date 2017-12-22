from pyface.api import ImageResource
from pyface.action.api import Group
from pyface.tasks.action.api import SToolBar
from traits.api import Any
from traits.api import on_trait_change

from omnivore.framework.enthought_api import EditorAction

import mouse_handler as modes

valid_mouse_modes = {
    'VectorLayerToolBar': [
        modes.PointSelectionMode,
        modes.PanMode,
        modes.ZoomRectMode,
        modes.RulerMode,
        modes.PointEditMode,
        modes.LineEditMode,
    ],
    'PolygonLayerToolBar': [
        modes.PolygonSelectionMode,
        modes.PanMode,
        modes.ZoomRectMode,
        modes.RulerMode,
        modes.CropRectMode,
    ],
    'AnnotationLayerToolBar': [
        modes.ControlPointEditMode,
        modes.PanMode,
        modes.ZoomRectMode,
        modes.RulerMode,
        modes.AddLineMode,
        modes.AddPolylineMode,
        modes.AddRectangleMode,
        modes.AddEllipseMode,
        modes.AddCircleMode,
        modes.AddPolygonMode,
        modes.AddOverlayTextMode,
        modes.AddOverlayIconMode,
        modes.AddArrowTextMode,
        modes.AddArrowTextIconMode,
    ],
    'BaseLayerToolBar': [
        modes.SelectionMode,
        modes.PanMode,
        modes.ZoomRectMode,
        modes.RulerMode,
    ],
    'RNCToolBar': [
        modes.RNCSelectionMode,
        modes.PanMode,
        modes.ZoomRectMode,
        modes.RulerMode,
    ],
}


def get_valid_mouse_mode(mouse_mode, mode_mode_toolbar_name):
    """
    Return a valid mouse mode for the specified toolbar

    Used when switching modes to guarantee a valid mouse mode.
    """
    valid = valid_mouse_modes.get(mode_mode_toolbar_name, valid_mouse_modes['BaseLayerToolBar'])
    if mouse_mode not in valid:
        group = mouse_mode.toolbar_group
        if group == "select":
            # find another select mode
            for m in valid:
                if m.toolbar_group == "select":
                    return m
        return valid[0]
    return mouse_mode


class MouseHandlerBaseAction(EditorAction):
    """Save a bit of boilerplate with a base class for toolbar mouse mode buttons

    Note that the traits for name, tooltip, and image must be repeated
    in subclasses because the trait initialization appears to reference
    the handler in the class that is named, not superclasses.  E.g.:
    handler.menu_item_name in this base class doesn't appear to look at the
    handler class attribute of subclasses.
    """
    # Traits
    handler = Any

    style = 'radio'

    def _name_default(self):
        return self.handler.menu_item_name

    def _tooltip_default(self):
        return self.handler.menu_item_tooltip

    def _image_default(self):
        return ImageResource(self.handler.icon)

    def perform(self, event):
        self.active_editor.mouse_mode_factory = self.handler
        self.active_editor.update_layer_selection_ui()

    def _update_checked(self, ui_state):
        self.checked = self.active_editor.mouse_mode_factory == self.handler


def get_toolbar_group(toolbar_name):
    """Create the toolbar groups with buttons in the order specified in the
    valid_mouse_modes dict.
    """
    actions = [MouseHandlerBaseAction(handler=mode) for mode in valid_mouse_modes[toolbar_name]]
    return SToolBar(Group(*actions),
                    show_tool_names=False,
                    # image_size=(22,22),
                    id=toolbar_name)


def get_all_toolbars():
    """Return a list of all toolbar definitions for inclusion in a toolbars list
    to be passed up to the Enthough library.
    """
    toolbars = [get_toolbar_group(n) for n in valid_mouse_modes.keys()]
    return toolbars
