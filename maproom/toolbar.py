from pyface.api import ImageResource
from pyface.action.api import Group
from pyface.tasks.action.api import SToolBar, EditorAction
from traits.api import Any

from mouse_handler import *

valid_mouse_modes = {
    'VectorLayerToolBar': [PanMode, ZoomRectMode, RulerMode, PointSelectionMode, LineSelectionMode],
    'PolygonLayerToolBar': [PanMode, ZoomRectMode, RulerMode, CropRectMode],
    'AnnotationLayerToolBar': [PanMode, ZoomRectMode, RulerMode, ControlPointSelectionMode, AddLineMode, AddPolylineMode, AddRectangleMode, AddEllipseMode, AddCircleMode, AddPolygonMode, AddOverlayTextMode, AddOverlayIconMode],
    'BaseLayerToolBar': [PanMode, ZoomRectMode, RulerMode],
    }

def get_valid_mouse_mode(mouse_mode, mode_mode_toolbar_name):
    """
    Return a valid mouse mode for the specified toolbar
    
    Used when switching modes to guarantee a valid mouse mode.
    """
    valid = valid_mouse_modes.get(mode_mode_toolbar_name, valid_mouse_modes['BaseLayerToolBar'])
    if mouse_mode not in valid:
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
        self.active_editor.mouse_mode = self.handler
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == self.handler

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
