""" Skeleton sample task

"""
import os
import sys

# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, YES, OK, CANCEL
from pyface.tasks.api import Task, TaskWindow, IEditor, \
    IEditorAreaPane, EditorAreaPane, Editor, DockPane
from pyface.action.api import Group, Separator, Action, ActionItem
from pyface.tasks.action.api import DockPaneToggleGroup, SMenuBar, \
    SMenu, SToolBar, TaskAction, EditorAction, SchemaAddition
from traits.api import provides, on_trait_change, Property, Instance, Str, Unicode, Any, List, Event

from peppy2.framework.task import FrameworkTask
from peppy2.framework.i_about import IAbout

from project_editor import ProjectEditor
import pane_layout
from preferences import MaproomPreferences
from library.mem_use import get_mem_use
from mouse_handler import *
from menu_commands import *
from vector_object_commands import *
from ui.dialogs import StyleDialog
from peppy2.framework.actions import PreferencesAction

import logging
log = logging.getLogger(__name__)

class NewProjectAction(Action):
    """ An action for creating a new empty file that can be edited by a particular task
    """
    name = 'New Project'
    tooltip = 'Open an empty grid to create new layers'
    
    def perform(self, event=None):
        task = event.task.window.application.find_or_create_task_of_type(pane_layout.task_id_with_pane_layout)
        task.new()

class SaveProjectAction(EditorAction):
    name = 'Save Project'
    accelerator = 'Ctrl+S'
    tooltip = 'Save the current project'
    image = ImageResource('file_save')
    enabled_name = 'dirty' # enabled based on state of task.active_editor.dirty

    def perform(self, event):
        self.active_editor.save(None)

class SaveProjectAsAction(EditorAction):
    name = 'Save Project As...'
    accelerator = 'Ctrl+Shift+S'
    tooltip = 'Save the current project with a new name'
    image = ImageResource('file_save_as')

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as', wildcard="MapRoom Project Files (*.maproom)|*.maproom")
        if dialog.open() == OK:
            self.active_editor.save(dialog.path)

class SaveCommandLogAction(EditorAction):
    name = 'Save Command Log...'
    tooltip = 'Save a copy of the command log'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as', wildcard="MapRoom Command Log Files (*.mrc)|*.mrc")
        if dialog.open() == OK:
            self.active_editor.save_log(dialog.path)

class SaveLayerAction(EditorAction):
    name = 'Save Layer'
    tooltip = 'Save the currently selected layer'
    enabled_name = 'layer_can_save'

    def perform(self, event):
        self.active_editor.save_layer(None)

class SaveLayerAsAction(EditorAction):
    name = 'Save Layer As...'
    tooltip = 'Save the current project with a new name'
    enabled_name = 'layer_can_save_as'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as')
        if dialog.open() == OK:
            self.active_editor.save_layer(dialog.path)

class SaveLayerAsFormatAction(EditorAction):
    loader = Any
    
    def _name_default(self):
        return "%s (%s)" % (self.loader.name, self.loader.extensions[0])
    
    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as', wildcard=self.loader.get_file_dialog_wildcard())
        if dialog.open() == OK:
            self.active_editor.save_layer(dialog.path)

class SaveLayerGroup(Group):
    """ A menu for changing the active task in a task window.
    """

    #### 'ActionManager' interface ############################################

    id = 'SaveLayerGroup'
    items = List

    #### 'TaskChangeMenuManager' interface ####################################

    # The ActionManager to which the group belongs.
    manager = Any

    # The window that contains the group.
    task = Instance('peppy2.framework.task.FrameworkTask')

    # ENTHOUGHT QUIRK: This doesn't work: can't have a property depending on
    # a task because this forces task_default to be called very early in the
    # initialization process, before the window hierarchy is defined.
    #
    # active_editor = Property(Instance(IEditor),
    #                         depends_on='task.active_editor')
        
    ###########################################################################
    # Private interface.
    ###########################################################################

    def _get_items(self, layer=None):
        items = []
        if layer is not None:
            from layers.loaders import valid_save_formats
            valid = valid_save_formats(layer)
            if valid:
                for item in valid:
                    action = SaveLayerAsFormatAction(loader=item[0])
                    items.append(ActionItem(action=action))
            
        return items

    def _rebuild(self, layer=None):
        # Clear out the old group, then build the new one.
        self.destroy()
        self.items = self._get_items(layer)

        # Inform our manager that it needs to be rebuilt.
        self.manager.changed = True
        
    #### Trait initializers ###################################################

    def _items_default(self):
        log.debug("SAVELAYERGROUP: _items_default!!!")
        self.task.on_trait_change(self._rebuild, 'layer_selection_changed')
        return self._get_items()

    def _manager_default(self):
        manager = self
        while isinstance(manager, Group):
            manager = manager.parent
        log.debug("SAVELAYERGROUP: _manager_default=%s!!!" % manager)
        return manager
    
    def _task_default(self):
        log.debug("SAVELAYERGROUP: _task_default=%s!!!" % self.manager.controller.task)
        return self.manager.controller.task
    
    # ENTHOUGHT QUIRK: This doesn't work: the trait change decorator never
    # seems to get called, however specifying the on_trait_change in the
    # _items_default method works.
    #
    #    @on_trait_change('task.layer_selection_changed')
    #    def updated_fired(self, event):
    #        log.debug("SAVELAYERGROUP: updated!!!")
    #        self._rebuild(event)

class SaveImageAction(EditorAction):
    name = 'Save As Image...'
    tooltip = 'Save a bitmap image of the current view'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as')
        if dialog.open() == OK:
            self.active_editor.save_image(dialog.path)

class DefaultStyleAction(EditorAction):
    name = 'Set Style...'
    tooltip = 'Choose the line, fill and font styles'

    def perform(self, event):
        dialog = StyleDialog(parent=event.task.window.control, action='save as')
        if dialog.open() == OK:
            self.active_editor.save_layer(dialog.path)

    def perform(self, event):
        GUI.invoke_later(self.show_dialog, self.active_editor)
    
    def show_dialog(self, project):
        dialog = StyleDialog(project)
        status = dialog.ShowModal()
        if status == wx.ID_OK:
            project.layer_manager.update_default_style(dialog.get_style())

class BoundingBoxAction(EditorAction):
    name = 'Show Bounding Boxes'
    tooltip = 'Display or hide bounding boxes for each layer'
    style = 'toggle'

    def perform(self, event):
        value = not self.active_editor.layer_canvas.debug_show_bounding_boxes
        self.active_editor.layer_canvas.debug_show_bounding_boxes = value
        GUI.invoke_later(self.active_editor.layer_canvas.render)

    @on_trait_change('active_editor')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.layer_canvas.debug_show_bounding_boxes

class PickerFramebufferAction(EditorAction):
    name = 'Show Picker Framebuffer'
    tooltip = 'Display the picker framebuffer instead of the normal view'
    style = 'toggle'

    def perform(self, event):
        value = not self.active_editor.layer_canvas.debug_show_picker_framebuffer
        self.active_editor.layer_canvas.debug_show_picker_framebuffer = value
        GUI.invoke_later(self.active_editor.layer_canvas.render)

    @on_trait_change('active_editor')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.layer_canvas.debug_show_picker_framebuffer

class ZoomInAction(EditorAction):
    name = 'Zoom In'
    tooltip = 'Increase magnification'
    image = ImageResource('zoom_in')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_canvas.zoom_in)

class ZoomOutAction(EditorAction):
    name = 'Zoom Out'
    tooltip = 'Decrease magnification'
    image = ImageResource('zoom_out')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_canvas.zoom_out)

class ZoomToFit(EditorAction):
    name = 'Zoom to Fit'
    tooltip = 'Set magnification to show all layers'
    image = ImageResource('zoom_fit')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_canvas.zoom_to_fit)

class ZoomToLayer(EditorAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'
    enabled_name = 'layer_zoomable'
    image = ImageResource('zoom_to_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.zoom_to_selected_layer)

class NewVectorLayerAction(EditorAction):
    name = 'New Verdat Layer'
    tooltip = 'Create new vector (grid) layer'
    image = ImageResource('add_layer')

    def perform(self, event):
        cmd = AddLayerCommand("vector")
        self.active_editor.process_command(cmd)

class NewLonLatLayerAction(EditorAction):
    name = 'New Lon/Lat Layer'
    tooltip = 'Create new longitude/latitude grid layer'

    def perform(self, event):
        cmd = AddLayerCommand("grid")
        self.active_editor.process_command(cmd)

class NewAnnotationLayerAction(EditorAction):
    name = 'New Annotation Layer'
    tooltip = 'Create new annotation layer'

    def perform(self, event):
        cmd = AddLayerCommand("annotation")
        self.active_editor.process_command(cmd)

class NewWMSLayerAction(EditorAction):
    name = 'New WMS Layer'
    tooltip = 'Create new Web Map Service layer'

    def perform(self, event):
        cmd = AddLayerCommand("wms")
        self.active_editor.process_command(cmd)

class DeleteLayerAction(EditorAction):
    name = 'Delete Layer'
    tooltip = 'Remove the layer from the project'
    enabled_name = 'layer_selected'
    image = ImageResource('delete_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.delete_selected_layer)

class RaiseLayerAction(EditorAction):
    name = 'Raise Layer'
    tooltip = 'Move layer up in the stacking order'
    enabled_name = 'layer_above'
    image = ImageResource('raise.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_selected_layer)

class RaiseToTopAction(EditorAction):
    name = 'Raise Layer To Top'
    tooltip = 'Move layer to the top'
    enabled_name = 'layer_above'
    image = ImageResource('raise_to_top.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_to_top)

class LowerToBottomAction(EditorAction):
    name = 'Lower Layer To Bottom'
    tooltip = 'Move layer to the bottom'
    enabled_name = 'layer_below'
    image = ImageResource('lower_to_bottom.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.lower_to_bottom)

class LowerLayerAction(EditorAction):
    name = 'Lower Layer'
    tooltip = 'Move layer down in the stacking order'
    enabled_name = 'layer_below'
    image = ImageResource('lower.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.lower_selected_layer)

class TriangulateLayerAction(EditorAction):
    name = 'Triangulate Layer'
    tooltip = 'Create triangular mesh'
    enabled_name = 'layer_has_points'
    image = ImageResource('triangulate.png')

    def perform(self, event):
        task = self.active_editor.task
        pane = task.window.get_dock_pane('maproom.triangulate_pane')
        pane.visible = True

class MergeLayersAction(EditorAction):
    name = 'Merge Layers'
    tooltip = 'Merge two vector layers'
    enabled_name = 'multiple_layers'
    image = ImageResource('merge.png')

    def perform(self, event):
        GUI.invoke_later(self.show_dialog, self.active_editor)
    
    def show_dialog(self, project):
        layers = project.layer_manager.get_mergeable_layers()

        if len(layers) < 2:
            project.window.error("Merge requires two vector layers.")
            return

        layer_names = [str(layer.name) for layer in layers]

        import wx
        dialog = wx.MultiChoiceDialog(
            project.window.control,
            "Please select two or more vector layers to merge together into one layer.\n\nOnly those layers that support merging are listed.",
            "Merge Layers",
            layer_names
        )

        # If there are exactly two layers, select them both as a convenience
        # to the user.
        if (len(layers) == 2):
            dialog.SetSelections([0, 1])

        result = dialog.ShowModal()
        if result == wx.ID_OK:
            selections = dialog.GetSelections()
            if len(selections) != 2:
                project.window.error("You must select exactly two layers to merge.")
            else:
                cmd = MergeLayersCommand(layers[selections[0]], layers[selections[1]])
                project.process_command(cmd)
        dialog.Destroy()

class MergePointsAction(EditorAction):
    name = 'Merge Duplicate Points'
    tooltip = 'Merge points within a layer'
    enabled_name = 'layer_has_points'
    image = ImageResource('merge_duplicates.png')

    def perform(self, event):
        task = self.active_editor.task
        pane = task.window.get_dock_pane('maproom.merge_points_pane')
        pane.visible = True

class JumpToCoordsAction(EditorAction):
    name = 'Jump to Coordinates'
    accelerator = 'Ctrl+J'
    tooltip = 'Center the screen on the specified coordinates'
    image = ImageResource('jump.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_canvas.do_jump_coords)

class ClearSelectionAction(EditorAction):
    name = 'Clear Selection'
    enabled_name = 'layer_has_selection'
    tooltip = 'Deselects all selected items in the current layer'
    image = ImageResource('clear_selection.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.clear_selection)

class DeleteSelectionAction(EditorAction):
    name = 'Delete Selection'
    accelerator = 'DEL'
    enabled_name = 'layer_has_selection'
    tooltip = 'Deletes the selected items in the current layer'
    image = ImageResource('delete_selection.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.delete_selection)

class ClearFlaggedAction(EditorAction):
    name = 'Clear Flagged'
    enabled_name = 'layer_has_flagged'
    tooltip = 'Deselects all flagged items in the current layer'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.clear_all_flagged)

class FlaggedToSelectionAction(EditorAction):
    name = 'Select Flagged'
    enabled_name = 'layer_has_flagged'
    tooltip = 'Select all flagged items in the current layer'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.select_all_flagged)

class BoundaryToSelectionAction(EditorAction):
    name = 'Select Boundary'
    enabled_name = 'layer_has_points'
    tooltip = 'Select the boundary of the current layer'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.select_boundary)

class MouseHandlerBaseAction(EditorAction):
    """Save a bit of boilerplate with a base class for toolbar mouse mode buttons
    
    Note that the traits for name, tooltip, and image must be repeated
    in subclasses because the trait initialization appears to reference
    the handler in the class that is named, not superclasses.  E.g.:
    handler.menu_item_name in this base class doesn't appear to look at the
    handler class attribute of subclasses.
    """
    handler = MouseHandler # Not a trait
    
    # Traits
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = self.__class__.handler
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == self.__class__.handler

class ZoomModeAction(MouseHandlerBaseAction):
    handler = ZoomRectMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class PanModeAction(MouseHandlerBaseAction):
    handler = PanMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddPointsAction(MouseHandlerBaseAction):
    handler = PointSelectionMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)
    enabled_name = handler.editor_trait_for_enabled

class AddLinesAction(MouseHandlerBaseAction):
    handler = LineSelectionMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)
    enabled_name = handler.editor_trait_for_enabled

class CropAction(MouseHandlerBaseAction):
    handler = CropRectMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class ControlPointAction(MouseHandlerBaseAction):
    handler = ControlPointSelectionMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddRectangleObjectAction(MouseHandlerBaseAction):
    handler = AddEllipseMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddEllipseObjectAction(MouseHandlerBaseAction):
    handler = AddRectangleMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddLineObjectAction(MouseHandlerBaseAction):
    handler = AddLineMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddPolylineObjectAction(MouseHandlerBaseAction):
    handler = AddPolylineMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddPolygonObjectAction(MouseHandlerBaseAction):
    handler = AddPolygonMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddOverlayTextAction(MouseHandlerBaseAction):
    handler = AddOverlayTextMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)

class AddOverlayIconAction(MouseHandlerBaseAction):
    handler = AddOverlayIconMode
    name = handler.menu_item_name
    tooltip = handler.menu_item_tooltip
    image = ImageResource(handler.icon)


class FindPointsAction(EditorAction):
    name = 'Find Points'
    accelerator = 'Ctrl+F'
    enabled_name = 'layer_has_points'
    tooltip = 'Find and highlight points or ranges of points'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_canvas.do_find_points)

class CheckLayerErrorAction(EditorAction):
    name = 'Check For Errors'
    enabled_name = 'layer_selected'
    tooltip = 'Check for valid layer construction'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.check_for_errors)

class OpenLogAction(Action):
    name = 'View Command History Log'
    tooltip = 'View the log containing a list of all user commands'

    def perform(self, event):
        app = event.task.window.application
        filename = app.get_log_file_name("command_log", ".mrc")
        app.load_file(filename, task_id="peppy.framework.text_edit_task")

class OpenLogDirectoryAction(Action):
    name = 'Open Log Directory in File Manager'
    tooltip = 'Open the log directory in the desktop file manager program'

    def perform(self, event):
        app = event.task.window.application
        filename = app.get_log_file_name("dummy")
        dirname = os.path.dirname(filename)
        import subprocess
        if sys.platform.startswith("win"):
            file_manager = 'explorer'
        elif sys.platform == "darwin":
            file_manager = '/usr/bin/open'
        else:
            file_manager = 'xdg-open'
        subprocess.call([file_manager, dirname])

class DebugAnnotationLayersAction(EditorAction):
    name = 'Sample Annotation Layer'
    tooltip = 'Create a sample annotation layer with examples of all vector objects'

    def perform(self, event):
        GUI.invoke_later(self.after, self.active_editor)
    
    def after(self, project):
        import debug
        lm = project.layer_manager
        debug.debug_objects(lm)
        lm.update_default_visibility(project.layer_visibility)
        project.layer_metadata_changed(None)
        project.layer_canvas.zoom_to_fit()


@provides(IAbout)
class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """
    
    id = pane_layout.task_id_with_pane_layout
    
    new_file_text = 'MapRoom Project'

    #### Task interface #######################################################

    name = 'MapRoom Project File'
    
    icon = ImageResource('maproom')
    
    preferences_helper = MaproomPreferences
    
    status_bar_debug_width = 300
    
    start_new_editor_in_new_window = True
    
    #### 'IAbout' interface ###################################################
    
    about_title = Str('MapRoom')
    
    about_version = Unicode
    
    about_description = Property(Unicode)
    
    about_website = Str('http://www.noaa.gov')
    
    about_image = Instance(ImageResource, ImageResource('maproom_large'))
    
    #### 'IErrorReporter' interface ###########################################
    
    error_email_to = Str('rob.mcmullen@noaa.gov')
    
    #### Menu events ##########################################################
    
    # Layer selection event placed here instead of in the ProjectEditor
    # because the trait events don't seem to be triggered in the
    # menu items on task.active_editor.layer_selection_changed
    # but they are on task.layer_selection_changed.  This means
    # ProjectEditor.update_layer_selection_ui() sets an event here in the
    # MaproomTask rather than in itself.
    layer_selection_changed = Event
    
    def _about_version_default(self):
        import Version
        return Version.VERSION
    
    def _get_about_description(self):
        desc = "High-performance 2d mapping developed by NOAA\n\nMemory usage: %.0fMB\n\nUsing libraries:\n" % get_mem_use()
        import wx
        desc += "  wxPython %s\n" % wx.version()
        try:
            import gdal
            desc += "  GDAL %s\n" % gdal.VersionInfo()
        except:
            pass
        try:
            import numpy
            desc += "  numpy %s\n" % numpy.version.version
        except:
            pass
        try:
            import OpenGL
            desc += "  PyOpenGL %s\n" % OpenGL.__version__
        except:
            pass
        try:
            import pyproj
            desc += "  PyProj %s\n" % pyproj.__version__
        except:
            pass
        return desc

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def _default_layout_default(self):
        return pane_layout.pane_layout()

    def create_dock_panes(self):
        """ Create the file browser and connect to its double click event.
        """
        return pane_layout.pane_create()

    def _tool_bars_default(self):
        toolbars = [
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction()),
                     show_tool_names=False,
#                     image_size=(22,22),
                     id="BaseLayerToolBar",),
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction(),
                           AddPointsAction(),
                           AddLinesAction()),
                     show_tool_names=False,
                     id="VectorLayerToolBar",),
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction(),
                           CropAction()),
                     show_tool_names=False,
                     id="PolygonLayerToolBar",),
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction(),
                           ControlPointAction(),
                           AddLineObjectAction(),
                           AddPolylineObjectAction(),
                           AddRectangleObjectAction(),
                           AddEllipseObjectAction(),
                           AddPolygonObjectAction(),
                           AddOverlayTextAction(),
                           AddOverlayIconAction(),
                           ),
                     show_tool_names=False,
                     id="AnnotationLayerToolBar",),
            ]
        toolbars.extend(FrameworkTask._tool_bars_default(self))
        return toolbars

    def _extra_actions_default(self):
        # FIXME: Is there no way to add an item to an existing group?
        zoomgroup = lambda : Group(ZoomInAction(),
                                   ZoomOutAction(),
                                   ZoomToFit(),
                                   ZoomToLayer(),
                                   id="zoomgroup")
        layer = lambda: SMenu(
            Separator(id="LayerMenuStart", separator=False),
            id= 'Layer', name="Layer"
        )
        layertools = lambda : Group(
            RaiseToTopAction(),
            RaiseLayerAction(),
            LowerLayerAction(),
            LowerToBottomAction(),
            TriangulateLayerAction(),
            DeleteLayerAction(),
            id="layertools")
        layermenu = lambda : Group(
            Separator(id="LayerMainMenuStart", separator=False),
            Group(RaiseToTopAction(),
                  RaiseLayerAction(),
                  LowerLayerAction(),
                  LowerToBottomAction(),
                  id="raisegroup", separator=False),
            Group(TriangulateLayerAction(),
                  MergeLayersAction(),
                  MergePointsAction(),
                  id="utilgroup"),
            Group(DeleteLayerAction(),
                  id="deletegroup"),
            Group(ZoomModeAction(),
                  PanModeAction(),
                  AddPointsAction(),
                  AddLinesAction(),
                  id="modegroup"),
            Group(CheckLayerErrorAction(),
                  id="checkgroup"),
            id="layermenu")
        edittools = lambda : Group(
            ClearSelectionAction(),
            DeleteSelectionAction(),
            id="edittools")
        actions = [
            # Menubar additions
            SchemaAddition(id='bb',
                           factory=BoundingBoxAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(id='pfb',
                           factory=PickerFramebufferAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(id='jump',
                           factory=JumpToCoordsAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(factory=layer,
                           path='MenuBar',
                           after="Edit",
                           ),
            SchemaAddition(factory=layermenu,
                           path='MenuBar/Layer',
                           after='New',
                           ),
            SchemaAddition(factory=zoomgroup,
                           path='MenuBar/View',
                           absolute_position="first",
                           ),
            SchemaAddition(id='dal',
                           factory=DebugAnnotationLayersAction,
                           path='MenuBar/Help/Debug',
                           ),
            
            # Toolbar additions
            SchemaAddition(id="layer",
                           factory=layertools,
                           path='ToolBar',
                           after="Undo",
                           ),
            SchemaAddition(id="edit",
                           factory=edittools,
                           path='ToolBar',
                           before="layer",
                           after="Undo",
                           ),
            SchemaAddition(id="zoom",
                           factory=zoomgroup,
                           path='ToolBar',
                           after="layer",
                           ),
            ]
        return actions

    ###########################################################################
    # 'FrameworkTask' interface.
    ###########################################################################
    
    def activated(self):
        FrameworkTask.activated(self)
        visible = pane_layout.pane_initially_visible()
        for pane in self.window.dock_panes:
            if pane.id in visible:
                pane.visible = visible[pane.id]
    
    def get_actions(self, location, menu_name, group_name):
        if location == "Menu":
            if menu_name == "File":
                if group_name == "NewGroup":
                    return [
                        NewProjectAction(),
                        NewVectorLayerAction(),
                        NewAnnotationLayerAction(),
                        NewWMSLayerAction(),
                        NewLonLatLayerAction(),
                        ]
                elif group_name == "SaveGroup":
                    return [
                        SaveProjectAction(),
                        SaveProjectAsAction(),
                        SaveCommandLogAction(),
                        SaveLayerAction(),
                        SMenu(SaveLayerGroup(),
                            id='SaveLayerAsSubmenu', name="Save Layer As"),
                        SaveImageAction(),
                        ]
            if menu_name == "Edit":
                if group_name == "SelectGroup":
                    return [
                        ClearSelectionAction(),
                        DeleteSelectionAction(),
                        Separator(),
                        BoundaryToSelectionAction(),
                        Separator(),
                        ClearFlaggedAction(),
                        FlaggedToSelectionAction(),
                        ]
                elif group_name == "PrefGroup":
                    return [
                        Group(
                            DefaultStyleAction(),
                            PreferencesAction(),
                            absolute_position="last"),
                        ]
                elif group_name == "FindGroup":
                    return [
                        FindPointsAction(),
                        ]
            if menu_name == "Help":
                if group_name == "BugReportGroup":
                    return [
                        OpenLogDirectoryAction(),
                        OpenLogAction(),
                        ]

        if location.startswith("Tool"):
            if menu_name == "File":
                if group_name == "SaveGroup":
                    return [
                        SaveProjectAction(),
                        SaveProjectAsAction(),
                        ]
        
        # fall back to parent if it's not found here
        return FrameworkTask.get_actions(self, location, menu_name, group_name)

    def get_editor(self, guess=None):
        """ Opens a new empty window
        """
        editor = ProjectEditor()
        return editor

    def new(self, source=None, **kwargs):
        """Open a maproom file.
        
        If the file is a maproom project, it will open a new tab.
        
        If the file is something that can be added as a layer, it will be added
        to the current project, unless a project doesn't exist in which case
        it will open in a new, empty project.
        
        :param source: optional :class:`FileGuess` or :class:`Editor` instance
        that will load a new file or create a new view of the existing editor,
        respectively.
        """
        log.debug("In new...")
        log.debug(" active editor is: %s"%self.active_editor)
        if self.active_editor and hasattr(source, 'get_metadata') and not self.active_editor.load_in_new_tab(source):
            editor = self.active_editor
            editor.load(source, **kwargs)
            self._active_editor_changed()
            self.activated()
            self.window.application.successfully_loaded_event = source.metadata.uri
        else:
            FrameworkTask.new(self, source, **kwargs)

    def allow_different_task(self, guess, other_task):
        return self.window.confirm("The (MIME type %s) file\n\n%s\n\ncan't be edited in a MapRoom project.\nOpen a new %s window to edit?" % (guess.metadata.mime, guess.metadata.uri, other_task.new_file_text)) == YES
    
    @on_trait_change('active_editor.mouse_mode_toolbar')
    def mode_toolbar_changed(self, changed_to):
        for toolbar in self.window.tool_bar_managers:
            name = toolbar.id
            if name == "ToolBar" or name == changed_to:
                state = True
            else:
                state = False
            toolbar.visible = state
            log.debug("toolbar: %s = %s" % (name, state))

    def _active_editor_changed(self):
        tree = self.window.get_dock_pane('maproom.layer_selection_pane')
        if tree is not None and tree.control is not None:
            # We must be in an event handler during trait change callbacks,
            # because we segfault without the GUI.invoke_later (equivalent
            # to wx.CallAfter)
            GUI.invoke_later(tree.control.set_project, self.active_editor)
    
    def _wx_on_mousewheel_from_window(self, event):
        if self.active_editor:
            self.active_editor.layer_canvas.on_mouse_wheel_scroll(event)
    
    @on_trait_change('window.application.preferences_changed_event')
    def preferences_changed(self, evt):
        if self.active_editor:
            self.active_editor.refresh()

    ###
    @classmethod
    def can_edit(cls, mime):
        return ( mime.startswith("image") or
                 mime.startswith("application/x-maproom-") or
                 mime == "application/x-nc_ugrid" or
                 mime == "application/x-nc_particles"
                 )
