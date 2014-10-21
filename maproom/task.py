""" Skeleton sample task

"""
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
from peppy2.utils.jobs import create_global_job_manager

from project_editor import ProjectEditor
import pane_layout
from layer_control_wx import LayerControl
from preferences import MaproomPreferences
from library.mem_use import get_mem_use

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
            
            log.debug("Save Layer As: layer=%s, formats=%s" % (layer, valid))
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

class BoundingBoxAction(EditorAction):
    name = 'Show Bounding Boxes'
    tooltip = 'Display or hide bounding boxes for each layer'
    style = 'toggle'

    def perform(self, event):
        value = not self.active_editor.control.bounding_boxes_shown
        self.active_editor.control.bounding_boxes_shown = value
        GUI.invoke_later(self.active_editor.control.render)

    @on_trait_change('active_editor')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.control.bounding_boxes_shown

class ZoomInAction(EditorAction):
    name = 'Zoom In'
    tooltip = 'Increase magnification'
    image = ImageResource('zoom_in')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_in)

class ZoomOutAction(EditorAction):
    name = 'Zoom Out'
    tooltip = 'Decrease magnification'
    image = ImageResource('zoom_out')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_out)

class ZoomToFit(EditorAction):
    name = 'Zoom to Fit'
    tooltip = 'Set magnification to show all layers'
    image = ImageResource('zoom_fit')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_to_fit)

class ZoomToLayer(EditorAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'
    enabled_name = 'layer_zoomable'
    image = ImageResource('zoom_to_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.zoom_to_selected_layer)

class NewVectorLayerAction(EditorAction):
    name = 'New Ugrid Layer'
    tooltip = 'Create new vector (grid) layer'
    image = ImageResource('add_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_manager.add_layer, "vector", self.active_editor)

class NewLonLatLayerAction(EditorAction):
    name = 'New Lon/Lat Layer'
    tooltip = 'Create new longitude/latitude grid layer'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_manager.add_layer, "grid", self.active_editor)

class DeleteLayerAction(EditorAction):
    name = 'Delete Layer'
    tooltip = 'Remove the layer from the project'
    enabled_name = 'layer_selected'
    image = ImageResource('delete_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_manager.delete_selected_layer)

class RaiseLayerAction(EditorAction):
    name = 'Raise Layer'
    tooltip = 'Move layer up in the stacking order'
    enabled_name = 'layer_above'
    image = ImageResource('raise.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_selected_layer)

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
                project.layer_manager.merge_layers(layers[selections[0]], layers[selections[1]])
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
        GUI.invoke_later(self.active_editor.control.do_jump_coords)

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
        GUI.invoke_later(self.active_editor.clear_flagged)

class ZoomModeAction(EditorAction):
    name = 'Zoom Mode'
    tooltip = 'Zoom to box'
    image = ImageResource('zoom_box.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_ZOOM_RECT
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_ZOOM_RECT

class PanModeAction(EditorAction):
    name = 'Pan Mode'
    tooltip = 'Pan the viewport'
    image = ImageResource('pan.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_PAN
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_PAN

class AddPointsAction(EditorAction):
    name = 'Add Points Mode'
    enabled_name = 'layer_has_points'
    tooltip = 'Add points to the current layer'
    image = ImageResource('add_points.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_EDIT_POINTS
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_EDIT_POINTS

class AddLinesAction(EditorAction):
    name = 'Add Lines Mode'
    enabled_name = 'layer_has_points'
    tooltip = 'Add lines to the current layer'
    image = ImageResource('add_lines.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_EDIT_LINES
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_EDIT_LINES

class CropAction(EditorAction):
    name = 'Crop'
    tooltip = 'Crop layer'
    image = ImageResource('crop.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_CROP
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_CROP

class FindPointsAction(EditorAction):
    name = 'Find Points'
    accelerator = 'Ctrl+F'
    enabled_name = 'layer_has_points'
    tooltip = 'Find and highlight points or ranges of points'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.do_find_points)

class CheckLayerErrorAction(EditorAction):
    name = 'Check For Errors'
    enabled_name = 'layer_selected'
    tooltip = 'Check for valid layer construction'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.check_for_errors)


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
            NewVectorLayerAction(),
            RaiseLayerAction(),
            LowerLayerAction(),
            TriangulateLayerAction(),
            DeleteLayerAction(),
            id="layertools")
        layermenu = lambda : Group(
            Separator(id="LayerMainMenuStart", separator=False),
            Group(RaiseLayerAction(),
                  LowerLayerAction(),
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
            pane.visible = (pane.id in visible)
        
        self.init_background_processing()
    
    def get_actions(self, location, menu_name, group_name):
        if location == "Menu":
            if menu_name == "File":
                if group_name == "NewGroup":
                    return [
                        NewProjectAction(),
                        NewVectorLayerAction(),
                        NewLonLatLayerAction(),
                        ]
                elif group_name == "SaveGroup":
                    return [
                        SaveProjectAction(),
                        SaveProjectAsAction(),
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
                        ClearFlaggedAction(),
                        ]
                elif group_name == "FindGroup":
                    return [
                        FindPointsAction(),
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
    
    @on_trait_change('active_editor.mouse_mode_category')
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
            self.active_editor.control.on_mouse_wheel_scroll(event)

    ###
    @classmethod
    def can_edit(cls, mime):
        return mime.startswith("image") or mime.startswith("application/x-maproom-") or mime == "application/x-hdf"
    
    
    
    # Not traits, just normal class instances
    
    job_manager = None
    
    def init_background_processing(self):
        if self.job_manager is None:
            self.__class__.job_manager = create_global_job_manager(self.receive_job_event)
    
    def receive_job_event(self, event):
        log.debug("receive_job_event: received %s" % repr(event))
        GUI.invoke_later(self.process_job_event, event)
    
    def process_job_event(self, event):
        log.debug("process_job_event: handling %s" % repr(event))
        self.job_manager.get_finished()
        self.job_manager.handle_job_id_callback(event)
