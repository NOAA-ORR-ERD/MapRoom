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
from traits.api import provides, on_trait_change, Property, Instance, Str, Unicode, Any, List, Event, Dict

from omnivore.framework.task import FrameworkTask
from omnivore.framework.i_about import IAbout

from project_editor import ProjectEditor
import pane_layout
from preferences import MaproomPreferences
from library.mem_use import get_mem_use
from mouse_handler import *
import toolbar
from menu_commands import *
from vector_object_commands import *
from ui.dialogs import StyleDialog
from library.thread_utils import BackgroundWMSDownloader
from library.tile_utils import BackgroundTileDownloader
from omnivore.framework.actions import PreferencesAction, TaskDynamicSubmenuGroup, CutAction, CopyAction, PasteAction

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

class SaveLayerGroup(TaskDynamicSubmenuGroup):
    """ A menu for changing the active task in a task window.
    """
    id = 'SaveLayerGroup'
    
    event_name = Str('layer_selection_changed')

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

class SaveImageAction(EditorAction):
    name = 'Save As Image...'
    tooltip = 'Save a bitmap image of the current view'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as')
        if dialog.open() == OK:
            self.active_editor.save_image(dialog.path)

class DefaultStyleAction(EditorAction):
    name = 'Default Style...'
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
        c = self.active_editor.layer_canvas
        units_per_pixel = c.zoom_in()
        cmd = ViewportCommand(None, c.projected_point_center, units_per_pixel)
        self.active_editor.process_command(cmd)

class ZoomOutAction(EditorAction):
    name = 'Zoom Out'
    tooltip = 'Decrease magnification'
    image = ImageResource('zoom_out')

    def perform(self, event):
        c = self.active_editor.layer_canvas
        units_per_pixel = c.zoom_out()
        cmd = ViewportCommand(None, c.projected_point_center, units_per_pixel)
        self.active_editor.process_command(cmd)

class ZoomToFit(EditorAction):
    name = 'Zoom to Fit'
    tooltip = 'Set magnification to show all layers'
    image = ImageResource('zoom_fit')

    def perform(self, event):
        c = self.active_editor.layer_canvas
        center, units_per_pixel = c.calc_zoom_to_fit()
        cmd = ViewportCommand(None, center, units_per_pixel)
        self.active_editor.process_command(cmd)

class ZoomToLayer(EditorAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'
    enabled_name = 'layer_zoomable'
    image = ImageResource('zoom_to_layer')

    def perform(self, event):
        sel_layer = self.active_editor.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            cmd = ViewportCommand(sel_layer)
            self.active_editor.process_command(cmd)

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

class NewTileLayerAction(EditorAction):
    name = 'New Tile Layer'
    tooltip = 'Create new tile background service layer'

    def perform(self, event):
        cmd = AddLayerCommand("tile")
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

class ToPolygonLayerAction(EditorAction):
    name = 'Convert to Polygon Layer'
    tooltip = 'Create new polygon layer from boundaries of current layer'
    enabled_name = 'layer_has_boundaries'

    def perform(self, event):
        sel_layer = self.active_editor.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            cmd = ToPolygonLayerCommand(sel_layer)
            self.active_editor.process_command(cmd)

class ToVerdatLayerAction(EditorAction):
    name = 'Convert to Editable Layer'
    tooltip = 'Create new editable layer from polygons of current layer'
    enabled_name = 'layer_has_boundaries'

    def perform(self, event):
        sel_layer = self.active_editor.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            cmd = ToVerdatLayerCommand(sel_layer)
            self.active_editor.process_command(cmd)

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
        app.load_file(filename, task_id="omnivore.framework.text_edit_task")

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

class CopyStyleAction(EditorAction):
    name = 'Copy Style'
    accelerator = 'Alt+Ctrl+C'
    tooltip = 'Copy the style of the current selection'
    enabled_name = 'can_copy'

    def perform(self, event):
        self.active_editor.copy_style()

class PasteStyleAction(EditorAction):
    name = 'Paste Style'
    accelerator = 'Alt+Ctrl+V'
    tooltip = 'Apply the style from the clipboard'
    enabled_name = 'can_paste_style'

    def perform(self, event):
        self.active_editor.paste_style()

class DuplicateLayerAction(EditorAction):
    name = 'Duplicate'
    accelerator = 'Ctrl+D'
    tooltip = 'Duplicate the current layer'
    enabled_name = 'can_copy'

    def perform(self, event):
        sel_layer = self.active_editor.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            json_data = sel_layer.serialize_json(-999)
            text = json.dumps(json_data)
            cmd = PasteLayerCommand(sel_layer, text)
            self.active_editor.process_command(cmd)


@provides(IAbout)
class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """
    
    id = pane_layout.task_id_with_pane_layout
    
    new_file_text = 'MapRoom Project'
    
    about_application = ""

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
        toolbars = toolbar.get_all_toolbars()
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
                  ToPolygonLayerAction(),
                  ToVerdatLayerAction(),
                  MergeLayersAction(),
                  MergePointsAction(),
                  id="utilgroup"),
            Group(DeleteLayerAction(),
                  id="deletegroup"),
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
        
        self.init_threaded_processing()
        
        # This trait can't be set as a decorator on the method because
        # active_editor can be None during the initialization process.  Set
        # here because it's guaranteed not to be None
        self.on_trait_change(self.mode_toolbar_changed, 'active_editor.mouse_mode_toolbar')

    def prepare_destroy(self):
        self.stop_threaded_processing()
    
    def get_actions_Menu_File_NewGroup(self):
        return [
            NewProjectAction(),
            NewVectorLayerAction(),
            NewAnnotationLayerAction(),
            NewWMSLayerAction(),
            NewTileLayerAction(),
            NewLonLatLayerAction(),
            ]
    
    def get_actions_Menu_File_SaveGroup(self):
        return [
            SaveProjectAction(),
            SaveProjectAsAction(),
            SaveCommandLogAction(),
            SaveLayerAction(),
            SMenu(SaveLayerGroup(),
                  id='SaveLayerAsSubmenu', name="Save Layer As"),
            SaveImageAction(),
            ]
    
    def get_actions_Menu_Edit_CopyPasteGroup(self):
        return [
            CutAction(),
            CopyAction(),
            DuplicateLayerAction(),
            PasteAction(),
            Separator(),
            CopyStyleAction(),
            PasteStyleAction(),
            ]
    
    def get_actions_Menu_Edit_SelectGroup(self):
        return [
            ClearSelectionAction(),
            DeleteSelectionAction(),
            Separator(),
            BoundaryToSelectionAction(),
            Separator(),
            ClearFlaggedAction(),
            FlaggedToSelectionAction(),
            ]
    
    def get_actions_Menu_Edit_PrefGroup(self):
        return [
            DefaultStyleAction(),
            PreferencesAction(),
            ]
    
    def get_actions_Menu_Edit_FindGroup(self):
        return [
            FindPointsAction(),
            ]
    
    def get_actions_Menu_Help_BugReportGroup(self):
        return [
            OpenLogDirectoryAction(),
            OpenLogAction(),
            ]
    
    def get_actions_Tool_File_SaveGroup(self):
        return [
            SaveProjectAction(),
            SaveProjectAsAction(),
            ]

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

# This trait change is set in activated() rather than as a decorator (see above)
#    @on_trait_change('active_editor.mouse_mode_toolbar')
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


    ##### WMS and Tile processing

    # Traits
    downloaders = Dict
    
    # class attributes
    
    wms_extra_loaded = False
    
    tile_extra_loaded = None
    
    @classmethod
    def init_extra_servers(cls, application):
        if cls.wms_extra_loaded is False:
            # try once
            cls.wms_extra_loaded = True
            try:
                wms_list = application.get_json_data("wms_list")
                BackgroundWMSDownloader.set_known_wms(wms_list)
            except IOError:
                # file not found
                pass
            except ValueError:
                # bad JSON format
                log.error("Invalid format of WMS saved data")
                raise
    
    def remember_wms(self, host=None):
        if host is not None:
            BackgroundWMSDownloader.add_wms_host(host)
        wms_list = BackgroundWMSDownloader.get_known_wms()
        self.window.application.save_json_data("wms_list", wms_list)

    def init_threaded_processing(self):
        self.init_extra_servers(self.window.application)
#        if "OpenStreetMap Test" not in self.get_known_wms_names():
#            BackgroundWMSDownloader.add_wms("OpenStreetMap Test", "http://ows.terrestris.de/osm/service?", "1.1.1")
#            self.remember_wms()
    
    def stop_threaded_processing(self):
        log.debug("Stopping threaded services...")
        while len(self.downloaders) > 0:
            url, wms = self.downloaders.popitem()
            log.debug("Stopping threaded downloader %s" % wms)
            wms = None

    def get_threaded_wms(self, wmshost=None):
        if wmshost is None:
            wmshost = BackgroundWMSDownloader.get_known_wms()[0]
        if wmshost.url not in self.downloaders:
            wms = BackgroundWMSDownloader(wmshost)
            self.downloaders[wmshost.url] = wms
        return self.downloaders[wmshost.url]

    def get_threaded_wms_by_id(self, id):
        wmshost = BackgroundWMSDownloader.get_known_wms()[id]
        return self.get_threaded_wms(wmshost)

    def get_known_wms_names(self):
        return [s.name for s in BackgroundWMSDownloader.get_known_wms()]

    def get_threaded_tile_server(self, tilehost=None):
        if tilehost is None:
            tilehost = BackgroundTileDownloader.get_known_tile_server()[0]
        if tilehost not in self.downloaders:
            cache_dir = os.path.join(self.window.application.cache_dir, "tiles")
            ts = BackgroundTileDownloader(tilehost, cache_dir)
            self.downloaders[tilehost] = ts
        return self.downloaders[tilehost]

    def get_threaded_tile_server_by_id(self, id):
        tilehost = BackgroundTileDownloader.get_known_tile_server()[id]
        return self.get_threaded_tile_server(tilehost)

    def get_known_tile_server_names(self):
        return [s.name for s in BackgroundTileDownloader.get_known_tile_server()]
