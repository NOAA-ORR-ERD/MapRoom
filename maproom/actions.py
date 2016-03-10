import os
import sys

# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, YES, NO, OK, CANCEL
from pyface.action.api import Action, ActionItem
from pyface.tasks.action.api import TaskAction, EditorAction
from traits.api import provides, on_trait_change, Property, Instance, Str, Unicode, Any, List, Event, Dict

from menu_commands import *
from vector_object_commands import *
from mouse_commands import *
from ui.dialogs import StyleDialog
from omnivore.framework.actions import TaskDynamicSubmenuGroup

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

class RevertProjectAction(EditorAction):
    name = 'Revert Project'
    tooltip = 'Revert project to last saved version'
    enabled_name = 'document.can_revert'

    def perform(self, event):
        message = "Revert project from\n\n%s?" % self.active_editor.document.metadata.uri
        result = event.task.confirm(message=message, default=NO, title='Revert Project?')
        if result == CANCEL:
            return
        elif result == YES:
            self.active_editor.load_omnivore_document(self.active_editor.document)

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

class CheckSelectedLayerAction(EditorAction):
    name = 'Check Layer For Errors'
    accelerator = 'Ctrl+E'
    enabled_name = 'layer_selected'
    tooltip = 'Check for valid layer construction'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.check_for_errors)

class CheckAllLayersAction(EditorAction):
    name = 'Check All Layers For Errors'
    accelerator = 'Shift+Ctrl+E'
    tooltip = 'Check for valid layer construction'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.check_all_layers_for_errors)

class OpenLogAction(Action):
    name = 'View Command History Log'
    tooltip = 'View the log containing a list of all user commands'

    def perform(self, event):
        app = event.task.window.application
        filename = app.get_log_file_name("command_log", ".mrc")
        app.load_file(filename, task_id="omnivore.framework.text_edit_task")

class DebugAnnotationLayersAction(EditorAction):
    name = 'Sample Annotation Layer'
    tooltip = 'Create a sample annotation layer with examples of all vector objects'

    def perform(self, event):
        GUI.invoke_later(self.after, self.active_editor)
    
    def after(self, project):
        import debug
        lm = project.layer_manager
        debug.debug_objects(lm)
        project.update_default_visibility()
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
            json_data = sel_layer.serialize_json(-999, True)
            if json_data:
                text = json.dumps(json_data)
                cmd = PasteLayerCommand(sel_layer, text, self.active_editor.layer_canvas.world_center)
                self.active_editor.process_command(cmd)
