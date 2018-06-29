import sys
import wx
import json

# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, OK
from traits.api import Any
from traits.api import Str
from traits.api import on_trait_change

from omnivore.framework.enthought_api import Action, ActionItem, EditorAction, TaskDynamicSubmenuGroup
from omnivore.utils.wx.dialogs import ListReorderDialog, CheckItemDialog

from . import pane_layout
from .menu_commands import AddLayerCommand, ToPolygonLayerCommand, ToVerdatLayerCommand, MergeLayersCommand, PasteLayerCommand, StartTimeCommand, EndTimeCommand, PolygonEditLayerCommand, PolygonSaveEditLayerCommand, DeletePolygonCommand
from .mouse_commands import ViewportCommand, NormalizeLongitudeCommand, SwapLatLonCommand
from .ui.dialogs import StyleDialog, prompt_for_wms, prompt_for_tile
from .library.thread_utils import BackgroundWMSDownloader
from .library.tile_utils import BackgroundTileDownloader
from . import layers

import logging
log = logging.getLogger(__name__)


class LayerAction(EditorAction):
    """Superclass for actions that operate on layers.

    Provides a common framework for usage in menubars and popup menus
    """
    def _update_popup_enabled(self, ui_state, popup_data):
        layer = popup_data['layer']
        self.enabled = self.is_popup_enabled(ui_state, layer)

    def is_popup_enabled(self, ui_state, layer):
        if self.enabled_name:
            return getattr(ui_state, self.enabled_name)
        return True

    def get_layer(self, event):
        if hasattr(event, 'popup_data'):
            return event.popup_data['layer']
        return self.active_editor.layer_tree_control.get_edit_layer()

    def perform(self, event):
        layer = self.get_layer(event)
        if layer is not None:
            self.perform_on_layer(layer, event)

    def perform_on_layer(self, layer, event):
        log.warning("Missing perform_on_layer method for %s" % self.name)


class NewProjectAction(Action):
    """ An action for creating a new empty file that can be edited by a particular task
    """
    name = 'New Default Project'
    tooltip = 'Open a new copy of the default project'

    def perform(self, event=None):
        task = event.task.window.application.find_or_create_task_of_type(pane_layout.task_id_with_pane_layout)
        wx.CallAfter(event.task.window.application.load_file, task.about_application, task)


class NewEmptyProjectAction(Action):
    """ An action for creating a new empty file that can be edited by a particular task
    """
    name = 'New Empty Project'
    tooltip = 'Open an empty grid to create new layers'

    def perform(self, event=None):
        task = event.task.window.application.find_or_create_task_of_type(pane_layout.task_id_with_pane_layout)
        wx.CallAfter(task.new)


class SaveProjectAction(EditorAction):
    name = 'Save Project'
    accelerator = 'Ctrl+S'
    tooltip = 'Save the current project'
    image = ImageResource('file_save')
    enabled_name = 'can_save'  # enabled based on state of task.active_editor.dirty

    def perform(self, event):
        self.active_editor.save()


class SaveProjectAsAction(EditorAction):
    name = 'Save Project As...'
    accelerator = 'Ctrl+Shift+S'
    tooltip = 'Save the current project with a new name'
    image = ImageResource('file_save_as')

    def perform(self, event):
        self.active_editor.save(prompt=True)


class SaveProjectTemplateAction(EditorAction):
    name = 'Save Project as Template...'
    tooltip = 'Save the current project as a template that can be reused'
    image = ImageResource('file_save_as')

    def perform(self, event):
        name = event.task.prompt("Enter name for this template", "Save Project as Template", self.active_editor.layer_manager.root_name)
        if name is not None:
            self.active_editor.save_as_template(name)


class LoadProjectTemplateGroup(TaskDynamicSubmenuGroup):
    id = 'LoadProjectTemplateGroup'

    event_name = Str('templates_changed')

    def _get_items(self, layer=None):
        items = []
        names = self.task.window.application.get_available_user_data('project_templates')
        print(names)
        for name in sorted(names):
            action = LoadProjectTemplateAction(name=name)
            items.append(ActionItem(action=action))
        return items


class LoadProjectTemplateAction(Action):
    name = '<load template>'
    tooltip = 'Open project template'

    def perform(self, event=None):
        uri = event.task.window.application.get_user_dir_filename('project_templates', self.name)
        wx.CallAfter(event.task.window.application.load_file, uri, event.task)


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
    ext = Str

    def _name_default(self):
        return "%s (%s)" % (self.loader.extension_name(self.ext), self.ext)

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
            from .layers.loaders import valid_save_formats
            valid = valid_save_formats(layer)
            if valid:
                for item in valid:
                    loader = item[0]
                    for ext in loader.extensions:
                        action = SaveLayerAsFormatAction(loader=loader, ext=ext)
                        items.append(ActionItem(action=action))

        return items


class SaveMovieAction(EditorAction):
    name = 'Save Latest Playback...'
    loader = Any
    enabled_name = 'latest_movie'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as', wildcard="PNG Movies (*.png)|*.png")
        if dialog.open() == OK:
            self.active_editor.save_latest_movie(dialog.path)


class RevertProjectAction(EditorAction):
    name = 'Revert Project'
    tooltip = 'Revert project to last saved version'
    enabled_name = 'document.can_revert'

    def perform(self, event):
        message = "Revert project from\n\n%s?" % self.active_editor.document.metadata.uri
        if event.task.confirm(message=message, title='Revert Project?'):
            self.active_editor.load_omnivore_document(self.active_editor.document)


class DefaultStyleAction(EditorAction):
    name = 'Default Styles...'
    tooltip = 'Choose the line, fill and font styles'

    def perform(self, event):
        GUI.invoke_later(self.show_dialog, self.active_editor)

    def show_dialog(self, project):
        dialog = StyleDialog(project, layers.styleable_layers)
        status = dialog.ShowModal()
        if status == wx.ID_OK:
            project.layer_manager.update_default_styles(dialog.get_styles())
            if dialog.save_for_future:
                project.task.remember_styles(project.layer_manager.default_styles)
            if dialog.apply_to_current:
                project.layer_manager.apply_default_styles()


class BoundingBoxAction(EditorAction):
    name = 'Show Bounding Boxes'
    tooltip = 'Display or hide bounding boxes for each layer'
    style = 'toggle'

    def perform(self, event):
        value = not self.active_editor.layer_canvas.debug_show_bounding_boxes
        self.active_editor.layer_canvas.debug_show_bounding_boxes = value
        GUI.invoke_later(self.active_editor.layer_canvas.render)

    @on_trait_change('active_editor')
    def _update_checked(self, ui_state):
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
    def _update_checked(self, ui_state):
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


class ZoomToLayer(LayerAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'
    enabled_name = 'layer_zoomable'
    image = ImageResource('zoom_to_layer')

    def is_popup_enabled(self, ui_state, layer):
        return layer.is_zoomable()

    def perform_on_layer(self, layer, event):
        cmd = ViewportCommand(layer)
        self.active_editor.process_command(cmd)


class NewLayerBaseAction(EditorAction):
    layer_class = None

    def perform(self, event):
        cmd = AddLayerCommand(self.layer_class)
        self.active_editor.process_command(cmd)

class NewVectorLayerAction(NewLayerBaseAction):
    name = 'New Verdat Layer'
    tooltip = 'Create new vector (grid) layer'
    image = ImageResource('add_layer')
    layer_class = layers.LineLayer


class NewLonLatLayerAction(NewLayerBaseAction):
    name = 'New Graticule Layer'
    tooltip = 'Create new longitude/latitude grid layer'
    layer_class = layers.Graticule


class NewCompassRoseLayerAction(NewLayerBaseAction):
    name = 'New Compass Rose Layer'
    tooltip = 'Create new compass rose or north-up arrow layer'
    layer_class = layers.CompassRose


class NewTimestampLayerAction(NewLayerBaseAction):
    name = 'New Timestamp Layer'
    tooltip = 'Create new timestamp to display current time in playback'
    layer_class = layers.Timestamp


class NewAnnotationLayerAction(NewLayerBaseAction):
    name = 'New Annotation Layer'
    tooltip = 'Create new annotation layer'
    layer_class = layers.AnnotationLayer


class NewWMSLayerAction(NewLayerBaseAction):
    name = 'New WMS Layer'
    tooltip = 'Create new Web Map Service layer'
    layer_class = layers.WMSLayer


class NewTileLayerAction(NewLayerBaseAction):
    name = 'New Tile Layer'
    tooltip = 'Create new tile background service layer'
    layer_class = layers.TileLayer


class NewRNCLayerAction(EditorAction):
    name = 'New RNC Download Selection Layer (-360 - 0)'
    tooltip = 'Create new layer for downloading RNC images in the -360 to 0 map regime'

    def perform(self, event):
        from maproom.templates import get_template_path
        path = get_template_path("RNCProdCat_*.bna")
        event.task.window.application.load_file(path, event.task, regime=0)


class NewRNCLayer360Action(EditorAction):
    name = 'New RNC Download Selection Layer (0 - 360)'
    tooltip = 'Create new layer for downloading RNC images in the 0 to 360 map regime'

    def perform(self, event):
        from maproom.templates import get_template_path
        path = get_template_path("RNCProdCat_*.bna")
        event.task.window.application.load_file(path, event.task, regime=360)


class DeleteLayerAction(LayerAction):
    name = 'Delete Layer'
    tooltip = 'Remove the layer from the project'
    enabled_name = 'layer_selected'
    image = ImageResource('delete_layer')

    def is_popup_enabled(self, ui_state, layer):
        return True  # layer may not be selected using context menu

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.delete_selected_layer, layer)


class RaiseLayerAction(LayerAction):
    name = 'Raise Layer'
    tooltip = 'Move layer up in the stacking order'
    enabled_name = 'layer_above'
    image = ImageResource('raise.png')

    def is_popup_enabled(self, ui_state, layer):
        return ui_state.layer_manager.is_raisable(layer)

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_selected_layer, layer)


class RaiseToTopAction(LayerAction):
    name = 'Raise Layer To Top'
    tooltip = 'Move layer to the top'
    enabled_name = 'layer_above'
    image = ImageResource('raise_to_top.png')

    def is_popup_enabled(self, ui_state, layer):
        return ui_state.layer_manager.is_raisable(layer)

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_to_top, layer)


class LowerToBottomAction(LayerAction):
    name = 'Lower Layer To Bottom'
    tooltip = 'Move layer to the bottom'
    enabled_name = 'layer_below'
    image = ImageResource('lower_to_bottom.png')

    def is_popup_enabled(self, ui_state, layer):
        return ui_state.layer_manager.is_lowerable(layer)

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.lower_to_bottom, layer)


class LowerLayerAction(LayerAction):
    name = 'Lower Layer'
    tooltip = 'Move layer down in the stacking order'
    enabled_name = 'layer_below'
    image = ImageResource('lower.png')

    def is_popup_enabled(self, ui_state, layer):
        return ui_state.layer_manager.is_lowerable(layer)

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.lower_selected_layer, layer)


class TriangulateLayerAction(EditorAction):
    name = 'Triangulate Layer'
    tooltip = 'Create triangular mesh'
    enabled_name = 'layer_has_points'
    image = ImageResource('triangulate.png')

    def perform(self, event):
        e = self.active_editor
        e.control.force_focus(e.triangle_panel)


class ToPolygonLayerAction(EditorAction):
    name = 'Convert to Polygon Layer'
    tooltip = 'Create new polygon layer from boundaries of current layer'
    enabled_name = 'layer_has_boundaries'

    def perform(self, event):
        edit_layer = self.active_editor.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = ToPolygonLayerCommand(edit_layer)
            self.active_editor.process_command(cmd)


class ToVerdatLayerAction(EditorAction):
    name = 'Convert to Editable Layer'
    tooltip = 'Create new editable layer from rings of current layer'
    enabled_name = 'layer_has_boundaries'

    def perform(self, event):
        edit_layer = self.active_editor.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = ToVerdatLayerCommand(edit_layer)
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
            "Please select two vector layers to merge together into one layer.\n\nOnly those layers that support merging are listed.",
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
        else:
            selections = []
        dialog.Destroy()
        if len(selections) != 2:
            project.window.error("You must select exactly two layers to merge.")
        else:
            layer_a = layers[selections[0]]
            layer_b = layers[selections[1]]
            if hasattr(layer_a, "depth_unit") and hasattr(layer_b, "depth_unit"):
                da = layer_a.depth_unit
                db = layer_b.depth_unit
                if da != db:
                    dialog = wx.SingleChoiceDialog(project.window.control, "Choose units for merged layer", "Depth Units", [da, db])
                    result = dialog.ShowModal()
                    if result == wx.ID_OK:
                        depth_unit = dialog.GetStringSelection()
                    else:
                        depth_unit = None
                    dialog.Destroy()
                else:
                    depth_unit = da
            if depth_unit is not None:
                cmd = MergeLayersCommand(layer_a, layer_b, depth_unit)
                project.process_command(cmd)


class MergePointsAction(EditorAction):
    name = 'Merge Duplicate Points'
    tooltip = 'Merge points within a layer'
    enabled_name = 'layer_has_points'
    image = ImageResource('merge_duplicates.png')

    def perform(self, event):
        e = self.active_editor
        e.control.force_focus(e.merge_points_panel)


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
        # FIXME: OS X hack! DELETE key in the menu overrides any text field
        # usage, so we have to catch it here and force the textctrl to do the
        # delete programmatically
        active = wx.Window.FindFocus()
        if sys.platform == "darwin" and hasattr(active, "EmulateKeyPress"):
            # EmulateKeyPress on wx.TextCtrl requires an actual KeyEvent
            # which I haven't been able to create. Workaround: simulate what
            # the delete key should do
            start, end = active.GetSelection()
            if start == end:
                active.Remove(start, end + 1)
            else:
                active.Remove(start, end)
        else:
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


class CheckSelectedLayerAction(LayerAction):
    name = 'Check Layer For Errors'
    accelerator = 'Ctrl+E'
    enabled_name = 'layer_selected'
    tooltip = 'Check for valid layer construction'

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.check_for_errors, layer)


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
        from . import debug
        lm = project.layer_manager
        undo = debug.debug_objects(lm)
        project.process_flags(undo.flags)
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


class DuplicateLayerAction(LayerAction):
    name = 'Duplicate Layer'
    accelerator = 'Ctrl+D'
    tooltip = 'Duplicate the current layer'
    enabled_name = 'can_copy'

    def is_popup_enabled(self, ui_state, layer):
        return hasattr(layer, "center_point_index")  # only vector layers

    def perform_on_layer(self, layer, event):
        json_data = layer.serialize_json(-999, True)
        if json_data:
            text = json.dumps(json_data)
            cmd = PasteLayerCommand(layer, text, self.active_editor.layer_canvas.world_center)
            self.active_editor.process_command(cmd)


class ManageWMSAction(EditorAction):
    name = 'Manage WMS Servers...'

    def perform(self, event):
        hosts = BackgroundWMSDownloader.get_known_hosts()
        dlg = ListReorderDialog(event.task.window.control, hosts, lambda a: a.label_helper(), prompt_for_wms, "Manage WMS Servers", default_helper=lambda a,v: a.default_helper(v))
        if dlg.ShowModal() == wx.ID_OK:
            items = dlg.get_items()
            BackgroundWMSDownloader.set_known_hosts(items)
            event.task.remember_wms()
            self.active_editor.layer_manager.update_map_server_ids("wms", hosts, items)
            self.active_editor.refresh()
        dlg.Destroy()


class ManageTileServersAction(EditorAction):
    name = 'Manage Tile Servers...'

    def perform(self, event):
        hosts = BackgroundTileDownloader.get_known_hosts()
        dlg = ListReorderDialog(event.task.window.control, hosts, lambda a: a.label_helper(), prompt_for_tile, "Manage Tile Servers", default_helper=lambda a,v: a.default_helper(v))
        if dlg.ShowModal() == wx.ID_OK:
            items = dlg.get_items()
            BackgroundTileDownloader.set_known_hosts(items)
            event.task.remember_tile_servers()
            self.active_editor.layer_manager.update_map_server_ids("tiles", hosts, items)
            self.active_editor.refresh()
        dlg.Destroy()


class ClearTileCacheAction(EditorAction):
    name = 'Clear Tile Cache...'

    def perform(self, event):
        hosts = BackgroundTileDownloader.get_known_hosts()
        dlg = CheckItemDialog(event.task.window.control, hosts, lambda a: getattr(a, 'name'), title="Clear Tile Cache", instructions="Clear cache of selected tile servers:")
        if dlg.ShowModal() == wx.ID_OK:
            try:
                for host in dlg.get_checked_items():
                    host.clear_cache(event.task.get_tile_cache_root())
            except OSError as e:
                event.task.window.window.error("Error clearing cache for %s\n\n%s" % (host.name, str(e)))

        dlg.Destroy()


class NormalizeLongitudeAction(EditorAction):
    name = 'Normalize Longitude'
    tooltip = 'Adjust longitudes so the map lies between 0 and 360W'

    def perform(self, event):
        edit_layer = self.active_editor.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = NormalizeLongitudeCommand(edit_layer)
            self.active_editor.process_command(cmd)


class SwapLatLonAction(EditorAction):
    name = 'Swap Lat && Lon'
    tooltip = 'Exchange coordinate pairs to repair an incorrectly formatted input file'

    def perform(self, event):
        edit_layer = self.active_editor.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = SwapLatLonCommand(edit_layer)
            self.active_editor.process_command(cmd)


class DebugLayerManagerAction(EditorAction):
    name = 'Show Layer Manager Info'
    tooltip = 'Show a debug output describing the currently displayed layers'

    def perform(self, event):
        lm = self.active_editor.layer_manager
        text = lm.debug_structure()
        print(text)



class GroupLayerAction(LayerAction):
    name = 'Group Sublayers'
    tooltip = 'Group all children of the selected layer into a single unit'
    enabled_name = 'layer_is_groupable'
    image = ImageResource('shape_group.png')

    def is_popup_enabled(self, ui_state, layer):
        return layer.has_groupable_objects() and not layer.grouped

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.group_children, layer)


class UngroupLayerAction(LayerAction):
    name = 'Ungroup Into Sublayers'
    tooltip = 'Remove grouping and display child layers'
    enabled_name = 'layer_is_groupable'
    image = ImageResource('shape_ungroup.png')

    def is_popup_enabled(self, ui_state, layer):
        return layer.has_groupable_objects() and layer.grouped

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.ungroup_children, layer)


class RenameLayerAction(LayerAction):
    name = 'Rename Layer'
    tooltip = 'Rename layer'

    def perform_on_layer(self, layer, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.start_rename, layer)


class EditLayerAction(EditorAction):
    name = 'Edit Layer'
    tooltip = 'Edit the currently selected layer'

    def perform(self, event):
        d = event.popup_data
        cmd = PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'])
        self.active_editor.process_command(cmd)


class AddPolygonBoundaryAction(EditorAction):
    name = 'Add Polygon'
    tooltip = 'Add a new boundary polygon'

    def perform(self, event):
        d = event.popup_data
        cmd = PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'], new_boundary=True)
        self.active_editor.process_command(cmd)


class AddPolygonHoleAction(EditorAction):
    name = 'Add Hole'
    tooltip = 'Add a new hole polygon'

    def perform(self, event):
        d = event.popup_data
        cmd = PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'], new_hole=True)
        self.active_editor.process_command(cmd)


class DeletePolygonAction(EditorAction):
    name = 'Delete Polygon'
    tooltip = 'Remove a polygon or hole; note that if a polygon is removed, any holes in that polygon are also removed'

    def perform(self, event):
        d = event.popup_data
        cmd = DeletePolygonCommand(d['layer'], d['object_type'], d['object_index'])
        self.active_editor.process_command(cmd)


class SaveRingEditAction(EditorAction):
    name = 'Save Changes in Polygon'
    tooltip = 'Save the current edits in the parent polygon'

    def perform(self, event):
        d = event.popup_data
        cmd = PolygonSaveEditLayerCommand(d['layer'])
        self.active_editor.process_command(cmd)


class StartTimeAction(LayerAction):
    name = 'Start Time'
    tooltip = 'Set time that layer becomes active'
    dialog_info = 'Set start time of %s\nto start time of:'
    cmd = StartTimeCommand

    def get_time(self, layer):
        return layer.start_time

    def perform_on_layer(self, layer, event):
        layers = self.active_editor.layer_manager.get_playback_layers(layer)
        dlg = wx.SingleChoiceDialog(
                self.active_editor.layer_tree_control, self.dialog_info % layer.name, self.name,
                [a.name for a in layers],
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            source_layer = layers[dlg.GetSelection()]
            print(('You selected: %s\n' % source_layer))
            cmd = self.cmd(layer, self.get_time(source_layer))
            self.active_editor.process_command(cmd)


class EndTimeAction(StartTimeAction):
    name = 'End Time'
    tooltip = 'Set time that layer stops being active'
    dialog_info = 'Set end time of %s\nto end time of:'
    cmd = EndTimeCommand

    def get_time(self, layer):
        return layer.end_time
