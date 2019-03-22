import sys
import wx
import json

# Enthought library imports.
from traits.api import Any
from traits.api import Str
from traits.api import on_trait_change

from sawx import persistence
from sawx.action import SawxAction, SawxListAction
from sawx.ui.dialogs import ListReorderDialog, CheckItemDialog

from . import pane_layout
from . import menu_commands as mec
from . import mouse_commands as moc
from .ui.dialogs import StyleDialog, prompt_for_wms, prompt_for_tile, SimplifyDialog
from .library.thread_utils import BackgroundWMSDownloader
from .library.tile_utils import BackgroundTileDownloader
from . import layers

import logging
log = logging.getLogger(__name__)


class LayerAction(SawxAction):
    """Superclass for actions that operate on layers.

    Provides a common framework for usage in menubars and popup menus
    """
    def calc_enabled(self, action_key):
        layer = self.get_layer(action_key)
        return self.is_enabled_for_layer(action_key, layer)

    def is_enabled_for_layer(self, action_key, layer):
        return True

    def get_layer(self, action_key):
        return self.editor.layer_tree_control.get_edit_layer()

    def perform(self, action_key):
        layer = self.get_layer(action_key)
        if layer is not None:
            self.perform_on_layer(layer, action_key)

    def perform_on_layer(self, action_key, layer):
        log.warning("Missing perform_on_layer method for %s" % self.name)


class new_project(SawxAction):
    """ An action for creating a new empty file that can be edited by a particular task
    """
    name = 'New Default Project'
    tooltip = 'Open a new copy of the default project'

    template = None

    def perform(self, action_key):
        template = self.template if self.template is not None else wx.GetApp().default_uri
        self.editor.frame.load_file(template, self.editor)


class new_empty_project(new_project):
    """ An action for creating a new empty file that can be edited by a particular task
    """
    name = 'New Empty Project'
    tooltip = 'Open an empty grid to create new layers'

    template = "template://blank_project.maproom"


class save_project(SawxAction):
    name = 'Save Project'
    tooltip = 'Save the current project'

    def calc_icon_name(self, action_key):
        return "save_file"

    def calc_enabled(self, action_key):
        return self.editor.is_dirty

    def perform(self, action_key):
        self.editor.save()


class save_project_as(SawxAction):
    name = 'Save Project As...'
    tooltip = 'Save the current project with a new name'

    def calc_icon_name(self, action_key):
        return "save_file_as"

    def perform(self, action_key):
        self.editor.save_as()


class save_project_template(SawxAction):
    name = 'Save Project as Template...'
    tooltip = 'Save the current project as a template that can be reused'

    def calc_icon_name(self, action_key):
        return 'file_save_as'

    def perform(self, action_key):
        name = self.editor.frame.prompt("Enter name for this template", "Save Project as Template", self.editor.layer_manager.root_name)
        if name is not None:
            self.editor.save_as_template(name)


class load_project_template(SawxListAction):
    tooltip = 'Open project template'

    def calc_list_items(self):
        items = persistence.get_available_user_data('project_templates')
        return items.sorted()

    def perform(self, action_key):
        name = self.get_item(action_key)
        uri = persistence.get_user_dir_filename('project_templates', name)
        self.editor.frame.load_file(uri, event.task)


class save_command_log(SawxAction):
    name = 'Save Command Log...'
    tooltip = 'Save a copy of the command log'

    def perform(self, action_key):
        path = self.editor.frame.prompt_local_file_dialog("Save Command Log", save=True, wildcard="MapRoom Command Log Files (*.mrc)|*.mrc")
        if path:
            self.editor.save_log(dialog.path)


class save_layer(SawxAction):
    name = 'Save Layer'
    tooltip = 'Save the currently selected layer'

    def calc_enabled(self, action_key):
        return self.editor.current_layer.can_save()

    def perform(self, action_key):
        self.editor.save_layer(None)


class save_layer_as(SawxAction):
    name = 'Save Layer As...'
    tooltip = 'Save the current project with a new name'

    def calc_enabled(self, action_key):
        return self.editor.current_layer.can_save_as()

    def perform(self, action_key):
        path = self.editor.frame.prompt_local_file_dialog("Save Layer", save=True)
        if path:
            self.editor.save_layer(path)


class LoaderForExt:
    def __init__(self, loader, ext):
        self.loader = loader
        self.ext = ext

    def __str__(self):
        return "%s (%s)" % (self.loader.extension_name(self.ext), self.ext)


class save_layer_as(SawxListAction):
    """ A menu for changing the active task in a task window.
    """

    def calc_items(self, layer=None):
        items = []
        if layer is not None:
            from .loaders import valid_save_formats
            valid = valid_save_formats(layer)
            if valid:
                for item in valid:
                    loader = item[0]
                    for ext in loader.extensions:
                        items.append(LoaderForExt(loader, ext))
        return items

    def perform(self, action_key):
        item = self.get_item(action_key)
        path = self.editor.frame.prompt_local_file_dialog("Save Layer", save=True, wildcard=item.loader.get_file_dialog_wildcard())
        if path:
            self.editor.save_layer(path)


class save_movie(SawxAction):
    name = 'Save Latest Playback...'
    loader = Any

    def calc_enabled(self, action_key):
        return bool(self.editor.latest_movie)

    def perform(self, action_key):
        dialog = FileDialog(parent=event.task.window.control, action='save as', wildcard="PNG Movies (*.png)|*.png")
        if dialog.open() == OK:
            self.editor.save_latest_movie(dialog.path)


class revert_project(SawxAction):
    name = 'Revert Project'
    tooltip = 'Revert project to last saved version'

    def calc_enabled(self, action_key):
        return self.editor.document.can_revert

    def perform(self, action_key):
        message = "Revert project from\n\n%s?" % self.editor.document.metadata.uri
        if event.task.confirm(message=message, title='Revert Project?'):
            self.editor.load_omnivore_document(self.editor.document)


class default_style(SawxAction):
    name = 'Default Styles...'
    tooltip = 'Choose the line, fill and font styles'

    def perform(self, action_key):
        self.show_dialog(self.editor)

    def show_dialog(self, project):
        dialog = StyleDialog(project, layers.styleable_layers)
        status = dialog.ShowModal()
        if status == wx.ID_OK:
            project.layer_manager.update_default_styles(dialog.get_styles())
            if dialog.save_for_future:
                style.remember_styles(project.layer_manager.default_styles)
            if dialog.apply_to_current:
                project.layer_manager.apply_default_styles()


class bounding_box(SawxAction):
    name = 'Show Bounding Boxes'
    tooltip = 'Display or hide bounding boxes for each layer'
    style = 'toggle'

    def perform(self, action_key):
        value = not self.editor.layer_canvas.debug_show_bounding_boxes
        self.editor.layer_canvas.debug_show_bounding_boxes = value
        self.editor.layer_canvas.render()

    @on_trait_change('active_editor')
    def _update_checked(self, ui_state):
        if self.editor:
            self.checked = self.editor.layer_canvas.debug_show_bounding_boxes


class picker_framebuffer(SawxAction):
    name = 'Show Picker Framebuffer'
    tooltip = 'Display the picker framebuffer instead of the normal view'
    style = 'toggle'

    def perform(self, action_key):
        value = not self.editor.layer_canvas.debug_show_picker_framebuffer
        self.editor.layer_canvas.debug_show_picker_framebuffer = value
        self.editor.layer_canvas.render()

    @on_trait_change('active_editor')
    def _update_checked(self, ui_state):
        if self.editor:
            self.checked = self.editor.layer_canvas.debug_show_picker_framebuffer


class zoom_in(SawxAction):
    name = 'Zoom In'
    tooltip = 'Increase magnification'

    def perform(self, action_key):
        c = self.editor.layer_canvas
        units_per_pixel = c.zoom_in()
        cmd = moc.ViewportCommand(None, c.projected_point_center, units_per_pixel)
        self.editor.process_command(cmd)


class zoom_out(SawxAction):
    name = 'Zoom Out'
    tooltip = 'Decrease magnification'

    def perform(self, action_key):
        c = self.editor.layer_canvas
        units_per_pixel = c.zoom_out()
        cmd = moc.ViewportCommand(None, c.projected_point_center, units_per_pixel)
        self.editor.process_command(cmd)


class zoom_to_fit(SawxAction):
    name = 'Zoom to Fit'
    tooltip = 'Set magnification to show all layers'

    def perform(self, action_key):
        c = self.editor.layer_canvas
        center, units_per_pixel = c.calc_zoom_to_fit()
        cmd = moc.ViewportCommand(None, center, units_per_pixel)
        self.editor.process_command(cmd)


class zoom_to_layer(LayerAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.is_zoomable()

    def perform_on_layer(self, action_key, layer):
        cmd = moc.ViewportCommand(layer)
        self.editor.process_command(cmd)


class NewLayerBaseAction(SawxAction):
    layer_class = None

    def perform(self, action_key):
        cmd = mec.AddLayerCommand(self.layer_class)
        self.editor.process_command(cmd)

class new_vector_layer(NewLayerBaseAction):
    name = 'New Verdat Layer'
    tooltip = 'Create new vector (grid) layer'
#    image = ImageResource('add_layer')
    layer_class = layers.LineLayer


class new_lon_lat_layer(NewLayerBaseAction):
    name = 'New Graticule Layer'
    tooltip = 'Create new longitude/latitude grid layer'
    layer_class = layers.Graticule


class new_noaa_logo_layer(NewLayerBaseAction):
    name = 'New NOAA Logo Layer'
    tooltip = 'Create new NOAA logo overlay layer'
    layer_class = layers.NOAALogo


class new_compass_rose_layer(NewLayerBaseAction):
    name = 'New Compass Rose Layer'
    tooltip = 'Create new compass rose or north-up arrow layer'
    layer_class = layers.CompassRose


class new_timestamp_layer(NewLayerBaseAction):
    name = 'New Timestamp Layer'
    tooltip = 'Create new timestamp to display current time in playback'
    layer_class = layers.Timestamp


class new_annotation_layer(NewLayerBaseAction):
    name = 'New Annotation Layer'
    tooltip = 'Create new annotation layer'
    layer_class = layers.AnnotationLayer


class new_wms_layer(NewLayerBaseAction):
    name = 'New WMS Layer'
    tooltip = 'Create new Web Map Service layer'
    layer_class = layers.WMSLayer


class new_tile_layer(NewLayerBaseAction):
    name = 'New Tile Layer'
    tooltip = 'Create new tile background service layer'
    layer_class = layers.TileLayer


class new_shapefile_layer(NewLayerBaseAction):
    name = 'New Shapefile/Polygon Layer'
    tooltip = 'Create new layer of polygons'
    layer_class = layers.PolygonParentLayer


class new_rnc_layer(SawxAction):
    name = 'New RNC Download Selection Layer (-360 - 0)'
    tooltip = 'Create new layer for downloading RNC images in the -360 to 0 map regime'

    def perform(self, action_key):
        from maproom.templates import get_template_path
        path = get_template_path("RNCProdCat_*.bna")
        self.editor.frame.load_file(path, event.task, regime=0)


class new_rnc_layer360(SawxAction):
    name = 'New RNC Download Selection Layer (0 - 360)'
    tooltip = 'Create new layer for downloading RNC images in the 0 to 360 map regime'

    def perform(self, action_key):
        from maproom.templates import get_template_path
        path = get_template_path("RNCProdCat_*.bna")
        self.editor.frame.load_file(path, event.task, regime=360)


class delete_layer(LayerAction):
    name = 'Delete Layer'
    tooltip = 'Remove the layer from the project'

    def is_enabled_for_layer(self, action_key, layer):
        return not layer.is_root()

    def perform_on_layer(self, action_key, layer):
        self.editor.delete_selected_layer(layer)


class raise_layer(LayerAction):
    name = 'Raise Layer'
    tooltip = 'Move layer up in the stacking order'

    def is_enabled_for_layer(self, action_key, layer):
        return self.editor.layer_manager.is_raisable(layer)

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.raise_selected_layer(layer)


class raise_to_top(LayerAction):
    name = 'Raise Layer To Top'
    tooltip = 'Move layer to the top'

    def is_enabled_for_layer(self, action_key, layer):
        return self.editor.layer_manager.is_raisable(layer)

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.raise_to_top(layer)


class lower_to_bottom(LayerAction):
    name = 'Lower Layer To Bottom'
    tooltip = 'Move layer to the bottom'

    def is_enabled_for_layer(self, action_key, layer):
        return self.editor.layer_manager.is_lowerable(layer)

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.lower_to_bottom(layer)


class lower_layer(LayerAction):
    name = 'Lower Layer'
    tooltip = 'Move layer down in the stacking order'

    def is_enabled_for_layer(self, action_key, layer):
        return self.editor.layer_manager.is_lowerable(layer)

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.lower_selected_layer(layer)


class triangulate_layer(LayerAction):
    name = 'Triangulate Layer'
    tooltip = 'Create triangular mesh'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_points()

    def perform_on_layer(self, action_key, layer):
        e = self.editor
        e.control.force_focus(e.triangle_panel)


class to_polygon_layer(LayerAction):
    name = 'Convert to Polygon Layer'
    tooltip = 'Create new polygon layer from boundaries of current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_boundaries()

    def perform_on_layer(self, action_key, layer):
        cmd = mec.ToPolygonLayerCommand(layer)
        self.editor.process_command(cmd)


class to_verdat_layer(LayerAction):
    name = 'Convert to Editable Layer'
    tooltip = 'Create new editable layer from rings of current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_boundaries()

    def perform_on_layer(self, action_key, layer):
        cmd = mec.ToVerdatLayerCommand(layer)
        self.editor.process_command(cmd)


class convex_hull(SawxAction):
    name = 'Convex Hull'
    tooltip = 'Create convex hull of a set of points'

    def calc_enabled(self, action_key):
        return self.editor.layer_manager.count_layers() > 1

    def perform(self, action_key):
        self.show_dialog(self.editor)

    def show_dialog(self, project):
        layers = project.layer_manager.flatten()

        if len(layers) < 1:
            project.task.error("Convex hull requires at least one layer.")
            return

        layer_names = [str(layer.name) for layer in layers]

        import wx
        dialog = wx.MultiChoiceDialog(
            project.window.control,
            "Please select at least one layer.",
            "Point Layers",
            layer_names
        )

        result = dialog.ShowModal()
        if result == wx.ID_OK:
            selections = dialog.GetSelections()
        else:
            selections = []
        dialog.Destroy()
        selected_layers = [layers[s] for s in selections]
        if len(selected_layers) < 1:
            project.task.error("You must select a layer.")
        else:
            cmd = mec.ConvexHullCommand(selected_layers)
            project.process_command(cmd)


class merge_layers(SawxAction):
    name = 'Merge Layers'
    tooltip = 'Merge two vector layers'

    def calc_enabled(self, action_key):
        return self.editor.layer_manager.count_layers() > 1

    def perform(self, action_key):
        self.show_dialog(self.editor)

    def show_dialog(self, project):
        layers = project.layer_manager.get_mergeable_layers()

        if len(layers) < 2:
            project.task.error("Merge requires two vector layers.")
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
            project.task.error("You must select exactly two layers to merge.")
        else:
            layer_a = layers[selections[0]]
            layer_b = layers[selections[1]]
            depth_unit = None
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
                cmd = mec.MergeLayersCommand(layer_a, layer_b, depth_unit)
                project.process_command(cmd)


class merge_points(LayerAction):
    name = 'Merge Duplicate Points'
    tooltip = 'Merge points within a layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_points()

    def perform_on_layer(self, action_key, layer):
        e = self.editor
        e.control.force_focus(e.merge_points_panel)


class jump_to_coords(SawxAction):
    name = 'Jump to Coordinates'
    
    tooltip = 'Center the screen on the specified coordinates'

    def perform(self, action_key):
        self.editor.layer_canvas.do_jump_coords()


class clear_selection(LayerAction):
    name = 'Clear Selection'
    tooltip = 'Deselects all selected items in the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_selection()

    def perform_on_layer(self, action_key, layer):
        self.editor.clear_selection()


class delete_selection(LayerAction):
    name = 'Delete Selection'
    tooltip = 'Deletes the selected items in the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_selection()

    def perform_on_layer(self, action_key, layer):
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
            self.editor.delete_selection()


class clear_flagged(SawxAction):
    name = 'Clear Flagged'
    tooltip = 'Deselects all flagged items in the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_flagged()

    def perform_on_layer(self, action_key, layer):
        self.editor.clear_all_flagged()


class flagged_to_selection(SawxAction):
    name = 'Select Flagged'
    tooltip = 'Select all flagged items in the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_flagged()

    def perform_on_layer(self, action_key, layer):
        self.editor.select_all_flagged()


class boundary_to_selection(LayerAction):
    name = 'Select Boundary'
    tooltip = 'Select the boundary of the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_points()

    def perform_on_layer(self, action_key, layer):
        self.editor.select_boundary()


class find_points(LayerAction):
    name = 'Find Points'
    tooltip = 'Find and highlight points or ranges of points'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_points()

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_canvas.do_find_points()


class check_selected_layer(LayerAction):
    name = 'Check Layer For Errors'
    tooltip = 'Check for valid layer construction'

    def perform_on_layer(self, action_key, layer):
        self.editor.check_for_errors(layer)


class check_all_layers(SawxAction):
    name = 'Check All Layers For Errors'
    tooltip = 'Check for valid layer construction'

    def perform(self, action_key):
        self.editor.check_all_layers_for_errors()


class open_log(SawxAction):
    name = 'View Command History Log'
    tooltip = 'View the log containing a list of all user commands'

    def perform(self, action_key):
        filename = persistence.get_log_file_name("command_log", ".mrc")
        self.editor.frame.load_file(filename)


class debug_annotation_layers(SawxAction):
    name = 'Sample Annotation Layer'
    tooltip = 'Create a sample annotation layer with examples of all vector objects'

    def perform(self, action_key):
        from . import debug
        lm = project.layer_manager
        undo = debug.debug_objects(lm)
        project.process_flags(undo.flags)
        project.update_default_visibility()
        project.layer_metadata_changed(None)
        project.layer_canvas.zoom_to_fit()


class copy_style(SawxAction):
    name = 'Copy Style'
    tooltip = 'Copy the style of the current selection'

    def calc_enabled(self, action_key):
        return self.editor.can_copy

    def perform(self, action_key):
        self.editor.copy_style()


class paste_style(SawxAction):
    name = 'Paste Style'
    tooltip = 'Apply the style from the clipboard'

    def calc_enabled(self, action_key):
        return self.editor.clipboard_style is not None

    def perform(self, action_key):
        self.editor.paste_style()


class duplicate_layer(LayerAction):
    name = 'Duplicate Layer'
    tooltip = 'Duplicate the current layer'

    def is_enabled_for_layer(self, action_key, layer):
        return hasattr(layer, "center_point_index")  # only vector layers

    def perform_on_layer(self, action_key, layer):
        json_data = layer.serialize_json(-999, True)
        if json_data:
            text = json.dumps(json_data)
            cmd = mec.PasteLayerCommand(layer, text, self.editor.layer_canvas.world_center)
            self.editor.process_command(cmd)


class manage_wms_servers(SawxAction):
    name = 'Manage WMS Servers...'

    def perform(self, action_key):
        hosts = BackgroundWMSDownloader.get_known_hosts()
        dlg = ListReorderDialog(event.task.window.control, hosts, lambda a: a.label_helper(), prompt_for_wms, "Manage WMS Servers", default_helper=lambda a,v: a.default_helper(v))
        if dlg.ShowModal() == wx.ID_OK:
            items = dlg.get_items()
            BackgroundWMSDownloader.set_known_hosts(items)
            event.task.remember_wms()
            self.editor.layer_manager.update_map_server_ids("wms", hosts, items)
            self.editor.refresh()
        dlg.Destroy()


class manage_tile_servers(SawxAction):
    name = 'Manage Tile Servers...'

    def perform(self, action_key):
        hosts = BackgroundTileDownloader.get_known_hosts()
        dlg = ListReorderDialog(event.task.window.control, hosts, lambda a: a.label_helper(), prompt_for_tile, "Manage Tile Servers", default_helper=lambda a,v: a.default_helper(v))
        if dlg.ShowModal() == wx.ID_OK:
            items = dlg.get_items()
            BackgroundTileDownloader.set_known_hosts(items)
            event.task.remember_tile_servers()
            self.editor.layer_manager.update_map_server_ids("tiles", hosts, items)
            self.editor.refresh()
        dlg.Destroy()


class clear_tile_cache(SawxAction):
    name = 'Clear Tile Cache...'

    def perform(self, action_key):
        hosts = BackgroundTileDownloader.get_known_hosts()
        dlg = CheckItemDialog(event.task.window.control, hosts, lambda a: getattr(a, 'name'), title="Clear Tile Cache", instructions="Clear cache of selected tile servers:")
        if dlg.ShowModal() == wx.ID_OK:
            try:
                for host in dlg.get_checked_items():
                    host.clear_cache(event.task.get_tile_cache_root())
            except OSError as e:
                event.task.error("Error clearing cache for %s\n\n%s" % (host.name, str(e)))

        dlg.Destroy()


class normalize_longitude(SawxAction):
    name = 'Normalize Longitude'
    tooltip = 'Adjust longitudes so the map lies between 0 and 360W'

    def perform(self, action_key):
        edit_layer = self.editor.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = moc.NormalizeLongitudeCommand(edit_layer)
            self.editor.process_command(cmd)


class swap_lat_lon(LayerAction):
    name = 'Swap Lat && Lon'
    tooltip = 'Exchange coordinate pairs to repair an incorrectly formatted input file'

    def perform_on_layer(self, action_key, layer):
        cmd = moc.SwapLatLonCommand(layer)
        self.editor.process_command(cmd)


class debug_layer_manager(SawxAction):
    name = 'Show Layer Manager Info'
    tooltip = 'Show a debug output describing the currently displayed layers'

    def perform(self, action_key):
        lm = self.editor.layer_manager
        text = lm.debug_structure()
        print(text)



class group_layer(LayerAction):
    name = 'Group Sublayers'
    tooltip = 'Group all children of the selected layer into a single unit'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_groupable_objects() and not layer.grouped

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.group_children(layer)


class ungroup_layer(LayerAction):
    name = 'Ungroup Into Sublayers'
    tooltip = 'Remove grouping and display child layers'

    def is_enabled_for_layer(self, action_key, layer):
        return layer.has_groupable_objects() and layer.grouped

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.ungroup_children(layer)


class rename_layer(LayerAction):
    name = 'Rename Layer'
    tooltip = 'Rename layer'

    def perform_on_layer(self, action_key, layer):
        self.editor.layer_tree_control.start_rename(layer)


class edit_layer(SawxAction):
    name = 'Edit Layer'
    tooltip = 'Edit the currently selected layer'

    def perform(self, action_key):
        d = event.popup_data
        layer = d['layer']
        feature_code = layer.get_feature_code(d['object_index'])
        cmd = mec.PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'], feature_code=feature_code, new_boundary=False)
        self.editor.process_command(cmd)


class add_polygon_to_edit_layer(SawxAction):
    name = 'Add Polygon to Edit Layer'
    tooltip = 'Add a polygon to the current editing layer'

    def perform(self, action_key):
        d = event.popup_data
        cmd = mec.AddPolygonToEditLayerCommand(d['layer'], d['object_type'], d['object_index'], None, False)
        self.editor.process_command(cmd)


class add_polygon_boundary(SawxAction):
    name = 'Add Polygon'
    tooltip = 'Add a new boundary polygon'

    def perform(self, action_key):
        d = event.popup_data
        layer = d['layer']
        try:
            feature_code = layer.get_feature_code(d['object_index'])
        except IndexError:
            feature_code = 1
        cmd = mec.PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'], feature_code=feature_code, new_boundary=True)
        self.editor.process_command(cmd)


class add_polygon_hole(SawxAction):
    name = 'Add Hole'
    tooltip = 'Add a new hole polygon'

    def perform(self, action_key):
        d = event.popup_data
        cmd = mec.PolygonEditLayerCommand(d['layer'], d['object_type'], d['object_index'], feature_code=-1, new_boundary=True)
        self.editor.process_command(cmd)


class delete_polygon(SawxAction):
    name = 'Delete Polygon'
    tooltip = 'Remove a polygon or hole; note that if a polygon is removed, any holes in that polygon are also removed'

    def perform(self, action_key):
        d = event.popup_data
        cmd = mec.DeletePolygonCommand(d['layer'], d['object_type'], d['object_index'])
        self.editor.process_command(cmd)


class simplify_polygon(SawxAction):
    name = 'Simplify Polygon'
    tooltip = 'Remove points using Visvalingam algorithm'

    def perform(self, action_key):
        d = event.popup_data
        dlg = SimplifyDialog(self.editor, d['layer'], d['object_type'], d['object_index'])
        if dlg.ShowModal() != wx.ID_OK:
            dlg.roll_back()

class save_ring_edit(SawxAction):
    name = 'Save Changes in Polygon'
    tooltip = 'Save the current edits in the parent polygon'

    def perform(self, action_key):
        d = event.popup_data
        cmd = mec.PolygonSaveEditLayerCommand(d['layer'])
        self.editor.process_command(cmd)


class cancel_ring_edit(SawxAction):
    name = 'Cancel Edit'
    tooltip = 'Abandon the current edits in the parent polygon'

    def perform(self, action_key):
        d = event.popup_data
        cmd = mec.PolygonCancelEditLayerCommand(d['layer'])
        self.editor.process_command(cmd)


class start_time(LayerAction):
    name = 'Start Time'
    tooltip = 'Set time that layer becomes active'
    dialog_info = 'Set start time of %s\nto start time of:'
    cmd = mec.StartTimeCommand

    def get_time(self, layer):
        return layer.start_time

    def perform_on_layer(self, action_key, layer):
        layers = self.editor.layer_manager.get_playback_layers(layer)
        dlg = wx.SingleChoiceDialog(
                self.editor.layer_tree_control, self.dialog_info % layer.name, self.name,
                [a.name for a in layers],
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            source_layer = layers[dlg.GetSelection()]
            print(('You selected: %s\n' % source_layer))
            cmd = self.cmd(layer, self.get_time(source_layer))
            self.editor.process_command(cmd)


class end_time(start_time):
    name = 'End Time'
    tooltip = 'Set time that layer stops being active'
    dialog_info = 'Set end time of %s\nto end time of:'
    cmd = mec.EndTimeCommand

    def get_time(self, layer):
        return layer.end_time
