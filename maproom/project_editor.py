# Standard library imports.
import sys
import os.path
import time
import json

# Major package imports.
import wx

# Enthought library imports.
from pyface.api import YES, NO
from traits.api import provides, on_trait_change, Any, Bool, Int, Str, Float, Event, Dict

from omnivore.framework.editor import FrameworkEditor
from omnivore.framework.errors import ProgressCancelError

# Local imports.
from layer_canvas import LayerCanvas
from layer_manager import LayerManager
import renderer
from layers import loaders
from layers.constants import *
from command import UndoStack, BatchStatus
from mouse_handler import *
from menu_commands import *
from serializer import UnknownCommandError
import toolbar

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

class ProjectEditor(FrameworkEditor):
    """The wx implementation of a ProjectEditor.
    
    See the IProjectEditor interface for the API documentation.
    """
    
    printable = True
    
    layer_manager = Any
    
    layer_zoomable = Bool
    
    layer_can_save = Bool
    
    layer_can_save_as = Bool
    
    layer_selected = Bool
    
    layer_above = Bool
    
    layer_below = Bool
    
    multiple_layers = Bool
    
    layer_has_points = Bool
    
    layer_has_selection = Bool
    
    layer_has_flagged = Bool
    
    layer_has_boundaries = Bool
    
    clickable_object_mouse_is_over = Any
    
    clickable_object_in_layer = Any
    
    last_refresh = Float(0.0)
    
    layer_visibility = Dict()
    
    # can_paste_style is set by copy_style if there's a style that can be
    # applied
    can_paste_style = Bool(False)
    
    # NOTE: Class attribute!
    clipboard_style = None
    
    # Force mouse mode toolbar to be blank so that the initial trait change
    # that occurs during initialization of this class doesn't match a real
    # mouse mode.  If it does match, the toolbar won't be properly adjusted
    # during the first trait change in response to update_layer_selection_ui
    # and there will be an empty between named toolbars
    mouse_mode_toolbar = Str("")
    
    mouse_mode_factory = Any(PanMode)

    ###########################################################################
    # 'FrameworkEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)
    
    def load_in_new_tab(self, metadata):
        if metadata.mime == "application/x-maproom-project-json":
            return self.layer_manager.has_user_created_layers()
        return False

    def load_omnivore_document(self, document, layer=False, **kwargs):
        """ Loads the data from the Omnivore document
        """
        metadata = document.metadata
        loader = loaders.get_loader(metadata)
        if hasattr(loader, "load_project"):
            document = LayerManager.create(self)
            document.metadata = metadata.clone_traits()
            batch_flags = BatchStatus()
            print "FIXME: Add load project command that clears all layers"
            extra = loader.load_project(metadata, document, batch_flags)
            self.document = self.layer_manager = document
            if extra is not None:
                self.parse_extra_json(extra, batch_flags)
            self.layer_tree_control.clear_all_items()
            self.layer_tree_control.rebuild()
            self.perform_batch_flags(batch_flags)
            center, units_per_pixel = self.layer_canvas.calc_zoom_to_layers(batch_flags.layers)
            cmd = ViewportCommand(None, center, units_per_pixel)
            self.process_command(cmd)
            
            # Clear modified flag
            self.layer_manager.undo_stack.set_save_point()
            self.dirty = self.layer_manager.undo_stack.is_dirty()
        elif hasattr(loader, "iter_log"):
            line = 0
            batch_flags = BatchStatus()
            for cmd in loader.iter_log(metadata, self.layer_manager):
                line += 1
                errors = None
                try:
                    undo = self.process_batch_command(cmd, batch_flags)
                    if not undo.flags.success:
                        errors = undo.errors
                        break
                except Exception, e:
                    errors = [str(e)]
                    import traceback
                    print traceback.format_exc(e)
                    break
            if errors is not None:
                header = [
                    "While restoring from the command log file:\n\n%s\n" % metadata.uri,
                    "an error occurred on line %d while processing" % line,
                    "the command '%s':" % cmd.short_name,
                    ""
                    ]
                header.extend(errors)
                text = "\n".join(header)
                self.task.error(text, "Error restoring from command log")
            self.perform_batch_flags(batch_flags)
        else:
            cmd = LoadLayersCommand(metadata)
            self.process_command(cmd)
            layers = cmd.undo_info.affected_layers()
            if len(layers) == 1:
                cmd = ViewportCommand(layers[0])
            else:
                center, units_per_pixel = self.layer_canvas.calc_zoom_to_layers(layers)
                cmd = ViewportCommand(None, center, units_per_pixel)
            self.process_command(cmd)
        self.view_document(self.document)
    
    def parse_extra_json(self, json, batch_flags):
        # handle old version which was a two element list
        try:
            if len(json) == 2 and json[0] == "commands":
                json = {json[0]: json[1]}
        except KeyError:
            # a two element dict won't have a key of zero
            pass
        
        if "layer_visibility" in json:
            self.layer_visibility_from_json(json["layer_visibility"])
            
            # Just as a sanity check, make sure all layers have visibility
            # info. Very old json formats may not include visibility info for
            # all layers
            self.update_default_visibility(warn=True)
        else:
            self.layer_visibility = self.get_default_visibility()
        if "commands" in json:
            for cmd in self.layer_manager.undo_stack.unserialize_text(json["commands"], self.layer_manager):
                self.process_batch_command(cmd, batch_flags)
    
    def layer_visibility_to_json(self):
        v = dict()
        for layer, vis in self.layer_visibility.iteritems():
            v[layer.invariant] = vis
        return v
    
    def layer_visibility_from_json(self, json_data):
        lm = self.layer_manager
        v = dict()
        for invariant, vis in json_data.iteritems():
            layer = lm.get_layer_by_invariant(int(invariant))
            if layer is not None:
                # Some layer visibility data from deleted layers may have been
                # saved, so only restore visibility for layers that are known
                v[layer] = vis
        self.layer_visibility = v

    def get_default_visibility(self):
        layer_visibility = dict()
        for layer in self.layer_manager.flatten():
            layer_visibility[layer] = layer.get_visibility_dict()
        return layer_visibility
    
    def update_default_visibility(self, warn=False):
        for layer in self.layer_manager.flatten():
            if layer not in self.layer_visibility:
                if warn:
                    log.warning("adding default visibility for layer %s" % layer)
                self.layer_visibility[layer] = layer.get_visibility_dict()

    def rebuild_document_properties(self):
        self.layer_manager = self.document
        self.update_default_visibility()
        self.layer_canvas.zoom_to_fit()

    def save(self, path=None):
        """ Saves the contents of the editor in a maproom project file
        """
        if path is None:
            path = self.document.uri
        if not path:
            path = "%s.maproom" % self.name
        
        prefs = self.task.get_preferences()
        if prefs.check_errors_on_save:
            if not self.check_all_layers_for_errors(True):
                return
        
        try:
            progress_log.info("START=Saving %s" % path)
            
            cmd = self.get_savepoint()
            u = UndoStack()
            u.add_command(cmd)
            text = str(u.serialize())
            vis = self.layer_visibility_to_json()
            error = self.layer_manager.save_all(path, {"commands": text, "layer_visibility": vis})
        except ProgressCancelError, e:
            error = e.message
        finally:
            progress_log.info("END")
        
        if error:
            self.task.error(error)
        else:
            self.layer_manager.undo_stack.set_save_point()
            
            # Update tab name.  Note that dirty must be changed in order for
            # the trait to be updated, so force a change if needed.  Also,
            # update the URI first because trait callbacks happen immediately
            # and because properties are used for the editor name, no trait
            # event gets called on updating the metadata URI.
            self.layer_manager.metadata.uri = path
            if not self.dirty:
                self.dirty = True
            self.dirty = self.layer_manager.undo_stack.is_dirty()
            
            # Push URL to top of recent files list
            self.window.application.successfully_loaded_event = self.layer_manager.metadata.uri
            
            # refresh window name in case filename has changed
            self.task._active_editor_tab_change(None)
    
    def get_savepoint(self):
        layer = self.layer_tree_control.get_selected_layer()
        cmd = SavepointCommand(layer, self.layer_canvas.get_zoom_rect())
        return cmd

    def save_log(self, path):
        """ Saves the command log to a text file
        """
        # Add temporary SavepointCommand to command history so that it can be
        # serialized, but remove it after seriarization so it doesn't clutter
        # the history
        cmd = self.get_savepoint()
        self.layer_manager.undo_stack.add_command(cmd)
        serializer = self.layer_manager.undo_stack.serialize()
        try:
            fh = open(path, "wb")
            fh.write(str(serializer))
            fh.close()
        except IOError, e:
            self.task.error(str(e))
        self.layer_manager.undo_stack.pop_command()

    def save_layer(self, path, loader=None):
        """ Saves the contents of the current layer in an appropriate file
        """
        layer = self.layer_tree_control.get_selected_layer()
        if layer is None:
            return
        
        prefs = self.task.get_preferences()
        if prefs.check_errors_on_save:
            if not self.check_all_layers_for_errors(True):
                return

        try:
            progress_log.info("START")
            error = self.layer_manager.save_layer(layer, path, loader)
        except ProgressCancelError, e:
            error = e.message
        finally:
            progress_log.info("END")
        if error:
            self.task.error(error)
        else:
            self.window.application.successfully_loaded_event = path
        self.layer_metadata_changed(layer)
        self.update_layer_selection_ui()

    def save_image(self, path):
        """ Saves the contents of the editor in a maproom project file
        """
        valid = {
            '.png': wx.BITMAP_TYPE_PNG,
            '.tif': wx.BITMAP_TYPE_TIFF,
            '.tiff': wx.BITMAP_TYPE_TIFF,
            '.jpg': wx.BITMAP_TYPE_JPEG,
            '.jpeg': wx.BITMAP_TYPE_JPEG,
            }
        _, ext = os.path.splitext(path)
        if ext not in valid:
            path += ".png"
            t = wx.BITMAP_TYPE_PNG
        else:
            t = valid[ext]
        image = self.layer_canvas.get_canvas_as_image()
        # "save as" dialog contains confirmation for overwriting existing
        # file, so just write the file here
        image.SaveFile(path, t)
    
    def print_preview(self):
        import os
        temp = os.path.join(self.window.application.cache_dir, "preview.pdf")
        self.save_as_pdf(temp)
        try:
            os.startfile(temp)
        except AttributeError:
            import subprocess
            if sys.platform == "darwin":
                file_manager = '/usr/bin/open'
            else:
                file_manager = 'xdg-open'
            subprocess.call([file_manager, temp])
    
    def print_page(self):
        self.print_preview()
    
    def save_as_pdf(self, path=None):
        pdf_canvas = renderer.PDFCanvas(project=self, path=path)
        pdf_canvas.copy_viewport_from(self.layer_canvas)
        pdf_canvas.update_renderers()
        pdf_canvas.render()

    @property
    def most_recent_uri(self):
        cmd = self.layer_manager.undo_stack.find_most_recent(LoadLayersCommand)
        if cmd is None:
            return self.layer_manager.metadata.uri
        return cmd.metadata.uri

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_control(self, parent):
        """ Creates the toolkit-specific control for the widget. """

        self.document = self.layer_manager = LayerManager.create(self)
        self.layer_visibility = self.get_default_visibility()

        # Base-class constructor.
        self.layer_canvas = LayerCanvas(parent, project=self)
        
        # Tree/Properties controls referenced from MapController
        self.layer_tree_control = self.window.get_dock_pane('maproom.layer_selection_pane').control
        self.layer_info = self.window.get_dock_pane('maproom.layer_info_pane').control
        self.selection_info = self.window.get_dock_pane('maproom.selection_info_pane').control
        self.undo_history = self.window.get_dock_pane('maproom.undo_history_pane').control
        self.sidebar = self.window.get_dock_pane('maproom.sidebar')
        
        log.debug("LayerEditor: task=%s" % self.task)

        return self.layer_canvas.get_native_control()
    
    #### Traits event handlers
    
    @on_trait_change('layer_manager:layer_loaded')
    def layer_loaded(self, layer):
        log.debug("layer_loaded called for %s" % layer)
        self.layer_visibility[layer] = layer.get_visibility_dict()
    
    @on_trait_change('layer_manager:layers_changed')
    def layers_changed(self):
        log.debug("layers_changed called!!!")
        self.layer_tree_control.rebuild()
    
    def update_layer_menu_ui(self, sel_layer):
        if sel_layer is not None:
            self.can_copy = sel_layer.can_copy()
            self.can_paste = True
            self.can_paste_style = self.clipboard_style is not None
            self.layer_can_save = sel_layer.can_save()
            self.layer_can_save_as = sel_layer.can_save_as()
            self.layer_selected = not sel_layer.is_root()
            self.layer_zoomable = sel_layer.is_zoomable()
            self.layer_above = self.layer_manager.is_raisable(sel_layer)
            self.layer_below = self.layer_manager.is_lowerable(sel_layer)
        else:
            self.can_copy = False
            self.can_paste = False
            self.can_paste_style = False
            self.layer_can_save = False
            self.layer_can_save_as = False
            self.layer_selected = False
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
    
    def update_layer_selection_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            # leave mouse_mode set to current setting
            self.mouse_mode_toolbar = sel_layer.mouse_mode_toolbar
            self.mouse_mode_factory = toolbar.get_valid_mouse_mode(self.mouse_mode_factory, self.mouse_mode_toolbar)
        else:
            self.mouse_mode_factory = PanMode
        self.update_layer_menu_ui(sel_layer)
        self.layer_canvas.set_mouse_handler(self.mouse_mode_factory)
        self.multiple_layers = self.layer_manager.count_layers() > 1
        self.update_info_panels(sel_layer)
        self.update_layer_contents_ui(sel_layer)
        self.task.layer_selection_changed = sel_layer
    
    def update_layer_contents_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_has_points = sel_layer.has_points()
            self.layer_has_selection = sel_layer.has_selection()
            self.layer_has_flagged = sel_layer.has_flagged()
            self.layer_has_boundaries = sel_layer.has_boundaries()
        else:
            self.layer_has_points = False
            self.layer_has_selection = False
            self.layer_has_flagged = False
            self.layer_has_boundaries = False
        log.debug("has_points=%s, has_selection = %s, has_flagged=%s, has_boundaries = %s" % (self.layer_has_points, self.layer_has_selection, self.layer_has_flagged, self.layer_has_boundaries))
        self.update_undo_redo()
        self.sidebar.refresh_active()
        self.window._aui_manager.Update()
    
    def update_info_panels(self, layer, force=False):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer == layer:
            self.layer_info.display_panel_for_layer(self, layer, force)
            self.selection_info.display_panel_for_layer(self, layer, force)
        else:
            log.debug("Attempting to update panel for layer %s that isn't current", layer)
    
    def process_info_panel_keystroke(self, event, text):
        if self.layer_info.process_initial_key(event, text):
            return
        if self.selection_info.process_initial_key(event, text):
            return
    
    @on_trait_change('layer_manager:undo_stack_changed')
    def undo_stack_changed(self):
        log.debug("undo_stack_changed called!!!")
        self.refresh()
    
    @on_trait_change('layer_manager:layer_contents_changed')
    def layer_contents_changed(self, layer):
        log.debug("layer_contents_changed called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_contents_changed_in_place')
    def layer_contents_changed_in_place(self, layer):
        log.debug("layer_contents_changed_in_place called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer, in_place=True)
    
    @on_trait_change('layer_manager:layer_contents_deleted')
    def layer_contents_deleted(self, layer):
        log.debug("layer_contents_deleted called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_metadata_changed')
    def layer_metadata_changed(self, layer):
        log.debug("layer_metadata_changed called!!! layer=%s" % layer)
        self.layer_tree_control.rebuild()
    
    @on_trait_change('layer_manager:refresh_needed')
    def refresh(self, editable_properties_changed=False):
        log.debug("refresh called; editable_properties_changed=%s" % editable_properties_changed)
        if self.control is None:
            return
        
        # current control with focus is used to prevent usability issues with
        # text field editing in the calls to the info panel displays below.
        # Without checking for the current text field it is reformatted every
        # time, moving the cursor position to the beginning and generally
        # being annoying
        current = self.window.control.FindFocus()
        
        # On Mac this is neither necessary nor desired.
        if not sys.platform.startswith('darwin'):
            self.control.Update()
        
        sel_layer = self.layer_tree_control.get_selected_layer()
        self.update_layer_contents_ui(sel_layer)
        self.update_layer_menu_ui(sel_layer)
        self.layer_info.display_panel_for_layer(self, sel_layer, editable_properties_changed, has_focus=current)
        self.selection_info.display_panel_for_layer(self, sel_layer, editable_properties_changed, has_focus=current)
        self.last_refresh = time.clock()
        self.control.Refresh()
    
    @on_trait_change('layer_manager:background_refresh_needed')
    def background_refresh(self):
        log.debug("background refresh called")
        t = time.clock()
        if t < self.last_refresh + 0.5:
            log.debug("refreshed too recently; skipping.")
            return
        self.refresh()
    
    @on_trait_change('layer_manager:threaded_image_loaded')
    def threaded_image_loaded(self, data):
        log.debug("threaded image loaded called")
        (layer, map_server_id), wms_request = data
        print "event happed on %s for map server id %d" % (layer, map_server_id)
        print "wms_request:", wms_request
        if layer.is_valid_threaded_result(map_server_id, wms_request):
            wx.CallAfter(self.layer_canvas.render)
        else:
            print "Throwing away result from old map server id"
    
    
    # New Command processor
    
    def process_command(self, command, new_mouse_mode=None, override_editable_properties_changed=None):
        """Process a single command and immediately update the UI to reflect
        the results of the command.
        """
        b = BatchStatus()
        undo = self.process_batch_command(command, b)
        if override_editable_properties_changed is not None:
            b.editable_properties_changed = override_editable_properties_changed
        self.perform_batch_flags(b)
        history = self.layer_manager.undo_stack.serialize()
        self.window.application.save_log(str(history), "command_log", ".mrc")
        if new_mouse_mode is not None:
            self.mouse_mode_factory = new_mouse_mode
            self.update_layer_selection_ui()

        return undo
    
    def process_flags(self, flags):
        b = BatchStatus()
        self.add_batch_flags(None, flags, b)
        self.perform_batch_flags(b)
        
    def process_batch_command(self, command, b):
        """Process a single command but don't update the UI immediately.
        Instead, update the batch flags to reflect the changes needed to
        the UI.
        
        """
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.add_batch_flags(command, undo.flags, b)
        return undo
    
    def add_batch_flags(self, cmd, f, b):
        """Make additions to the batch flags as a result of the passed-in flags
        
        """
        if f.layers_changed:
            # Only set this to True, never back to False once True
            b.layers_changed = True
        if f.refresh_needed:
            b.refresh_needed = True
        if f.fast_viewport_refresh_needed:
            b.fast_viewport_refresh_needed = True
        if f.errors:
            b.errors.append("In %s:" % str(cmd))
            for e in f.errors:
                b.errors.append("- %s" % e)
            b.errors.append("")
        for lf in f.layer_flags:
            layer = lf.layer
            b.layers.append(layer)
            if lf.layer_items_moved:
                layer.update_bounds()
                b.need_rebuild[layer] = True
                b.editable_properties_changed = True
            if lf.layer_display_properties_changed:
                b.need_rebuild[layer] = False
                b.refresh_needed = True
            if lf.layer_contents_added or lf.layer_contents_deleted:
                b.need_rebuild[layer] = False
                b.refresh_needed = True
            # Hidden layer check only displayed in current window, not any others
            # that are displaying this project
            if lf.hidden_layer_check:
                vis = self.layer_visibility[layer]['layer']
                if not vis:
                    b.messages.append("Warning: operating on hidden layer %s" % layer.name)
            if lf.select_layer:
                # only the last layer in the list will be selected
                b.select_layer = layer
            if lf.layer_loaded:
                self.layer_manager.layer_loaded = layer
                b.layers_changed = True
            if lf.layer_metadata_changed:
                b.metadata_changed = True
    
    def perform_batch_flags(self, b):
        """Perform the UI updates given the BatchStatus flags
        
        """
        for layer in b.layers:
            layer.increment_change_count()
        
        # Use LayerManager events to trigger updates in all windows that are
        # displaying this project
        for layer, in_place in b.need_rebuild.iteritems():
            if in_place:
                self.layer_manager.layer_contents_changed_in_place = layer
            else:
                self.layer_manager.layer_contents_changed = layer
        
        if b.layers_changed:
            self.layer_manager.layers_changed = True
        if b.metadata_changed:
            self.layer_manager.layer_metadata_changed = True
        if b.refresh_needed:
            self.layer_manager.refresh_needed = b.editable_properties_changed
        if b.fast_viewport_refresh_needed:
            self.layer_canvas.render()
        if b.select_layer:
            self.layer_tree_control.select_layer(b.select_layer)
        
        if b.errors:
            self.task.error("\n".join(b.errors))
        if b.messages:
            self.task.information("\n".join(b.messages), "Messages")
        
        self.undo_history.update_history()

    supported_clipboard_data_objects = [wx.CustomDataObject("maproom")]
    
    def create_clipboard_data_object(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            json_data = sel_layer.serialize_json(-999, children=True)
            text = json.dumps(json_data, indent=4)
            print "clipboard object: json data", text
            data_obj = wx.CustomDataObject("maproom")
            data_obj.SetData(text)
            return data_obj

    def process_paste_data_object(self, data_obj, cmd_cls=None):
        print "Found data object %s" % data_obj
        text = data_obj.GetData()
        print "value:", text
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            cmd = PasteLayerCommand(sel_layer, text, self.layer_canvas.world_center)
            self.process_command(cmd)
    
    def copy_style(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.__class__.clipboard_style = sel_layer.style.get_copy()
            self.can_paste_style = True
        
    def paste_style(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            style = self.clipboard_style
            if style is not None:
                cmd = StyleChangeCommand(sel_layer, style)
                self.process_command(cmd)
        
    def clear_selection(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            sel_layer.clear_all_selections()
            self.update_layer_contents_ui()
            self.refresh()

    def delete_selection(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            cmd = sel_layer.delete_all_selected_objects()
            self.process_command(cmd)

    def clear_all_flagged(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            sel_layer.clear_flagged(refresh=False)
            self.update_layer_contents_ui()
            self.refresh()

    def select_all_flagged(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            sel_layer.select_flagged(refresh=False)
            self.update_layer_contents_ui()
            self.refresh()

    def select_boundary(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is None:
            return
        
        error = None
        try:
            progress_log.info("START=Finding outer boundary for %s" % sel_layer.name)
            status = sel_layer.select_outer_boundary()
        except ProgressCancelError, e:
            error = "cancel"
        except Exception, e:
            error = "Can't determine boundary"
        finally:
            progress_log.info("END")
        
        if error == "cancel":
            return
        elif error is not None:
            self.task.error(error, sel_layer.name)
        elif status is None:
            self.task.error("No complete boundary", sel_layer.name)
        else:
            self.update_layer_contents_ui()
            self.refresh()

    def point_tool_selected(self):
        for layer in self.layer_manager.flatten():
            layer.clear_all_line_segment_selections()
        self.refresh()

    def point_tool_deselected(self):
        pass

    def line_tool_selected(self):
        n = 0
        for layer in self.layer_manager.flatten():
            n += layer.get_num_points_selected()
        if (n > 1):
            for layer in self.layer_manager.flatten():
                layer.clear_all_point_selections()
            self.refresh()

    def line_tool_deselected(self):
        pass

    def clear_all_selections(self, refresh=True):
        for layer in self.layer_manager.flatten():
            layer.clear_all_selections()
            layer.clear_visibility_when_deselected(self.layer_visibility[layer])
        if refresh:
            self.refresh()

    def clickable_object_is_ugrid_point(self):
        return self.layer_canvas.picker.is_ugrid_point(self.clickable_object_mouse_is_over)

    def clickable_object_is_ugrid_line(self):
        return self.layer_canvas.picker.is_ugrid_line(self.clickable_object_mouse_is_over)

    def clickable_object_is_polygon_fill(self):
        return self.layer_canvas.picker.is_polygon_fill(self.clickable_object_mouse_is_over)

    def clickable_object_is_polygon_point(self):
        return self.layer_canvas.picker.is_polygon_point(self.clickable_object_mouse_is_over)

    def delete_selected_layer(self, layer=None):
        if layer is None:
            layer = self.layer_tree_control.get_selected_layer()
        if layer is None:
            self.window.status_bar.message = "Selected layer to delete!."
            return

        if layer.is_root():
            m = "The root node of the layer tree is selected. This will delete all layers in the tree."
        else:
            m = None

        if m is not None and self.task.confirm(m, default=YES) != YES:
            return

        cmd = DeleteLayerCommand(layer)
        self.process_command(cmd)

    def check_for_errors(self, save_message=False):
        error = None
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is None:
            return
        
        try:
            progress_log.info("START=Checking layer %s" % sel_layer.name)
            error = self.layer_manager.check_layer(sel_layer, self.window)
        except ProgressCancelError, e:
            error = "cancel"
        finally:
            progress_log.info("END")
        
        self.update_layer_contents_ui()
        all_ok = True
        if error == "cancel":
            all_ok = False
        elif error is not None:
            sel_layer.highlight_exception(error)
            if save_message:
                answer = self.task.confirm(error.message, "Layer Contains Problems; Save Anyway?")
                all_ok = (answer == YES)
            else:
                self.task.error(error.message, "Layer Contains Problems")
        else:
            sel_layer.clear_flagged(refresh=True)
            self.task.information("Layer %s OK" % sel_layer.name, "No Problems Found")
        return all_ok

    def check_all_layers_for_errors(self, save_message=False):
        messages = []
        error = None
        try:
            progress_log.info("START=Checking all layers for errors")
            for layer in self.layer_manager.flatten():
                progress_log.info("TITLE=Checking layer %s" % layer.name)
                error = self.layer_manager.check_layer(layer, self.window)
                if error is not None:
                    messages.append("%s: %s" % (layer.name, error.message))
                    layer.highlight_exception(error)
                    self.control.Refresh()
                    error = None
        except ProgressCancelError, e:
            error = "cancel"
        finally:
            progress_log.info("END")
        
        self.update_layer_contents_ui()
        all_ok = True
        if error == "cancel":
            all_ok = False
        elif messages:
            if save_message:
                msg = "Layers Contains Problems; Save Anyway?"
                answer = self.task.confirm("\n\n".join(messages), msg, no_label="Don't Save", yes_label="Save")
                all_ok = (answer == YES)
            else:
                msg = "Layers With Problems"
                self.task.information("\n\n".join(messages), msg)
        else:
            for layer in self.layer_manager.flatten():
                layer.clear_flagged()
            self.layer_manager.dispatch_event('refresh_needed')
            if not save_message:
                self.task.information("Layers OK", "No Problems Found")
        return all_ok
