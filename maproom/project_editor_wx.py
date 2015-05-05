# Standard library imports.
import sys
import os.path
import time

# Major package imports.
import wx

# Enthought library imports.
from pyface.api import YES, NO
from traits.api import provides, on_trait_change, Any, Bool, Int, Str, Float, Event

from peppy2.framework.editor import FrameworkEditor
from peppy2.framework.errors import ProgressCancelError

# Local imports.
from i_project_editor import IProjectEditor
from layer_canvas import LayerCanvas
from layer_manager import LayerManager
import renderer
from layers import loaders
from layers.constants import *
from command import BatchStatus
from mouse_handler import *
from menu_commands import *
from serializer import UnknownCommandError

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

@provides(IProjectEditor)
class ProjectEditor(FrameworkEditor):
    """The wx implementation of a ProjectEditor.
    
    See the IProjectEditor interface for the API documentation.
    """
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
    
    clickable_object_mouse_is_over = Any
    
    clickable_object_in_layer = Any
    
    last_refresh = Float(0.0)
    
    # Force mouse mode toolbar to be blank so that the initial trait change
    # that occurs during initialization of this class doesn't match a real
    # mouse mode.  If it does match, the toolbar won't be properly adjusted
    # during the first trait change in response to update_layer_selection_ui
    # and there will be an empty between named toolbars
    mouse_mode_toolbar = Str("")
    
    mouse_mode = Any(PanMode)

    #### property getters

    def _get_name(self):
        return os.path.basename(self.path) or 'Untitled Project'

    ###########################################################################
    # 'FrameworkEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)
    
    def load_in_new_tab(self, guess):
        metadata = guess.get_metadata()
        if metadata.mime == "application/x-maproom-project-json":
            return self.layer_manager.has_user_created_layers()
        return False

    def load(self, guess=None, layer=False, **kwargs):
        """ Loads the contents of the editor.
        """
        if guess is None:
            path = self.path
        else:
            metadata = guess.get_metadata()
            loader = loaders.get_loader(metadata)
            if hasattr(loader, "load_project"):
                print "FIXME: Add load project command that clears all layers"
                self.path = metadata.uri
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
                    self.window.error(text, "Error restoring from command log")
                batch_flags.zoom_to_all()
                self.perform_batch_flags(batch_flags)
            else:
                cmd = LoadLayersCommand(metadata)
                self.process_command(cmd)

        self.dirty = False

    def view_of(self, editor):
        """ Copy the view of the supplied editor.
        """
        self.layer_manager = editor.layer_manager
        self.layer_visibility = self.layer_manager.get_default_visibility()
        self.layer_canvas.change_view(self.layer_manager)
        self.layer_canvas.zoom_to_fit()
        self.dirty = editor.dirty

    def save(self, path=None):
        """ Saves the contents of the editor in a maproom project file
        """
        if path is None:
            path = self.path
        if not path:
            path = "%s.maproom" % self.name
        
        try:
            progress_log.info("START=Saving %s" % path)
            
            # FIXME: need to determine the project file format!!!
            error = self.layer_manager.save_all(path)
        except ProgressCancelError, e:
            error = e.message
        finally:
            progress_log.info("END")
        
        if error:
            self.window.error(error)
        else:
            self.path = path
        self.dirty = False

    def save_log(self, path):
        """ Saves the command log to a text file
        """
        serializer = self.layer_manager.undo_stack.serialize()
        try:
            fh = open(path, "wb")
            fh.write(str(serializer))
            fh.close()
        except IOError, e:
            self.window.error(str(e))

    def save_layer(self, path, loader=None):
        """ Saves the contents of the current layer in an appropriate file
        """
        layer = self.layer_tree_control.get_selected_layer()
        if layer is None:
            return
        try:
            progress_log.info("START")
            error = self.layer_manager.save_layer(layer, path, loader)
        except ProgressCancelError, e:
            error = e.message
        finally:
            progress_log.info("END")
        if error:
            self.window.error(error)
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
            }
        _, ext = os.path.splitext(path)
        t = valid.get(ext.lower(), None)
        if t is not None:
            image = self.layer_canvas.get_canvas_as_image()
            # "save as" dialog contains confirmation for overwriting existing
            # file, so just write the file here
            image.SaveFile(path, t)
        else:
            self.window.error("Unsupported image type %s" % ext)

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_control(self, parent):
        """ Creates the toolkit-specific control for the widget. """

        self.layer_manager = LayerManager.create(self)
        self.layer_visibility = self.layer_manager.get_default_visibility()

        # Base-class constructor.
        self.layer_canvas = LayerCanvas(parent, layer_manager=self.layer_manager, project=self)
        
        # Tree/Properties controls referenced from MapController
        self.layer_tree_control = self.window.get_dock_pane('maproom.layer_selection_pane').control
        self.layer_info = self.window.get_dock_pane('maproom.layer_info_pane').control
        self.selection_info = self.window.get_dock_pane('maproom.selection_info_pane').control
        self.undo_history = self.window.get_dock_pane('maproom.undo_history_pane').control
        
        # log.debug("LayerEditor: task=%s" % self.task)

        return self.layer_canvas.get_native_control()
    
    #### Traits event handlers
    
    @on_trait_change('layer_manager:layer_loaded')
    def layer_loaded(self, layer):
        # log.debug("layer_loaded called for %s" % layer)
        self.layer_visibility[layer] = layer.get_visibility_dict()
    
    @on_trait_change('layer_manager:layers_changed')
    def layers_changed(self):
        # log.debug("layers_changed called!!!")
        self.layer_tree_control.rebuild()
    
    def update_layer_selection_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_can_save = sel_layer.can_save()
            self.layer_can_save_as = sel_layer.can_save_as()
            self.layer_selected = not sel_layer.is_root()
            self.layer_zoomable = sel_layer.is_zoomable()
            self.layer_above = self.layer_manager.is_raisable(sel_layer)
            self.layer_below = self.layer_manager.is_lowerable(sel_layer)
            # leave mouse_mode set to current setting
            self.mouse_mode_toolbar = sel_layer.mouse_mode_toolbar
            self.mouse_mode = LayerCanvas.get_valid_mouse_mode(self.mouse_mode, self.mouse_mode_toolbar)
        else:
            self.layer_can_save = False
            self.layer_can_save_as = False
            self.layer_selected = False
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
            self.mouse_mode = PanMode
        self.layer_canvas.set_mouse_handler(self.mouse_mode)
        self.multiple_layers = self.layer_manager.count_layers() > 1
        self.update_layer_contents_ui(sel_layer)
        self.layer_info.display_panel_for_layer(self, sel_layer)
        self.selection_info.display_panel_for_layer(self, sel_layer)
        self.task.layer_selection_changed = sel_layer
    
    def update_layer_contents_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_has_points = sel_layer.has_points()
            self.layer_has_selection = sel_layer.has_selection()
            self.layer_has_flagged = sel_layer.has_flagged()
        else:
            self.layer_has_points = False
            self.layer_has_selection = False
            self.layer_has_flagged = False
        # log.debug("has_points=%s, has_selection = %s" % (self.layer_has_points, self.layer_has_selection))
        self.update_undo_redo()
        self.window._aui_manager.Update()
    
    @on_trait_change('layer_manager:undo_stack_changed')
    def undo_stack_changed(self):
        # log.debug("undo_stack_changed called!!!")
        self.refresh()
    
    @on_trait_change('layer_manager:layer_contents_changed')
    def layer_contents_changed(self, layer):
        log.debug("layer_contents_changed called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_contents_changed_in_place')
    def layer_contents_changed_in_place(self, layer):
        log.debug("layer_contents_changed called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer, in_place=True)
    
    @on_trait_change('layer_manager:layer_contents_deleted')
    def layer_contents_deleted(self, layer):
        log.debug("layer_contents_deleted called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_metadata_changed')
    def layer_metadata_changed(self, layer):
        # log.debug("layer_metadata_changed called!!! layer=%s" % layer)
        self.layer_tree_control.rebuild()
    
    @on_trait_change('layer_manager:refresh_needed')
    def refresh(self):
        log.debug("refresh called")
        if self.control is None:
            return
        
        # On Mac this is neither necessary nor desired.
        if not sys.platform.startswith('darwin'):
            self.control.Update()
        
        sel_layer = self.layer_tree_control.get_selected_layer()
        self.update_layer_contents_ui(sel_layer)
        self.layer_info.display_panel_for_layer(self, sel_layer)
        self.selection_info.display_panel_for_layer(self, sel_layer)
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
    
    
    
    # New Command processor
    
    def update_undo_redo(self):
        command = self.layer_manager.undo_stack.get_undo_command()
        if command is None:
            self.undo_label = "Undo"
            self.can_undo = False
        else:
            text = str(command).replace("&", "&&")
            self.undo_label = "Undo: %s" % text
            self.can_undo = True
            
        command = self.layer_manager.undo_stack.get_redo_command()
        if command is None:
            self.redo_label = "Redo"
            self.can_redo = False
        else:
            text = str(command).replace("&", "&&")
            self.redo_label = "Redo: %s" % text
            self.can_redo = True
            
        self.dirty = self.can_undo
    
    def undo(self):
        undo = self.layer_manager.undo_stack.undo(self)
        self.process_flags(undo.flags)
    
    def redo(self):
        undo = self.layer_manager.undo_stack.redo(self)
        self.process_flags(undo.flags)
    
    def process_command(self, command):
        """Process a single command and immediately update the UI to reflect
        the results of the command.
        """
        b = BatchStatus()
        undo = self.process_batch_command(command, b)
        self.perform_batch_flags(b)
        history = self.layer_manager.undo_stack.serialize()
        self.window.application.save_log(str(history), "command_log", ".mrc")
        return undo
    
    def process_batch_command(self, command, b):
        """Process a single command but don't update the UI immediately.
        Instead, update the batch flags to reflect the changes needed to
        the UI.
        
        """
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.add_batch_flags(undo.flags, b)
        return undo
    
    def add_batch_flags(self, f, b):
        """Make additions to the batch flags as a result of the passed-in flags
        
        """
        if f.layers_changed:
            # Only set this to True, never back to False once True
            b.layers_changed = True
        for lf in f.layer_flags:
            layer = lf.layer
            b.layers.append(layer)
            if lf.layer_items_moved:
                layer.update_bounds()
                b.need_rebuild[layer] = True
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
            if lf.zoom_to_layer:
                b.zoom_layers.append(layer)
            if lf.layer_metadata_changed:
                b.metadata_changed = True
    
    def perform_batch_flags(self, b):
        """Perform the UI updates given the BatchStatus flags
        
        """
        # Use LayerManager events to trigger updates in all windows that are
        # displaying this project
        for layer, in_place in b.need_rebuild.iteritems():
            if in_place:
                self.layer_manager.layer_contents_changed_in_place = layer
            else:
                self.layer_manager.layer_contents_changed = layer
        
        if b.layers_changed:
            self.layer_manager.layers_changed = True
        if b.zoom_layers:
            self.zoom_to_layers(b.zoom_layers)
        if b.metadata_changed:
            self.layer_manager.layer_metadata_changed = True
        if b.refresh_needed:
            self.layer_manager.refresh_needed = True
        if b.select_layer:
            self.layer_tree_control.select_layer(b.select_layer)
        
        self.undo_history.update_history()

    #### old Editor ########################################################

#    editing rules:
#        - click on point tool deselects any selected lines (DONE)
#        - click on line tool deselects all selected points, unless there is zero or one selected point (DONE)
#        - Esc key clears all selections (DONE)
#        - click on Clear Selection button clears all selections (DONE)
#        - selecting a different line in the tree clears all selections (DONE)
#        - moving a layer up or down in the tree clears all selections (DONE)
#        - Delete key or click on Delete button deletes all selected points and lines (and any lines using the deleted points) (DONE)
#        
#        - with point tool
#            - mouse-over point changes cursor to hand (DONE)
#            - mouse-over line changes cursor to bull's eye (DONE)
#            - click on line:
#                - if either Shift or Control key is down, do nothing (DONE)
#                - else insert point into line, select only the new point, and
#                  transition into drag mode (DONE)
#            - click on point:
#                - if Control key down toggle selection of the point (DONE)
#                - else if Shift key down:
#                    - if the point is already selected, do nothing (DONE)
#                    - else if the point is connected to other selected points via lines, select
#                      all points on the shortest such path (and leave any other selections) (DONE)
#                    - else select the point (and leave any other point selections) (DONE)
#                - else if the point was already selected, do nothing (DONE)
#                - else select only that point (DONE)
#            - drag on point moves all selected points/lines (DONE)
#            - click on empty space:
#                - if either Shift or Control key is down, do nothing (DONE)
#                - else add a new point, select only the new point, and do not transition
#                  into drag mode (DONE)
#        - with line tool
#            - mouse-over point changes cursor to hand (DONE)
#            - mouse-over line changes cursor to hand (DONE)
#            - click on point:
#                - if Shift or Control is down, do nothing (DONE)
#                - else if a point is selected, if there
#                  is not already a line from the selected point to the clicked point,
#                  add the line; leave only the clicked point selected (DONE)
#            - click on line:
#                - if Control key down toggle selection of the line (DONE)
#                - else if Shift key down
#                    - if the line is already selected, do nothing (DONE)
#                    - else if the line is connected to other selected lines via lines, select
#                      all line segments along the shortest such path (and leave any other selections) (DONE)
#                - else if line segment was not yet selected, select only this line segment (DONE)
#                - else do nothing (DONE)
#            - drag on point or line moves all selected points/lines (DONE)
#            - click on empty space:
#                - if either Shift or Control key is down, do nothing (DONE)
#                - else add a new point and connect the selected point (if any)
#                  to the new point; leave only the new point selected (DONE)
#        - properties panel
#            - update the properties panel when points or lines area added or deleted, or when
#              the selection changes:
#                - if points are selected show the list of points specified, and their depth
#                  in the "Point depth" field if they all have the same depth, else blank in
#                  the "Point depth" field (DONE)
#                - else show the properties panel for the layer, including the "Default depth" field (DONE)
#            - when the user changes the "Point depth" field to a valid number:
#                - set the depth of selected point(s) and update their labels (DONE)
#    
#    undoable operations:
#        - add points and/or lines
#        - delete points and/or lines
#        - drag points and/or lines
#        - change depth of points
#    non-undoable operations:
#        - load layer
#        - delete layer (deleting a layer deletes undo operations for that layer from the undo stack)
#        - rename layer
#        - triangulate

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
            self.window.error(error, sel_layer.name)
        elif status is None:
            self.window.error("No complete boundary", sel_layer.name)
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

    def dragged(self, world_d_x, world_d_y):
        if (self.clickable_object_mouse_is_over is None):
            return

        (layer_index, type, subtype, object_index) = self.layer_canvas.picker.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.layer_manager.get_layer_by_pick_index(layer_index)
        cmd = layer.dragging_selected_objects(world_d_x, world_d_y)
        self.process_command(cmd)

    def finished_drag(self, mouse_down_position, mouse_move_position, world_d_x, world_d_y):
        if (self.clickable_object_mouse_is_over is None):
            return

        # Can't compare mouse positions in screen coordinates, because the
        # screen may have been scrolled or zoomed.  Have to look at world
        # coords to see if there has been a change.
        
        if (world_d_x == 0 and world_d_y == 0):
            return

#        w_p0 = self.layer_canvas.get_world_point_from_screen_point(mouse_down_position)
#        w_p1 = self.layer_canvas.get_world_point_from_screen_point(mouse_move_position)
#        world_d_x = w_p1[0] - w_p0[0]
#        world_d_y = w_p1[1] - w_p0[1]

        (layer_index, type, subtype, object_index) = self.layer_canvas.picker.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.layer_manager.get_layer_by_pick_index(layer_index)

        cmd = layer.dragging_selected_objects(world_d_x, world_d_y)
        self.process_command(cmd)

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

        if (layer.type == "root"):
            m = "The root node of the layer tree is selected. This will delete all layers in the tree."
        elif (layer.type == "folder"):
            m = "A folder in the layer tree is selected. This will delete the entire sub-tree of layers."
        else:
            m = "Are you sure you want to delete " + layer.name + "?"

        if self.window.confirm(m, default=YES) != YES:
            return

        cmd = DeleteLayerCommand(layer)
        self.process_command(cmd)

    #### wx event handlers ####################################################

    #### old RenderController
    
    def zoom_to_selected_layer(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.layer_canvas.zoom_to_world_rect(layer.bounds)

    def zoom_to_layers(self, layers):
        rect = self.layer_manager.accumulate_layer_bounds_from_list(layers)
        self.layer_canvas.zoom_to_world_rect(rect)
    
    def check_for_errors(self):
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
        
        if error == "cancel":
            return
        elif error is not None:
            sel_layer.highlight_exception(error)
            self.window.error(error.message, "Layer Contains Problems")
        else:
            sel_layer.clear_flagged(refresh=True)
            self.window.information("Layer %s OK" % sel_layer.name, "No Problems Found")
