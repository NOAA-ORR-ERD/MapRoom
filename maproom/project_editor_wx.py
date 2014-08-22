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
from layer_control_wx import LayerControl
from layer_manager import LayerManager
from layer_undo import *
import Layer_tree_control
import renderer

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
    
    clickable_object_mouse_is_over = Any
    
    last_refresh = Float(0.0)
    
    # Force mouse mode category to be blank so that the initial trait change
    # that occurs during initialization of this class doesn't match a real
    # mouse mode.  If it does match, the toolbar won't be properly adjusted
    # during the first trait change in response to update_layer_selection_ui
    # and there will be an empty between named toolbars
    mouse_mode_category = Str("")
    
    mouse_mode = Int(LayerControl.MODE_PAN)

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
            try:
                metadata = guess.get_metadata()
                progress_log.info("START=Loading %s" % metadata.uri)
                layers = self.layer_manager.load_layers_from_metadata(metadata, self)
                if metadata.mime == "application/x-maproom-project-json":
                    self.path = metadata.uri
                
                if layers:
                    self.zoom_to_layers(layers)
            except ProgressCancelError, e:
                self.window.error(e.message)
            finally:
                progress_log.info("END")

        self.dirty = False

    def view_of(self, editor):
        """ Copy the view of the supplied editor.
        """
        self.layer_manager = editor.layer_manager
        self.layer_visibility = self.layer_manager.get_default_visibility()
        self.control.change_view(self.layer_manager)
        self.control.zoom_to_fit()
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
            image = self.control.get_canvas_as_image()
            # "save as" dialog contains confirmation for overwriting existing
            # file, so just write the file here
            image.SaveFile(path, t)
        else:
            self.window.error("Unsupported image type %s" % ext)
    
    def undo(self):
        self.layer_manager.undo()
    
    def redo(self):
        self.layer_manager.redo()

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_control(self, parent):
        """ Creates the toolkit-specific control for the widget. """

        self.layer_manager = LayerManager.create(self)
        self.layer_visibility = self.layer_manager.get_default_visibility()

        # Base-class constructor.
        self.control = LayerControl(parent, layer_manager=self.layer_manager, project=self)
        
        # Tree/Properties controls referenced from MapController
        self.layer_tree_control = self.window.get_dock_pane('maproom.layer_selection_pane').control
        self.layer_info = self.window.get_dock_pane('maproom.layer_info_pane').control
        self.selection_info = self.window.get_dock_pane('maproom.selection_info_pane').control
        
        log.debug("LayerEditor: task=%s" % self.task)

        return self.control
    
    #### Traits event handlers
    
    @on_trait_change('layer_manager:layer_loaded')
    def layer_loaded(self, layer):
        log.debug("layer_loaded called for %s" % layer)
        self.layer_visibility[layer] = layer.get_visibility_dict()
    
    @on_trait_change('layer_manager:layers_changed')
    def layers_changed(self):
        log.debug("layers_changed called!!!")
        self.layer_tree_control.rebuild()
        self.refresh()
    
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
            self.mouse_mode_category = sel_layer.mouse_selection_mode + "ToolBar"
            self.mouse_mode = LayerControl.get_valid_mouse_mode(self.mouse_mode, self.mouse_mode_category)
        else:
            self.layer_can_save = False
            self.layer_can_save_as = False
            self.layer_selected = False
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
            self.mouse_mode = LayerControl.MODE_PAN
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
            if self.layer_has_points:
                log.debug("selected points: %s"  % sel_layer.get_num_points_selected())
                self.layer_has_selection = sel_layer.get_num_points_selected() > 0
            else:
                self.layer_has_selection = False
        else:
            self.layer_has_points = False
            self.layer_has_selection = False
        log.debug("has_points=%s, has_selection = %s" % (self.layer_has_points, self.layer_has_selection))
        self.update_undo_redo()
        self.window._aui_manager.Update()
    
    def update_undo_redo(self):
        u = self.layer_manager.get_current_undoable_operation_text()
        r = self.layer_manager.get_current_redoable_operation_text()

        if (u == ""):
            undo_label = "Undo"
            self.can_undo = False
        else:
            undo_label = "Undo {0}".format(u)
            self.can_undo = True

        if (r == ""):
            redo_label = "Redo"
            self.can_redo = False
        else:
            redo_label = "Redo {0}".format(r)
            self.can_redo = True
        
        self.dirty = self.can_undo
    
    @on_trait_change('layer_manager:undo_stack_changed')
    def undo_stack_changed(self):
        log.debug("undo_stack_changed called!!!")
        self.refresh()
    
    @on_trait_change('layer_manager:layer_contents_changed')
    def layer_contents_changed(self, layer):
        log.debug("layer_contents_changed called!!! layer=%s" % layer)
        self.control.rebuild_points_and_lines_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_contents_deleted')
    def layer_contents_deleted(self, layer):
        log.debug("layer_contents_deleted called!!! layer=%s" % layer)
        self.control.rebuild_points_and_lines_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_metadata_changed')
    def layer_metadata_changed(self, layer):
        log.debug("layer_metadata_changed called!!! layer=%s" % layer)
        self.layer_tree_control.rebuild()
    
    @on_trait_change('layer_manager:refresh_needed')
    def refresh(self):
        log.debug("refresh called")
        if self.control is None:
            return
        
        self.window._aui_manager.Update()
        # On Mac this is neither necessary nor desired.
        if not sys.platform.startswith('darwin'):
            self.control.Update()
        self.control.Refresh()
        
        sel_layer = self.layer_tree_control.get_selected_layer()
        self.update_layer_contents_ui(sel_layer)
        self.layer_info.display_panel_for_layer(self, sel_layer)
        self.selection_info.display_panel_for_layer(self, sel_layer)
        self.last_refresh = time.clock()
    
    @on_trait_change('layer_manager:background_refresh_needed')
    def background_refresh(self):
        log.debug("background refresh called")
        t = time.clock()
        if t < self.last_refresh + 0.5:
            log.debug("refreshed too recently; skipping.")
            return
        self.refresh()
    
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
        if (self.mouse_mode == LayerControl.MODE_EDIT_POINTS or self.mouse_mode == LayerControl.MODE_EDIT_LINES):
            sel_layer = self.layer_tree_control.get_selected_layer()
            if sel_layer is not None:
                sel_layer.delete_all_selected_objects()
                self.layer_manager.end_operation_batch()
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

    def esc_key_pressed(self):
        for layer in self.layer_manager.flatten():
            layer.clear_all_selections()
        self.refresh()

    def delete_key_pressed(self):
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS or self.mouse_mode == self.control.MODE_EDIT_LINES):
            layer = self.layer_tree_control.get_selected_layer()
            if (layer != None):
                layer.delete_all_selected_objects()
                self.layer_manager.end_operation_batch()
                self.refresh()

    def clicked_on_point(self, event, layer, point_index):
        act_like_point_tool = False
        vis = self.layer_visibility[layer]['layer']
        message = ""

        if (self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (event.ControlDown() or event.ShiftDown()):
                act_like_point_tool = True
                pass
            else:
                point_indexes = layer.get_selected_point_indexes()
                if len(point_indexes == 1):
                    if not layer.are_points_connected(point_index, point_indexes[0]):
                        layer.insert_line_segment(point_index, point_indexes[0])
                        self.layer_manager.end_operation_batch()
                        if not vis:
                            message = "Added line to hidden layer %s" % layer.name
                    layer.clear_all_point_selections()
                    if point_indexes[0] != point_index:
                        # allow for deselecting points by clicking them again.
                        # Only if the old selected point is not the same
                        # as the clicked point will the clicked point be
                        # highlighted.
                        layer.select_point(point_index)
                elif len(point_indexes) == 0:  # no currently selected point
                    # select this point
                    layer.select_point(point_index)

        if (self.mouse_mode == self.control.MODE_EDIT_POINTS or act_like_point_tool):
            if (event.ControlDown()):
                if (layer.is_point_selected(point_index)):
                    layer.deselect_point(point_index)
                else:
                    layer.select_point(point_index)
            elif (layer.is_point_selected(point_index)):
                layer.clear_all_selections()
            elif (event.ShiftDown()):
                path = layer.find_points_on_shortest_path_from_point_to_selected_point(point_index)
                if (path != []):
                    for p_index in path:
                        layer.select_point(p_index)
                else:
                    layer.select_point(point_index)
            else:
                layer.clear_all_selections()
                layer.select_point(point_index)

        self.refresh()
        if message:
            self.task.status_bar.message = message

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        vis = self.layer_visibility[layer]['layer']

        if (self.mouse_mode == self.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                self.esc_key_pressed()
                point_index = layer.insert_point_in_line(world_point, line_segment_index)
                self.layer_manager.end_operation_batch()
                self.control.forced_cursor = wx.StockCursor(wx.CURSOR_HAND)
                if not vis:
                    self.task.status_bar.message = "Split line in hidden layer %s" % layer.name
                else:
                    layer.select_point(point_index)

        if (self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (event.ControlDown()):
                if (layer.is_line_segment_selected(line_segment_index)):
                    layer.deselect_line_segment(line_segment_index)
                else:
                    layer.select_line_segment(line_segment_index)
            elif (layer.is_line_segment_selected(line_segment_index)):
                pass
            elif (event.ShiftDown()):
                path = layer.find_lines_on_shortest_path_from_line_to_selected_line(line_segment_index)
                if (path != []):
                    for l_s_i in path:
                        layer.select_line_segment(l_s_i)
                else:
                    layer.select_line_segment(line_segment_index)
            else:
                layer.clear_all_selections()
                layer.select_line_segment(line_segment_index)

        self.refresh()

    def clicked_on_polygon(self, layer, polygon_index):
        pass

    def clicked_on_empty_space(self, event, layer, world_point):
        log.debug("clicked on empty space: layer %s, point %s" % (layer, str(world_point)) )
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS or self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (layer.type == "root" or layer.type == "folder"):
                self.window.error("You cannot add points or lines to folder layers.",
                                  "Cannot Edit")

                return
        vis = self.layer_visibility[layer]['layer']

        log.debug("1: self.mouse_mode=%d" % self.mouse_mode)
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                log.debug("1.1")
                self.esc_key_pressed()
                # we release the focus because we don't want to immediately drag the new object (if any)
                # self.control.release_mouse() # shouldn't be captured now anyway
                point_index = layer.insert_point(world_point)
                layer.select_point(point_index)
                layer.update_bounds()
                self.layer_manager.end_operation_batch(refresh=False)
                self.refresh()
                if not vis:
                    self.task.status_bar.message = "Added point to hidden layer %s" % layer.name
        log.debug("2")
        if (self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (not event.ControlDown() and not event.ShiftDown()):
                point_indexes = layer.get_selected_point_indexes()
                if (len(point_indexes == 1)):
                    self.esc_key_pressed()
                    # we release the focus because we don't want to immediately drag the new object (if any)
                    # self.control.release_mouse()
                    point_index = layer.insert_point(world_point)
                    layer.update_bounds()
                    layer.insert_line_segment(point_index, point_indexes[0])
                    self.layer_manager.end_operation_batch()
                    layer.clear_all_point_selections()
                    layer.select_point(point_index)
                    if not vis:
                        self.task.status_bar.message = "Added line to hidden layer %s" % layer.name
                self.refresh()

    def dragged(self, world_d_x, world_d_y):
        if (self.clickable_object_mouse_is_over == None):
            return

        (layer_index, type, subtype, object_index) = renderer.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.layer_manager.get_layer_by_flattened_index(layer_index)
        layer.offset_selected_objects(world_d_x, world_d_y)
        # self.layer_manager.end_operation_batch()
        self.refresh()

    def finished_drag(self, mouse_down_position, mouse_move_position):
        if (self.clickable_object_mouse_is_over == None):
            return

        d_x = mouse_move_position[0] - mouse_down_position[0]
        d_y = mouse_down_position[1] - mouse_move_position[1]

        if (d_x == 0 and d_y == 0):
            return

        w_p0 = self.control.get_world_point_from_screen_point(mouse_down_position)
        w_p1 = self.control.get_world_point_from_screen_point(mouse_move_position)
        world_d_x = w_p1[0] - w_p0[0]
        world_d_y = w_p1[1] - w_p0[1]

        (layer_index, type, subtype, object_index) = renderer.parse_clickable_object(self.clickable_object_mouse_is_over)
        layer = self.layer_manager.get_layer_by_flattened_index(layer_index)

        s_p_i_s = layer.get_selected_and_dependent_point_indexes()
        for point_index in s_p_i_s:
            params = (world_d_x, world_d_y)
            self.layer_manager.add_undo_operation_to_operation_batch(OP_MOVE_POINT, layer, point_index, params)

        self.layer_manager.end_operation_batch()

    def clickable_object_is_ugrid_point(self):
        return renderer.is_ugrid_point(self.clickable_object_mouse_is_over)

    def clickable_object_is_ugrid_line(self):
        return renderer.is_ugrid_line(self.clickable_object_mouse_is_over)

    def clickable_object_is_polygon_fill(self):
        return renderer.is_polygon_fill(self.clickable_object_mouse_is_over)

    def clickable_object_is_polygon_point(self):
        return renderer.is_polygon_point(self.clickable_object_mouse_is_over)

    #### wx event handlers ####################################################

    #### old RenderController
    
    def zoom_to_selected_layer(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        log.debug("Selected layer = %r" % sel_layer)
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.control.zoom_to_world_rect(layer.bounds)

    def zoom_to_layers(self, layers):
        rect = self.layer_manager.accumulate_layer_bounds_from_list(layers)
        self.control.zoom_to_world_rect(rect)
    
    def check_for_errors(self):
        error = None
        sel_layer = self.layer_tree_control.get_selected_layer()
        try:
            progress_log.info("START=Checking layer %s" % sel_layer.name)
            self.layer_manager.check_layer(sel_layer, self.window)
        except ProgressCancelError, e:
            pass
        finally:
            progress_log.info("END")
