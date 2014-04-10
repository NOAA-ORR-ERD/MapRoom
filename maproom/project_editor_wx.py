# Standard library imports.
import sys
from os.path import basename

# Major package imports.
import wx

# Enthought library imports.
from traits.api import provides, on_trait_change, Any, Bool, Int

from peppy2.framework.editor import FrameworkEditor

# Local imports.
from i_project_editor import IProjectEditor
from layer_control_wx import LayerControl
from layer_manager import LayerManager
import Layer_tree_control
import renderer

@provides(IProjectEditor)
class ProjectEditor(FrameworkEditor):
    """The wx implementation of a ProjectEditor.
    
    See the IProjectEditor interface for the API documentation.
    """
    layer_manager = Any
    
    layer_zoomable = Bool
    
    layer_above = Bool
    
    layer_below = Bool
    
    layer_has_points = Bool
    
    layer_has_selection = Bool
    
    clickable_object_mouse_is_over = Any
    
    mouse_mode = Int(LayerControl.MODE_PAN)

    #### property getters

    def _get_name(self):
        return basename(self.path) or 'Untitled Project'

    ###########################################################################
    # 'FrameworkEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)

    def load(self, guess=None, layer=False, **kwargs):
        """ Loads the contents of the editor.
        """
        if guess is None:
            path = self.path
        else:
            metadata = guess.get_metadata()
            layer = self.layer_manager.load_layer_from_metadata(metadata)
            
            if layer:
                self.zoom_to_layer(layer)

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
        """ Saves the contents of the editor.
        """
        if path is None:
            path = self.path

        self.control.saveImage(path)

        self.dirty = False

    ###########################################################################
    # Trait handlers.
    ###########################################################################

    def _path_changed(self):
        if self.control is not None:
            self.load()

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
        self.layer_tree_control = self.editor_area.task.window.get_dock_pane('maproom.layer_selection_pane').control
        self.properties_panel = self.editor_area.task.window.get_dock_pane('maproom.layer_info_pane').control
        self.properties_panel = None
        
        print "LayerEditor: task=%s" % self.editor_area.task

        return self.control
    
    #### Traits event handlers
    
    @on_trait_change('layer_manager:layer_loaded')
    def layer_loaded(self, layer):
        print "layer_loaded called for %s" % layer
        self.layer_visibility[layer] = layer.get_visibility_dict()
    
    @on_trait_change('layer_manager:layers_changed')
    def layers_changed(self):
        print "layers_changed called!!!"
        self.layer_tree_control.rebuild()
        self.refresh()
    
    def update_layer_selection_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_zoomable = sel_layer.is_zoomable()
            self.layer_above = self.layer_manager.is_raisable(sel_layer)
            self.layer_below = self.layer_manager.is_lowerable(sel_layer)
            # leave mouse_mode set to current setting
        else:
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
            self.mouse_mode = LayerControl.MODE_PAN
        print "layer=%s, zoomable = %s" % (sel_layer, self.layer_zoomable)
        self.update_layer_contents_ui(sel_layer)
    
    def update_layer_contents_ui(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_has_points = sel_layer.has_points()
            if self.layer_has_points:
                print "selected points: %s"  % sel_layer.get_num_points_selected()
                self.layer_has_selection = sel_layer.get_num_points_selected() > 0
            else:
                self.layer_has_selection = False
        else:
            self.layer_has_points = False
            self.layer_has_selection = False
        print "has_points=%s, has_selection = %s" % (self.layer_has_points, self.layer_has_selection)
    
    @on_trait_change('layer_manager:layer_contents_changed')
    def layer_contents_changed(self, layer):
        print "layer_contents_changed called!!!"
        self.control.rebuild_points_and_lines_for_layer(layer)
    
    @on_trait_change('layer_manager:layer_contents_triangulated')
    def layer_contents_triangulated(self, layer):
        print "layer_contents_changed called!!!"
        self.control.rebuild_triangles_for_layer(layer)
    
    @on_trait_change('layer_manager:refresh_needed')
    def refresh(self):
        print "refresh called"
        if self.control is None:
            return
        
        # On Mac this is neither necessary nor desired.
        if not sys.platform.startswith('darwin'):
            self.control.Update()
        self.control.Refresh()
        
        if (self.properties_panel is not None):
            layer = self.layer_tree_control.get_selected_layer()
            # note that the following call only does work if the properties for the layer have changed
            self.properties_panel.display_panel_for_layer(layer)
    
    #### old Editor ########################################################

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

        if (self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (event.ControlDown() or event.ShiftDown()):
                act_like_point_tool = True
                pass
            else:
                point_indexes = layer.get_selected_point_indexes()
                if (len(point_indexes == 1) and not layer.are_points_connected(point_index, point_indexes[0])):
                    layer.insert_line_segment(point_index, point_indexes[0])
                    self.layer_manager.end_operation_batch()
                    layer.clear_all_point_selections()
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
                pass
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

    def clicked_on_line_segment(self, event, layer, line_segment_index, world_point):
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                self.esc_key_pressed()
                layer.insert_point_in_line(world_point, line_segment_index)
                self.layer_manager.end_operation_batch()
                self.control.forced_cursor = wx.StockCursor(wx.CURSOR_HAND)

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
        print "clicked on empty space: layer %s, point %s" % (layer, str(world_point)) 
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS or self.mouse_mode == self.control.MODE_EDIT_LINES):
            if (layer.type == "root" or layer.type == "folder"):
                wx.MessageDialog(
                    wx.GetApp().GetTopWindow(),
                    caption="Cannot Edit",
                    message="You cannot add points or lines to folder layers.",
                    style=wx.OK | wx.ICON_ERROR
                ).ShowModal()

                return
        print "1: self.mouse_mode=%d" % self.mouse_mode
        if (self.mouse_mode == self.control.MODE_EDIT_POINTS):
            if (not event.ControlDown() and not event.ShiftDown()):
                print "1.1"
                self.esc_key_pressed()
                # we release the focus because we don't want to immediately drag the new object (if any)
                # self.control.release_mouse() # shouldn't be captured now anyway
                layer.insert_point(world_point)
                layer.update_bounds()
                self.layer_manager.end_operation_batch()
                self.refresh()
        print "2"
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

        s_p_i_s = layer.get_selected_point_plus_line_point_indexes()
        for point_index in s_p_i_s:
            params = (world_d_x, world_d_y)
            self.add_undo_operation_to_operation_batch(OP_MOVE_POINT, layer, point_index, params)

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
        print "Selected layer = %r" % sel_layer
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.control.zoom_to_world_rect(layer.bounds)