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
import Editor as LegacyEditor
from layer_manager import LayerManager
import Layer_tree_control

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
    
    mouse_mode = Int

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
        self.control.change_view(self.layer_manager, editor.editor)
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
        self.editor = LegacyEditor.Editor(self)

        # Base-class constructor.
        self.control = LayerControl(parent, layer_manager=self.layer_manager, editor=self.editor, project=self)
        
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
            self.mouse_mode = self.control.mode
        else:
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
            self.mouse_mode = self.control.MODE_PAN
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
    
    #### old Editor

    def clear_selection(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            sel_layer.clear_all_selections()
            self.update_layer_contents_ui()
            self.refresh()

    def delete_selection(self):
        if (self.control.mode == self.control.MODE_EDIT_POINTS or self.control.mode == self.control.MODE_EDIT_LINES):
            sel_layer = self.layer_tree_control.get_selected_layer()
            if sel_layer is not None:
                sel_layer.delete_all_selected_objects()
                self.layer_manager.end_operation_batch()
                self.update_layer_contents_ui()
                self.refresh()

    #### wx event handlers ####################################################

    #### old RenderController
    
    def zoom_to_selected_layer(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        print "Selected layer = %r" % sel_layer
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.control.zoom_to_world_rect(layer.bounds)
