# Standard library imports.
import sys
from os.path import basename

# Major package imports.
import wx
from wx.lib.pubsub import pub

# Enthought library imports.
from traits.api import provides, on_trait_change, Any, Bool

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
        
        # Pubsub stuff from RenderController
        self.setup_pubsub()
        
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
    
    def layer_selection_changed(self, sel_layer=None):
        if sel_layer is None:
            sel_layer = self.layer_tree_control.get_selected_layer()
        if sel_layer is not None:
            self.layer_zoomable = sel_layer.is_zoomable()
            self.layer_above = self.layer_manager.is_raisable(sel_layer)
            self.layer_below = self.layer_manager.is_lowerable(sel_layer)
        else:
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False
        print "layer=%s, zoomable = %s" % (sel_layer, self.layer_zoomable)
    
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

    #### wx event handlers ####################################################

    #### old RenderController
    
    def setup_pubsub(self):
        pub.subscribe(self.on_layer_points_lines_changed, ('layer', 'lines', 'changed'))
        pub.subscribe(self.on_layer_points_lines_changed, ('layer', 'points', 'changed'))
        pub.subscribe(self.on_layer_points_lines_changed, ('layer', 'points', 'deleted'))

        pub.subscribe(self.on_projection_changed, ('layer', 'projection', 'changed'))
        pub.subscribe(self.on_layer_loaded, ('layer', 'loaded'))
        pub.subscribe(self.on_layer_updated, ('layer', 'updated'))
        pub.subscribe(self.on_layer_triangulated, ('layer', 'triangulated'))

    def on_layer_updated(self, layer):
        if layer in self.layer_manager.layers:
            self.control.update_renderers()

    def on_layer_loaded(self, layer):
        self.zoom_to_layer(layer)

    def on_layer_points_lines_changed(self, layer):
        if layer in self.layer_manager.layers:
            self.control.rebuild_points_and_lines_for_layer(layer)

    def on_projection_changed(self, layer, projection):
        if layer in self.layer_manager.layers:
            self.control.reproject_all(projection)

    def on_layer_triangulated(self, layer):
        if layer in self.layer_manager.layers:
            self.control.rebuild_triangles_for_layer(layer)

    def zoom_to_selected_layer(self):
        sel_layer = self.layer_tree_control.get_selected_layer()
        print "Selected layer = %r" % sel_layer
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.control.zoom_to_world_rect(layer.bounds)
