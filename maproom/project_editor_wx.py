# Standard library imports.
import sys
from os.path import basename

# Major package imports.
import wx
from wx.lib.pubsub import pub

# Enthought library imports.
from traits.api import provides

from peppy2.framework.editor import FrameworkEditor

# Local imports.
from i_project_editor import IProjectEditor
from layer_control_wx import LayerControl
import Editor as LegacyEditor
import Layer
import Layer_manager
import Layer_tree_control

@provides(IProjectEditor)
class ProjectEditor(FrameworkEditor):
    """The wx implementation of a ProjectEditor.
    
    See the IProjectEditor interface for the API documentation.
    """

    ###########################################################################
    # 'FrameworkEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)

    def load(self, guess=None):
        """ Loads the contents of the editor.
        """
        if guess is None:
            path = self.path
        else:
            metadata = guess.get_metadata()
            layer = self.layer_manager.load_layer_from_uri(metadata.uri)
            self.layer_visibility[layer] = layer.get_visibility_dict()
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

        self.layer_manager = Layer_manager.Layer_manager(self)
        self.layer_visibility = self.layer_manager.get_default_visibility()
        self.editor = LegacyEditor.Editor(self)

        # Base-class constructor.
        self.control = LayerControl(parent, layer_manager=self.layer_manager, editor=self.editor, project=self)
        
        # Tree/Properties controls referenced from MapController
        self.layer_tree_control = None
        self.properties_panel = None
        
        # Pubsub stuff from RenderController
        self.setup_pubsub()
        
        print "LayerEditor: task=%s" % self.editor_area.task

        return self.control

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
        sel_layer = self.layer_manager.get_selected_layer()
        print "Selected layer = %r" % sel_layer
        if sel_layer is not None:
            self.zoom_to_layer(sel_layer)

    def zoom_to_layer(self, layer):
        self.control.zoom_to_world_rect(layer.bounds)

    # Some conversion object reference types
    #
    # Application.current_map == MapController

    #### old MapController
    
    def refresh(self, rebuild_layer_tree_control=False):
        """from MapController
        """
        print "refresh called"
#        # fixme: this shouldn't be required!
#        if (self.is_closing):
#            return
        if (rebuild_layer_tree_control and self.layer_tree_control != None):
            self.layer_tree_control.rebuild()
        if self.control is not None:
        #    self.renderer.render()
            # On Mac this is neither necessary nor desired.
            if not sys.platform.startswith('darwin'):
                self.control.Update()
            self.control.Refresh()
        if (self.layer_tree_control != None and self.properties_panel != None):
            layer = self.layer_tree_control.get_selected_layer()
            # note that the following call only does work if the properties for the layer have changed
            self.properties_panel.display_panel_for_layer(layer)
