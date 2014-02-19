# Standard library imports.
import sys
from os.path import basename

# Major package imports.
import wx

# Enthought library imports.
from traits.api import Bool, Event, Instance, File, Unicode, Property, provides
from pyface.tasks.api import Editor

# Local imports.
from i_layer_editor import ILayerEditor
from layer_control_wx import LayerControl
import app_globals
import Editor as LegacyEditor
import Layer
import Layer_manager
import Layer_tree_control
import RenderController

@provides(ILayerEditor)
class LayerEditor(Editor):
    """ The toolkit specific implementation of a BitmapEditor.  See the
    IBitmapEditor interface for the API documentation.
    """

    #### 'IPythonEditor' interface ############################################

    obj = Instance(File)

    path = Unicode

    dirty = Bool(False)

    name = Property(Unicode, depends_on='path')

    tooltip = Property(Unicode, depends_on='path')

    #### Events ####

    changed = Event

    def _get_tooltip(self):
        return self.path

    def _get_name(self):
        return basename(self.path) or 'Untitled'

    ###########################################################################
    # 'BitmapEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)

    def load(self, guess=None):
        """ Loads the contents of the editor.
        """
        img = wx.EmptyImage(1,1)
        if guess is None:
            path = self.path
        else:
            metadata = guess.get_metadata()
            layer = Layer.Layer()
            layer.read_from_file(metadata.uri)
            if layer.load_error_string != "":
                print "LAYER LOAD ERROR: %s" % layer.load_error_string
                return False
            index = None
            self.layer_manager.insert_layer(index, layer)

        self.dirty = False

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

        self.layer_manager = Layer_manager.Layer_manager()
        self.editor = LegacyEditor.Editor(self.layer_manager)

        # Base-class constructor.
        self.control = LayerControl(parent, layer_manager=self.layer_manager, editor=self.editor)
        self.render_controller = RenderController.RenderController(self.layer_manager, self.control)
        app_globals.application = self
        app_globals.layer_manager = self.layer_manager
        app_globals.editor = self.editor

        # Load the editor's contents.
        self.load()

        return self.control

    #### wx event handlers ####################################################

