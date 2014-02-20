""" A widget for displaying bitmapped images. """


# Enthought library imports.
from traits.api import Bool, Event, Instance, File, Interface, Unicode
from pyface.tasks.i_editor import IEditor


class IProjectEditor(IEditor):
    """A widget for editing maproom projects.
    
    A maproom project consists of several layers of various types arranged in a
    stacking order.  Layers might be bitmap images, vector images, annotation
    layers, grid layers, etc.
    """

    #### 'IPythonEditor' interface ############################################

    # Object being editor is a file
    obj = Instance(File)

    # The pathname of the file being edited.
    path = Unicode

    #### Events ####

    # The contents of the editor has changed.
    changed = Event

    ###########################################################################
    # 'IPythonEditor' interface.
    ###########################################################################

    def load(self, path=None):
        """ Loads the contents of the editor. """

    def save(self, path=None):
        """ Saves the contents of the editor. """
