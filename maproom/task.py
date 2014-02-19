""" Skeleton sample task

"""
# Enthought library imports.
from pyface.api import ImageResource
from pyface.tasks.api import Task, TaskWindow, TaskLayout, PaneItem, IEditor, \
    IEditorAreaPane, EditorAreaPane, Editor, DockPane, HSplitter, VSplitter
from pyface.tasks.action.api import DockPaneToggleGroup, SMenuBar, \
    SMenu, SToolBar, TaskAction, TaskToggleGroup
from traits.api import on_trait_change, Property, Instance

from peppy2.framework.task import FrameworkTask

from layer_editor import LayerEditor
from panes import LayerSelectionPane, LayerInfoPane

class MaproomTask(FrameworkTask):
    """The Maproom Layer editor task.
    """

    #### Task interface #######################################################

    id = 'maproom.layer_task'
    name = 'Maproom'

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def _default_layout_default(self):
        return TaskLayout(
            left=VSplitter(
                PaneItem('maproom.layer_selection_pane'),
                PaneItem('maproom.layer_info_pane'),
                ))

    def create_central_pane(self):
        """ Create the central pane: the text editor.
        """
        self.editor_area = EditorAreaPane()
        return self.editor_area

    def create_dock_panes(self):
        """ Create the file browser and connect to its double click event.
        """
        return [ LayerSelectionPane(), LayerInfoPane() ]

    ###########################################################################
    # 'FrameworkTask' interface.
    ###########################################################################

    def get_editor(self, guess=None):
        """ Opens a new empty window
        """
        editor = LayerEditor()
        return editor

    ###
    @classmethod
    def can_edit(cls, mime):
        return mime == "application/x-maproom-verdat"
