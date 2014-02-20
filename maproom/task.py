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

from project_editor import ProjectEditor
from panes import LayerSelectionPane, LayerInfoPane

class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """

    #### Task interface #######################################################

    id = 'maproom.project_task'
    name = 'Maproom Project File'

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def _default_layout_default(self):
        return TaskLayout(
            left=VSplitter(
                PaneItem('maproom.layer_selection_pane'),
                PaneItem('maproom.layer_info_pane'),
                ))

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
        editor = ProjectEditor()
        return editor

    ###
    @classmethod
    def can_edit(cls, mime):
        return mime == "application/x-maproom-verdat"
