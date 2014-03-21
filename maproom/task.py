""" Skeleton sample task

"""
# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, YES, OK, CANCEL
from pyface.tasks.api import Task, TaskWindow, TaskLayout, PaneItem, IEditor, \
    IEditorAreaPane, EditorAreaPane, Editor, DockPane, HSplitter, VSplitter
from pyface.tasks.action.api import DockPaneToggleGroup, SMenuBar, \
    SMenu, SToolBar, TaskAction, TaskToggleGroup, SchemaAddition
from traits.api import on_trait_change, Property, Instance

from peppy2.framework.task import FrameworkTask

from project_editor import ProjectEditor
from panes import LayerSelectionPane, LayerInfoPane

class OpenLayerAction(TaskAction):
    name = 'Open Layer...'
    accelerator = 'Ctrl+L'
    tooltip = 'Open a file and add the layer to the current project'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control)
        if dialog.open() == OK:
            event.task.window.application.load_file(dialog.path, event.task, layer=True)

class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """

    #### Task interface #######################################################

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

    def _extra_actions_default(self):
        # FIXME: Is there no way to add an item to an existing group?
        actions = [ SchemaAddition(id='OpenLayer',
                                   factory=OpenLayerAction,
                                   path='MenuBar/File',
                                   after="OpenGroup",
                                   before="OpenGroupEnd",
                                   ),
                    ]
        return actions

    ###########################################################################
    # 'FrameworkTask' interface.
    ###########################################################################

    def get_editor(self, guess=None):
        """ Opens a new empty window
        """
        editor = ProjectEditor()
        return editor

    def new(self, source=None, layer=False, **kwargs):
        """ Opens a new tab, unless we are adding to the existing layers
        
        :param source: optional :class:`FileGuess` or :class:`Editor` instance
        that will load a new file or create a new view of the existing editor,
        respectively.
        """
        if layer and self.active_editor and hasattr(source, 'get_metadata'):
            editor = self.active_editor
            editor.load(source, **kwargs)
            self._active_editor_changed()
            self.activated()
        else:
            FrameworkTask.new(self, source, **kwargs)

    def _active_editor_changed(self):
        tree = self.window.get_dock_pane('maproom.layer_selection_pane')
        if tree is not None and tree.control is not None:
            # We must be in an event handler during trait change callbacks,
            # because we segfault without the GUI.invoke_later (equivalent
            # to wx.CallAfter)
            GUI.invoke_later(tree.control.set_project, self.active_editor)

    ###
    @classmethod
    def can_edit(cls, mime):
        return mime == "application/x-maproom-verdat"
