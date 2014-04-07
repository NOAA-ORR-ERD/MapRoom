""" Skeleton sample task

"""
# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, YES, OK, CANCEL
from pyface.tasks.api import Task, TaskWindow, TaskLayout, PaneItem, IEditor, \
    IEditorAreaPane, EditorAreaPane, Editor, DockPane, HSplitter, VSplitter
from pyface.action.api import Group
from pyface.tasks.action.api import DockPaneToggleGroup, SMenuBar, \
    SMenu, SToolBar, TaskAction, EditorAction, SchemaAddition
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

class BoundingBoxAction(EditorAction):
    name = 'Show Bounding Boxes'
    tooltip = 'Display or hide bounding boxes for each layer'
    style = 'toggle'

    def perform(self, event):
        value = not self.active_editor.control.bounding_boxes_shown
        self.active_editor.control.bounding_boxes_shown = value
        GUI.invoke_later(self.active_editor.control.render)

    @on_trait_change('active_editor')
    def _update_checked(self):
        self.checked = self.active_editor.control.bounding_boxes_shown

class ZoomInAction(EditorAction):
    name = 'Zoom In'
    tooltip = 'Increase magnification'
    image = ImageResource('zoom_in')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_in)

class ZoomOutAction(EditorAction):
    name = 'Zoom Out'
    tooltip = 'Decrease magnification'
    image = ImageResource('zoom_out')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_out)

class ZoomToFit(EditorAction):
    name = 'Zoom to Fit'
    tooltip = 'Set magnification to show all layers'
    image = ImageResource('zoom_fit')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.zoom_to_fit)

class ZoomToLayer(EditorAction):
    name = 'Zoom to Layer'
    tooltip = 'Set magnification to show current layer'
    enabled_name = 'layer_zoomable' # enabled based on state of task.active_editor.dirty
    image = ImageResource('zoom_to_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.zoom_to_selected_layer)

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
        zoomgroup = lambda : Group(ZoomInAction(),
                                   ZoomOutAction(),
                                   ZoomToFit(),
                                   ZoomToLayer(),
                                   id="zoomgroup")
        actions = [ SchemaAddition(id='OpenLayer',
                                   factory=OpenLayerAction,
                                   path='MenuBar/File',
                                   after="OpenGroup",
                                   before="OpenGroupEnd",
                                   ),
                    SchemaAddition(id='bb',
                                   factory=BoundingBoxAction,
                                   path='MenuBar/View',
                                   after="TaskGroupEnd",
                                   ),
                    SchemaAddition(factory=zoomgroup,
                                   path='MenuBar/View',
                                   absolute_position="first",
                                   ),
                    SchemaAddition(factory=zoomgroup,
                                   path='ToolBar',
                                   after="File",
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
        return mime.startswith("image") or mime == "application/x-maproom-verdat"
