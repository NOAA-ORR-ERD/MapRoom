""" Skeleton sample task

"""
# Enthought library imports.
from pyface.api import ImageResource, GUI, FileDialog, YES, OK, CANCEL
from pyface.tasks.api import Task, TaskWindow, TaskLayout, PaneItem, IEditor, \
    IEditorAreaPane, EditorAreaPane, Editor, DockPane, HSplitter, VSplitter
from pyface.action.api import Group, Separator
from pyface.tasks.action.api import DockPaneToggleGroup, SMenuBar, \
    SMenu, SToolBar, TaskAction, EditorAction, SchemaAddition
from traits.api import provides, on_trait_change, Property, Instance, Str, Unicode

from peppy2.framework.task import FrameworkTask
from peppy2.framework.i_about import IAbout

from project_editor import ProjectEditor
from panes import LayerSelectionPane, LayerInfoPane
from layer_control_wx import LayerControl

class OpenLayerAction(TaskAction):
    name = 'Open Layer...'
    accelerator = 'Ctrl+L'
    tooltip = 'Open a file and add the layer to the current project'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control)
        if dialog.open() == OK:
            event.task.window.application.load_file(dialog.path, event.task, layer=True)

class SaveImageAction(EditorAction):
    name = 'Save As Image...'
    tooltip = 'Save a bitmap image of the current view'

    def perform(self, event):
        dialog = FileDialog(parent=event.task.window.control, action='save as')
        if dialog.open() == OK:
            self.active_editor.save_image(dialog.path)

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
        if self.active_editor:
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
    enabled_name = 'layer_zoomable'
    image = ImageResource('zoom_to_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.zoom_to_selected_layer)

class NewLayerAction(EditorAction):
    name = 'New Vector Layer'
    tooltip = 'Create new vector (grid) layer'
    image = ImageResource('add_layer')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_manager.add_layer)

class RaiseLayerAction(EditorAction):
    name = 'Raise Layer'
    tooltip = 'Move layer up in the stacking order'
    enabled_name = 'layer_above'
    image = ImageResource('raise.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.raise_selected_layer)

class LowerLayerAction(EditorAction):
    name = 'Lower Layer'
    tooltip = 'Move layer down in the stacking order'
    enabled_name = 'layer_below'
    image = ImageResource('lower.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.layer_tree_control.lower_selected_layer)

class JumpToCoordsAction(EditorAction):
    name = 'Jump to Coordinates'
    accelerator = 'Ctrl+J'
    tooltip = 'Center the screen on the specified coordinates'
    image = ImageResource('jump.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.do_jump_coords)

class ClearSelectionAction(EditorAction):
    name = 'Clear Selection'
    enabled_name = 'layer_has_selection'
    tooltip = 'Deselects all selected items in the current layer'
    image = ImageResource('clear_selection.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.clear_selection)

class DeleteSelectionAction(EditorAction):
    name = 'Delete Selection'
    accelerator = 'DEL'
    enabled_name = 'layer_has_selection'
    tooltip = 'Deletes the selected items in the current layer'
    image = ImageResource('delete_selection.png')

    def perform(self, event):
        GUI.invoke_later(self.active_editor.delete_selection)

class ZoomModeAction(EditorAction):
    name = 'Zoom Mode'
    tooltip = 'Zoom to box'
    image = ImageResource('zoom_box.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_ZOOM_RECT
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_ZOOM_RECT

class PanModeAction(EditorAction):
    name = 'Pan Mode'
    tooltip = 'Pan the viewport'
    image = ImageResource('pan.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_PAN
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_PAN

class AddPointsAction(EditorAction):
    name = 'Add Points Mode'
    enabled_name = 'layer_has_points'
    tooltip = 'Add points to the current layer'
    image = ImageResource('add_points.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_EDIT_POINTS
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_EDIT_POINTS

class AddLinesAction(EditorAction):
    name = 'Add Lines Mode'
    enabled_name = 'layer_has_points'
    tooltip = 'Add lines to the current layer'
    image = ImageResource('add_lines.png')
    style = 'radio'

    def perform(self, event):
        self.active_editor.mouse_mode = LayerControl.MODE_EDIT_LINES
        self.active_editor.update_layer_selection_ui()

    @on_trait_change('active_editor.mouse_mode')
    def _update_checked(self):
        if self.active_editor:
            self.checked = self.active_editor.mouse_mode == LayerControl.MODE_EDIT_LINES

class FindPointsAction(EditorAction):
    name = 'Find Points'
    accelerator = 'Ctrl+F'
    enabled_name = 'layer_has_points'
    tooltip = 'Find and highlight points or ranges of points'

    def perform(self, event):
        GUI.invoke_later(self.active_editor.control.do_find_points)


@provides(IAbout)
class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """
    
    id = 'maproom.project'
    
    new_file_text = 'Maproom Project'

    #### Task interface #######################################################

    name = 'Maproom Project File'
    
    icon = ImageResource('maproom')
    
    #### 'IAbout' interface ###################################################
    
    about_title = Str('MapRoom')
    
    about_version = Unicode
    
    about_description = Unicode('Edit stuff.')
    
    about_website = Str('http://www.noaa.gov')
    
    about_image = Instance(ImageResource, ImageResource('maproom_large'))
    
    def _about_version_default(self):
        import Version
        return Version.VERSION
    
    def _about_description_default(self):
        desc = "High-performance 2d mapping developed by NOAA\n\nUsing:\n"
        import wx
        desc += "  wxPython %s\n" % wx.version()
        try:
            import gdal
            desc += "  GDAL %s\n" % gdal.VersionInfo()
        except:
            pass
        try:
            import numpy
            desc += "  numpy %s\n" % numpy.version.version
        except:
            pass
        try:
            import OpenGL
            desc += "  PyOpenGL %s\n" % OpenGL.__version__
        except:
            pass
        try:
            import pyproj
            desc += "  PyProj %s\n" % pyproj.__version__
        except:
            pass
        return desc

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

    def _tool_bars_default(self):
        toolbars = [
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction()),
                     show_tool_names=False,
#                     image_size=(22,22),
                     id="BaseLayerToolBar",),
            SToolBar(Group(ZoomModeAction(),
                           PanModeAction(),
                           AddPointsAction(),
                           AddLinesAction()),
                     show_tool_names=False,
                     id="VectorLayerToolBar",),
            ]
        toolbars.extend(FrameworkTask._tool_bars_default(self))
        return toolbars

    def _extra_actions_default(self):
        # FIXME: Is there no way to add an item to an existing group?
        zoomgroup = lambda : Group(ZoomInAction(),
                                   ZoomOutAction(),
                                   ZoomToFit(),
                                   ZoomToLayer(),
                                   id="zoomgroup")
        layer = lambda: SMenu(
            id= 'Layer', name="Layer"
        )
        layertools = lambda : Group(
            NewLayerAction(),
            RaiseLayerAction(),
            LowerLayerAction(),
            id="layertools")
        layermenu = lambda : Group(
            Group(NewLayerAction(),
                  id="addlayergroup"),
            Group(RaiseLayerAction(),
                  LowerLayerAction(),
                  id="raisegroup"),
            Group(ZoomModeAction(),
                  PanModeAction(),
                  AddPointsAction(),
                  AddLinesAction(),
                  id="modegroup"),
            id="layermenu")
        editmenu = lambda: Group(
            Group(ClearSelectionAction(),
                  DeleteSelectionAction(),
                  id="findgroup", separator=False),
            Group(FindPointsAction(),
                  id="findgroup"),
            id="editmenu")
        edittools = lambda : Group(
            ClearSelectionAction(),
            DeleteSelectionAction(),
            id="edittools")
        actions = [
            # Menubar additions
            SchemaAddition(id='OpenLayer',
                           factory=OpenLayerAction,
                           path='MenuBar/File',
                           after="OpenGroup",
                           before="OpenGroupEnd",
                           ),
            SchemaAddition(id='SaveImage',
                           factory=SaveImageAction,
                           path='MenuBar/File',
                           after="SaveGroup",
                           before="SaveGroupEnd",
                           ),
            SchemaAddition(factory=editmenu,
                           path='MenuBar/Edit',
                           before="PrefGroup",
                           ),
            SchemaAddition(id='bb',
                           factory=BoundingBoxAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(id='jump',
                           factory=JumpToCoordsAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(factory=layer,
                           path='MenuBar',
                           after="Edit",
                           ),
            SchemaAddition(factory=layermenu,
                           path='MenuBar/Layer',
                           ),
            SchemaAddition(factory=zoomgroup,
                           path='MenuBar/View',
                           absolute_position="first",
                           ),
            
            # Toolbar additions
            SchemaAddition(id="layer",
                           factory=layertools,
                           path='ToolBar',
                           after="File",
                           ),
            SchemaAddition(id="edit",
                           factory=edittools,
                           path='ToolBar',
                           before="layer",
                           ),
            SchemaAddition(id="zoom",
                           factory=zoomgroup,
                           path='ToolBar',
                           after="layer",
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
    
    @on_trait_change('active_editor.mouse_mode_category')
    def mode_toolbar_changed(self, changed_to):
        for toolbar in self.window.tool_bar_managers:
            name = toolbar.id
            if name == "ToolBar" or name == changed_to:
                state = True
            else:
                state = False
            toolbar.visible = state
            print "toolbar: %s = %s" % (name, state)

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
        return mime.startswith("image") or mime.startswith("application/x-maproom-")
