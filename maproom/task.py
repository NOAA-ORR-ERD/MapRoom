""" Skeleton sample task

"""
import os
import sys

# Enthought library imports.
from pyface.api import ImageResource, GUI, YES, OK, CANCEL
from pyface.action.api import Group, Separator
from pyface.tasks.action.api import SMenuBar, SMenu, SToolBar, SchemaAddition
from traits.api import provides, on_trait_change, Property, Instance, Str, Unicode, Any, List, Event, Dict

from omnivore.framework.task import FrameworkTask
from omnivore.framework.i_about import IAbout

from project_editor import ProjectEditor
import pane_layout
from preferences import MaproomPreferences
from library.mem_use import get_mem_use
import toolbar
from library.thread_utils import BackgroundWMSDownloader
from library.tile_utils import BackgroundTileDownloader

from actions import *
from omnivore.framework.actions import PreferencesAction, CutAction, CopyAction, PasteAction, OpenLogDirectoryAction

import logging
log = logging.getLogger(__name__)


@provides(IAbout)
class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """
    
    id = pane_layout.task_id_with_pane_layout
    
    new_file_text = 'MapRoom Project'
    
    about_application = ""

    #### Task interface #######################################################

    name = 'MapRoom Project File'
    
    icon = ImageResource('maproom')
    
    preferences_helper = MaproomPreferences
    
    status_bar_debug_width = 300
    
    start_new_editor_in_new_window = True
    
    #### 'IAbout' interface ###################################################
    
    about_title = Str('MapRoom')
    
    about_version = Unicode
    
    about_description = Property(Unicode)
    
    about_website = Str('http://www.noaa.gov')
    
    about_image = Instance(ImageResource, ImageResource('maproom_large'))
    
    #### 'IErrorReporter' interface ###########################################
    
    error_email_to = Str('rob.mcmullen@noaa.gov')
    
    #### Menu events ##########################################################
    
    # Layer selection event placed here instead of in the ProjectEditor
    # because the trait events don't seem to be triggered in the
    # menu items on task.active_editor.layer_selection_changed
    # but they are on task.layer_selection_changed.  This means
    # ProjectEditor.update_layer_selection_ui() sets an event here in the
    # MaproomTask rather than in itself.
    layer_selection_changed = Event
    
    def _about_version_default(self):
        import Version
        return Version.VERSION
    
    def _get_about_description(self):
        desc = "High-performance 2d mapping developed by NOAA\n\nMemory usage: %.0fMB\n\nUsing libraries:\n" % get_mem_use()
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
        return pane_layout.pane_layout()

    def create_dock_panes(self):
        """ Create the file browser and connect to its double click event.
        """
        return pane_layout.pane_create()

    def _tool_bars_default(self):
        toolbars = toolbar.get_all_toolbars()
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
            Separator(id="LayerMenuStart", separator=False),
            id= 'Layer', name="Layer"
        )
        layertools = lambda : Group(
            RaiseToTopAction(),
            RaiseLayerAction(),
            LowerLayerAction(),
            LowerToBottomAction(),
            TriangulateLayerAction(),
            DeleteLayerAction(),
            id="layertools")
        layermenu = lambda : Group(
            Separator(id="LayerMainMenuStart", separator=False),
            Group(RaiseToTopAction(),
                  RaiseLayerAction(),
                  LowerLayerAction(),
                  LowerToBottomAction(),
                  id="raisegroup", separator=False),
            Group(TriangulateLayerAction(),
                  ToPolygonLayerAction(),
                  ToVerdatLayerAction(),
                  MergeLayersAction(),
                  MergePointsAction(),
                  id="utilgroup"),
            Group(DeleteLayerAction(),
                  id="deletegroup"),
            Group(CheckLayerErrorAction(),
                  id="checkgroup"),
            id="layermenu")
        edittools = lambda : Group(
            ClearSelectionAction(),
            DeleteSelectionAction(),
            id="edittools")
        actions = [
            # Menubar additions
            SchemaAddition(id='bb',
                           factory=BoundingBoxAction,
                           path='MenuBar/View',
                           after="TaskGroupEnd",
                           ),
            SchemaAddition(id='pfb',
                           factory=PickerFramebufferAction,
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
                           after='New',
                           ),
            SchemaAddition(factory=zoomgroup,
                           path='MenuBar/View',
                           absolute_position="first",
                           ),
            SchemaAddition(id='dal',
                           factory=DebugAnnotationLayersAction,
                           path='MenuBar/Help/Debug',
                           ),
            
            # Toolbar additions
            SchemaAddition(id="layer",
                           factory=layertools,
                           path='ToolBar',
                           after="Undo",
                           ),
            SchemaAddition(id="edit",
                           factory=edittools,
                           path='ToolBar',
                           before="layer",
                           after="Undo",
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
    
    def activated(self):
        FrameworkTask.activated(self)
        visible = pane_layout.pane_initially_visible()
        for pane in self.window.dock_panes:
            if pane.id in visible:
                pane.visible = visible[pane.id]
        
        self.init_threaded_processing()
        
        # This trait can't be set as a decorator on the method because
        # active_editor can be None during the initialization process.  Set
        # here because it's guaranteed not to be None
        self.on_trait_change(self.mode_toolbar_changed, 'active_editor.mouse_mode_toolbar')

    def prepare_destroy(self):
        self.window.application.remember_perspectives(self.window)
        self.stop_threaded_processing()
    
    def get_actions_Menu_File_NewGroup(self):
        return [
            NewProjectAction(),
            NewVectorLayerAction(),
            NewAnnotationLayerAction(),
            NewWMSLayerAction(),
            NewTileLayerAction(),
            NewLonLatLayerAction(),
            ]
    
    def get_actions_Menu_File_SaveGroup(self):
        return [
            SaveProjectAction(),
            SaveProjectAsAction(),
            SaveCommandLogAction(),
            SaveLayerAction(),
            SMenu(SaveLayerGroup(),
                  id='SaveLayerAsSubmenu', name="Save Layer As"),
            SaveImageAction(),
            ]
    
    def get_actions_Menu_Edit_CopyPasteGroup(self):
        return [
            CutAction(),
            CopyAction(),
            DuplicateLayerAction(),
            PasteAction(),
            Separator(),
            CopyStyleAction(),
            PasteStyleAction(),
            ]
    
    def get_actions_Menu_Edit_SelectGroup(self):
        return [
            ClearSelectionAction(),
            DeleteSelectionAction(),
            Separator(),
            BoundaryToSelectionAction(),
            Separator(),
            ClearFlaggedAction(),
            FlaggedToSelectionAction(),
            ]
    
    def get_actions_Menu_Edit_PrefGroup(self):
        return [
            DefaultStyleAction(),
            PreferencesAction(),
            ]
    
    def get_actions_Menu_Edit_FindGroup(self):
        return [
            FindPointsAction(),
            ]
    
    def get_actions_Menu_Help_BugReportGroup(self):
        return [
            OpenLogDirectoryAction(),
            OpenLogAction(),
            ]
    
    def get_actions_Tool_File_SaveGroup(self):
        return [
            SaveProjectAction(),
            SaveProjectAsAction(),
            ]

    def get_editor(self, guess=None):
        """ Opens a new empty window
        """
        editor = ProjectEditor()
        return editor

    def new(self, source=None, **kwargs):
        """Open a maproom file.
        
        If the file is a maproom project, it will open a new tab.
        
        If the file is something that can be added as a layer, it will be added
        to the current project, unless a project doesn't exist in which case
        it will open in a new, empty project.
        
        :param source: optional :class:`FileGuess` or :class:`Editor` instance
        that will load a new file or create a new view of the existing editor,
        respectively.
        """
        log.debug("In new...")
        log.debug(" active editor is: %s"%self.active_editor)
        if hasattr(source, 'document_id'):
            if self.active_editor and not self.active_editor.load_in_new_tab(source.metadata):
                editor = self.active_editor
                editor.load_omnivore_document(source, **kwargs)
                self._active_editor_changed()
            else:
                editor = self.get_editor()
                self.editor_area.add_editor(editor)
                self.editor_area.activate_editor(editor)
                editor.load_omnivore_document(source, **kwargs)
            self.activated()
            self.window.application.successfully_loaded_event = source.metadata.uri
            self.window.application.restore_perspective(self.window, self)
        else:
            FrameworkTask.new(self, source, **kwargs)

    def allow_different_task(self, guess, other_task):
        return self.window.confirm("The (MIME type %s) file\n\n%s\n\ncan't be edited in a MapRoom project.\nOpen a new %s window to edit?" % (guess.metadata.mime, guess.metadata.uri, other_task.new_file_text)) == YES

    def restore_toolbars(self, window):
        # Omnivore framework calls this after every file load, normally to
        # restore the single toolbar per task.  But because MapRoom uses
        # dynamic toolbars based on layer, have to make sure that only the
        # correct layer toolbar is shown
        active_toolbar = self.active_editor.mouse_mode_toolbar
        for toolbar in window.tool_bar_managers:
            name = toolbar.id
            state = (name == "ToolBar" or name == active_toolbar)
            toolbar.visible = state
            info = window._aui_manager.GetPane(name)
            info.Show(state)
        window._aui_manager.Update()
    
# This trait change is set in activated() rather than as a decorator (see above)
#    @on_trait_change('active_editor.mouse_mode_toolbar')
    def mode_toolbar_changed(self, changed_to):
        for toolbar in self.window.tool_bar_managers:
            name = toolbar.id
            if name == "ToolBar" or name == changed_to:
                state = True
            else:
                state = False
            toolbar.visible = state
            log.debug("toolbar: %s = %s" % (name, state))

    def _active_editor_changed(self):
        tree = self.window.get_dock_pane('maproom.layer_selection_pane')
        if tree is not None and tree.control is not None:
            # We must be in an event handler during trait change callbacks,
            # because we segfault without the GUI.invoke_later (equivalent
            # to wx.CallAfter)
            GUI.invoke_later(tree.control.set_project, self.active_editor)
    
    def _wx_on_mousewheel_from_window(self, event):
        if self.active_editor:
            self.active_editor.layer_canvas.on_mouse_wheel_scroll(event)
    
    @on_trait_change('window.application.preferences_changed_event')
    def preferences_changed(self, evt):
        if self.active_editor:
            self.active_editor.refresh()

    ###
    @classmethod
    def can_edit(cls, document):
        mime = document.metadata.mime
        return ( mime.startswith("image") or
                 mime.startswith("application/x-maproom-") or
                 mime == "application/x-nc_ugrid" or
                 mime == "application/x-nc_particles"
                 )
    
    @classmethod
    def get_match_score(cls, document):
        if cls.can_edit(document):
            return 10
        return 0


    ##### WMS and Tile processing

    # Traits
    downloaders = Dict
    
    # class attributes
    
    wms_extra_loaded = False
    
    tile_extra_loaded = None
    
    @classmethod
    def init_extra_servers(cls, application):
        if cls.wms_extra_loaded is False:
            # try once
            cls.wms_extra_loaded = True
            try:
                wms_list = application.get_json_data("wms_list")
                BackgroundWMSDownloader.set_known_wms(wms_list)
            except IOError:
                # file not found
                pass
            except ValueError:
                # bad JSON format
                log.error("Invalid format of WMS saved data")
                raise
    
    def remember_wms(self, host=None):
        if host is not None:
            BackgroundWMSDownloader.add_wms_host(host)
        wms_list = BackgroundWMSDownloader.get_known_wms()
        self.window.application.save_json_data("wms_list", wms_list)

    def init_threaded_processing(self):
        self.init_extra_servers(self.window.application)
#        if "OpenStreetMap Test" not in self.get_known_wms_names():
#            BackgroundWMSDownloader.add_wms("OpenStreetMap Test", "http://ows.terrestris.de/osm/service?", "1.1.1")
#            self.remember_wms()
    
    def stop_threaded_processing(self):
        log.debug("Stopping threaded services...")
        while len(self.downloaders) > 0:
            url, wms = self.downloaders.popitem()
            log.debug("Stopping threaded downloader %s" % wms)
            wms = None

    def get_threaded_wms(self, wmshost=None):
        if wmshost is None:
            wmshost = BackgroundWMSDownloader.get_known_wms()[0]
        if wmshost.url not in self.downloaders:
            wms = BackgroundWMSDownloader(wmshost)
            self.downloaders[wmshost.url] = wms
        return self.downloaders[wmshost.url]

    def get_threaded_wms_by_id(self, id):
        wmshost = BackgroundWMSDownloader.get_known_wms()[id]
        return self.get_threaded_wms(wmshost)

    def get_known_wms_names(self):
        return [s.name for s in BackgroundWMSDownloader.get_known_wms()]

    def get_threaded_tile_server(self, tilehost=None):
        if tilehost is None:
            tilehost = BackgroundTileDownloader.get_known_tile_server()[0]
        if tilehost not in self.downloaders:
            cache_dir = os.path.join(self.window.application.cache_dir, "tiles")
            ts = BackgroundTileDownloader(tilehost, cache_dir)
            self.downloaders[tilehost] = ts
        return self.downloaders[tilehost]

    def get_threaded_tile_server_by_id(self, id):
        tilehost = BackgroundTileDownloader.get_known_tile_server()[id]
        return self.get_threaded_tile_server(tilehost)

    def get_known_tile_server_names(self):
        return [s.name for s in BackgroundTileDownloader.get_known_tile_server()]
