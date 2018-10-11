""" Skeleton sample task

"""
import os

# Enthought library imports.
from pyface.api import GUI
from pyface.api import ImageResource
from pyface.action.api import Separator
from pyface.tasks.action.api import SMenu
from pyface.tasks.action.api import SchemaAddition
from traits.api import Dict
from traits.api import Event
from traits.api import Instance
from traits.api import Property
from traits.api import Str
from traits.api import Unicode
from traits.api import on_trait_change
from traits.api import provides

from omnivore_framework.framework.task import FrameworkTask
from omnivore_framework.framework.i_about import IAbout

from .project_editor import ProjectEditor
from . import pane_layout
from .preferences import MaproomPreferences
from .library.mem_use import get_mem_use
from . import toolbar
from .library.thread_utils import BackgroundWMSDownloader
from .library.tile_utils import BackgroundTileDownloader
from .library.known_hosts import default_wms_hosts, default_tile_hosts
from .layers import LayerStyle, parse_styles_from_json, styles_to_json

from . import actions
from omnivore_framework.framework.actions import PreferencesAction, CutAction, CopyAction, PasteAction, OpenLogDirectoryAction, SaveAsImageAction

import logging
log = logging.getLogger(__name__)


@provides(IAbout)
class MaproomProjectTask(FrameworkTask):
    """The Maproom Project File editor task.
    """

    id = pane_layout.task_id_with_pane_layout

    new_file_text = 'MapRoom Project'

    about_application = "template://default_project.maproom"

    # Task interface #######################################################

    name = 'MapRoom Project File'

    icon = ImageResource('maproom')

    preferences_helper = MaproomPreferences

    status_bar_debug_width = 300

    start_new_editor_in_new_window = True

    # 'IAbout' interface ###################################################

    about_title = Str('MapRoom')

    about_version = Unicode

    about_description = Property(Unicode)

    about_website = Str('http://www.noaa.gov')

    about_image = Instance(ImageResource, ImageResource('maproom_large'))

    # 'IErrorReporter' interface ###########################################

    error_email_to = Str('rob.mcmullen@noaa.gov')

    # Menu events ##########################################################

    # Layer selection event placed here instead of in the ProjectEditor
    # because the trait events don't seem to be triggered in the
    # menu items on task.active_editor.layer_selection_changed
    # but they are on task.layer_selection_changed.  This means
    # ProjectEditor.update_layer_selection_ui() sets an event here in the
    # MaproomTask rather than in itself.
    layer_selection_changed = Event

    templates_changed = Event

    def _about_version_default(self):
        from . import Version
        return Version.VERSION

    def _get_about_description(self):
        import sys
        major, minor, micro = sys.version_info[0:3]
        desc = "High-performance 2d mapping developed by NOAA\n\nMemory usage: %.0fMB\n\nPython %d.%d.%d using libraries:\n" % (get_mem_use(), major, minor, micro)
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
            import OpenGL.GL as gl
            desc += "  PyOpenGL %s\n" % OpenGL.__version__
            desc += "  OpenGL %s\n" % gl.glGetString(gl.GL_VERSION).encode('utf-8')
            desc += "  OpenGL Vendor: %s\n" % gl.glGetString(gl.GL_VENDOR).encode('utf-8')
            desc += "  OpenGL Renderer: %s\n" % gl.glGetString(gl.GL_RENDERER).encode('utf-8')
            desc += "  GLSL primary: %s\n" % gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).encode('utf-8')
            num_glsl = gl.glGetInteger(gl.GL_NUM_SHADING_LANGUAGE_VERSIONS)
            desc += "  GLSL supported: "
            for i in range(num_glsl):
                v = gl.glGetStringi(gl.GL_SHADING_LANGUAGE_VERSION, i).encode('utf-8')
                desc += v + ", "
            desc += "\n"
        except:
            pass
        try:
            import pyproj
            desc += "  PyProj %s\n" % pyproj.__version__
        except:
            pass
        try:
            import netCDF4
            desc += "  netCDF4 %s\n" % netCDF4.__version__
        except:
            pass
        try:
            import shapely
            desc += "  Shapely %s\n" % shapely.__version__
        except:
            pass
        try:
            import omnivore
            desc += "  Omnivore %s\n" % omnivore_framework.__version__
        except:
            pass
        return desc

    ###########################################################################
    # 'Task' interface.
    ###########################################################################

    def _tool_bars_default(self):
        toolbars = toolbar.get_all_toolbars()
        toolbars.extend(FrameworkTask._tool_bars_default(self))
        return toolbars

    def _extra_actions_default(self):
        layer_menu = self.create_menu("Menu", "Layer", "LayerCreateGroup", "LayerStackGroup", "LayerUtilGroup", "LayerDeleteGroup", "LayerCheckGroup")
        tools_menu = self.create_menu("Menu", "Tools", "ToolsActionGroup", "ToolsManageGroup")
        additions = [
            # Menubar additions
            SchemaAddition(factory=lambda: layer_menu,
                           path='MenuBar',
                           after="Edit",
                           ),
            SchemaAddition(factory=lambda: tools_menu,
                           path='MenuBar',
                           after="Edit",
                           ),
            SchemaAddition(factory=actions.DebugAnnotationLayersAction,
                           path='MenuBar/Help/Debug'),
        ]
        return additions

    ###########################################################################
    # 'FrameworkTask' interface.
    ###########################################################################

    def activated(self):
        FrameworkTask.activated(self)

        self.one_time_init_driver()

        # This trait can't be set as a decorator on the method because
        # active_editor can be None during the initialization process.  Set
        # here because it's guaranteed not to be None
        self.on_trait_change(self.mode_toolbar_changed, 'active_editor.mouse_mode_toolbar')

    def prepare_destroy(self):
        self.on_trait_change(self.mode_toolbar_changed, 'active_editor.mouse_mode_toolbar', remove=True)
        self.window.application.remember_perspectives(self.window)
        self.stop_threaded_processing()

    def get_actions_Menu_File_NewGroup(self):
        return [
            actions.NewProjectAction(),
            actions.NewEmptyProjectAction(),
            SMenu(actions.LoadProjectTemplateGroup(),
                  id='LoadProjectTemplate', name="New Project From Template"),
        ]

    def get_actions_Menu_File_SaveGroup(self):
        return [
            actions.SaveProjectAction(),
            actions.SaveProjectAsAction(),
            actions.SaveProjectTemplateAction(),
            actions.SaveCommandLogAction(),
            actions.SaveLayerAction(),
            SMenu(actions.SaveLayerGroup(),
                  id='SaveLayerAsSubmenu', name="Save Layer As"),
            SaveAsImageAction(),
            actions.SaveMovieAction(),
        ]

    def get_actions_Menu_File_RevertGroup(self):
        return [
            actions.RevertProjectAction(),
        ]

    def get_actions_Menu_Edit_CopyPasteGroup(self):
        return [
            CutAction(),
            CopyAction(),
            actions.DuplicateLayerAction(),
            PasteAction(),
            Separator(),
            actions.CopyStyleAction(),
            actions.PasteStyleAction(),
        ]

    def get_actions_Menu_Edit_SelectGroup(self):
        return [
            actions.ClearSelectionAction(),
            actions.DeleteSelectionAction(),
            Separator(),
            actions.BoundaryToSelectionAction(),
            Separator(),
            actions.ClearFlaggedAction(),
            actions.FlaggedToSelectionAction(),
        ]

    def get_actions_Menu_Edit_PrefGroup(self):
        return [
            actions.DefaultStyleAction(),
            PreferencesAction(),
        ]

    def get_actions_Menu_Edit_FindGroup(self):
        return [
            actions.FindPointsAction(),
        ]

    def get_actions_Menu_Help_BugReportGroup(self):
        return [
            OpenLogDirectoryAction(),
            actions.OpenLogAction(),
        ]

    def get_actions_Menu_View_ZoomGroup(self):
        return [
            actions.ZoomInAction(),
            actions.ZoomOutAction(),
            actions.ZoomToFit(),
            actions.ZoomToLayer(),
        ]

    def get_actions_Menu_View_ChangeGroup(self):
        return [
            actions.JumpToCoordsAction(),
        ]

    def get_actions_Menu_View_DebugGroup(self):
        return [
            actions.BoundingBoxAction(),
            actions.PickerFramebufferAction(),
        ]

    def get_actions_Menu_Layer_LayerCreateGroup(self):
        return [
            actions.NewVectorLayerAction(),
            actions.NewAnnotationLayerAction(),
            actions.NewShapefileLayerAction(),
            actions.NewWMSLayerAction(),
            actions.NewTileLayerAction(),
            actions.NewCompassRoseLayerAction(),
            actions.NewTimestampLayerAction(),
            actions.NewLonLatLayerAction(),
            actions.NewRNCLayerAction(),
            actions.NewRNCLayer360Action(),
        ]

    def get_actions_Menu_Layer_LayerStackGroup(self):
        return [
            actions.RaiseToTopAction(),
            actions.RaiseLayerAction(),
            actions.LowerLayerAction(),
            actions.LowerToBottomAction(),
        ]

    def get_actions_Menu_Layer_LayerUtilGroup(self):
        return [
            actions.TriangulateLayerAction(),
            actions.ToPolygonLayerAction(),
            actions.ToVerdatLayerAction(),
            actions.MergeLayersAction(),
            actions.MergePointsAction(),
        ]

    def get_actions_Menu_Layer_LayerDeleteGroup(self):
        return [
            actions.DeleteLayerAction(),
        ]

    def get_actions_Menu_Layer_LayerCheckGroup(self):
        return [
            actions.CheckSelectedLayerAction(),
            actions.CheckAllLayersAction(),
        ]

    def get_actions_Menu_Tools_ToolsActionGroup(self):
        return [
            actions.NormalizeLongitudeAction(),
            actions.SwapLatLonAction(),
#            actions.TimelineAction(),
        ]

    def get_actions_Menu_Tools_ToolsManageGroup(self):
        return [
            actions.ManageWMSAction(),
            Separator(),
            actions.ManageTileServersAction(),
            actions.ClearTileCacheAction(),
        ]

    def get_actions_Tool_File_SaveGroup(self):
        return [
            actions.SaveProjectAction(),
            actions.SaveProjectAsAction(),
        ]

    def get_actions_Tool_Edit_SelectGroup(self):
        return [
            actions.ClearSelectionAction(),
            actions.DeleteSelectionAction(),
        ]

    def get_actions_Tool_View_ConfigGroup(self):
        return [
            actions.ZoomInAction(),
            actions.ZoomOutAction(),
            actions.ZoomToFit(),
            actions.ZoomToLayer(),
        ]

    def get_actions_Tool_View_ChangeGroup(self):
        return [
            actions.RaiseToTopAction(),
            actions.RaiseLayerAction(),
            actions.LowerLayerAction(),
            actions.LowerToBottomAction(),
            actions.GroupLayerAction(),
            actions.UngroupLayerAction(),
            actions.TriangulateLayerAction(),
            actions.DeleteLayerAction(),
        ]

    def get_actions_Menu_Help_DebugTaskGroup(self):
        return [
            actions.DebugLayerManagerAction(),
            ]

    def get_editor(self, guess=None, **kwargs):
        """ Opens a new empty window
        """
        editor = ProjectEditor()
        return editor

    def new(self, source=None, window_opening=False, **kwargs):
        """Open a maproom file.
        
        If the file is a maproom project, it will open a new tab.
        
        If the file is something that can be added as a layer, it will be added
        to the current project, unless a project doesn't exist in which case
        it will open in a new, empty project.
        
        :param source: optional :class:`FileGuess` or :class:`Editor` instance
        that will load a new file or create a new view of the existing editor,
        respectively.
        """
        log.debug("new: source=%s active editor=%s, kwargs=%s" % (source, self.active_editor, str(kwargs)))
        if hasattr(source, 'document_id'):
            if self.active_editor and not self.active_editor.load_in_new_tab(source.metadata):
                editor = self.active_editor
                editor.load_omnivore_document(source, **kwargs)
                #self._active_editor_changed()  # FIXME: unneeded now?
            else:
                editor = self.get_editor()
                self.editor_area.add_editor(editor)
                self.editor_area.activate_editor(editor)
                editor.load_omnivore_document(source, **kwargs)

                # Only restore perspective on a new window, otherwise it causes
                # an unnecessary update and incorrectly replaces existing pane
                # names
                self.window.application.restore_perspective(self.window, self)
            self.activated()
            self.window.application.successfully_loaded_event = source.metadata.uri
        elif not window_opening:
            log.debug("starting empty task")
            FrameworkTask.new(self, source, **kwargs)

    def allow_different_task(self, guess, other_task):
        return self.confirm("The (MIME type %s) file\n\n%s\n\ncan't be edited in a MapRoom project.\nOpen a new %s window to edit?" % (guess.metadata.mime, guess.metadata.uri, other_task.new_file_text))

    def restore_toolbars(self, window):
        # Omnivore framework calls this after every file load, normally to
        # restore the single toolbar per task.  But because MapRoom uses
        # dynamic toolbars based on layer, have to make sure that only the
        # correct layer toolbar is shown
        if self.active_editor is None:
            return
        active_toolbar = self.active_editor.mouse_mode_toolbar
        for toolbar in window.tool_bar_managers:
            name = toolbar.id
            state = (name == "%s:ToolBar" % self.id or name == active_toolbar)
            toolbar.visible = state
            info = window._aui_manager.GetPane(name)
            info.Show(state)
        window._aui_manager.Update()

# This trait change is set in activated() rather than as a decorator (see above)
#    @on_trait_change('active_editor.mouse_mode_toolbar')
    def mode_toolbar_changed(self, changed_to):
        for toolbar in self.window.tool_bar_managers:
            name = toolbar.id
            if name == "%s:ToolBar" % self.id or name == changed_to:
                state = True
            else:
                state = False
            toolbar.visible = state
            log.debug("toolbar: %s = %s" % (name, state))

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
        return (mime.startswith("image") or
                mime.startswith("application/x-maproom-") or
                mime == "application/x-nc_ugrid" or
                mime == "application/x-nc_particles" or
                mime == "text/latlon" or
                mime == "text/lonlat"
                )

    @classmethod
    def get_match_score(cls, document):
        if cls.can_edit(document):
            return 10
        return 0

    # WMS and Tile processing

    # Traits
    downloaders = Dict

    # class attributes

    extra_json_loaded = False

    _fallback_styles = {
        "other": LayerStyle(),
        "ui": LayerStyle(line_stipple=0xaaaa, line_width=1, line_color=LayerStyle.default_line_color, fill_style=0),
        "triangle": LayerStyle(line_width=1),
    }

    _default_styles = {}

    @property
    def default_styles(self):
        d = {}
        for type_name, style in self._default_styles.items():
            d[type_name] = style.get_copy()
        return d

    def default_styles_read_only(self, type_name):
        return self._default_styles.get(type_name, self._default_styles["other"])

    @classmethod
    def replace_default_styles(cls, styles):
        if styles:
            cls._default_styles = styles
        else:
            cls._default_styles = {}
        for type_name, style in cls._fallback_styles.items():
            if type_name not in cls._default_styles:
                cls._default_styles[type_name] = style.get_copy()

    @classmethod
    def override_default_styles(cls, styles):
        if styles:
            for type_name, style in styles.items():
                cls._default_styles[type_name] = style.get_copy()

    def one_time_init_driver(self):
        self.init_extra_json(self.window.application)

    @classmethod
    def init_extra_json(cls, application):
        if cls.extra_json_loaded is False:
            # try once
            cls.extra_json_loaded = True

            hosts = application.get_json_data("wms_servers")
            if hosts is None:
                hosts = default_wms_hosts
            BackgroundWMSDownloader.set_known_hosts(hosts)

            hosts = application.get_json_data("tile_servers")
            if hosts is None:
                hosts = default_tile_hosts
            BackgroundTileDownloader.set_known_hosts(hosts)

            data = application.get_json_data("styles")
            if data is not None:
                styles = parse_styles_from_json(data)
            else:
                styles = None
            cls.replace_default_styles(styles)

    def remember_styles(self, override_styles=None):
        self.override_default_styles(override_styles)
        data = styles_to_json(self._default_styles)
        self.window.application.save_json_data("styles", data)

    def remember_wms(self, host=None):
        if host is not None:
            BackgroundWMSDownloader.add_wms_host(host)
        hosts = BackgroundWMSDownloader.get_known_hosts()
        self.window.application.save_json_data("wms_servers", hosts)

    def stop_threaded_processing(self):
        log.debug("Stopping threaded services...")
        while len(self.downloaders) > 0:
            url, wms = self.downloaders.popitem()
            log.debug("Stopping threaded downloader %s" % wms)
            wms.stop_threads()
        log.debug("Stopped threaded services.")

        import threading
        for thread in threading.enumerate():
            log.debug("thread running: %s" % thread.name)

    def get_threaded_wms(self, host=None):
        if host is None:
            host = BackgroundWMSDownloader.get_known_hosts()[0]
        if host.url not in self.downloaders:
            wms = BackgroundWMSDownloader(host)
            self.downloaders[host.url] = wms
        return self.downloaders[host.url]

    def get_wms_server_by_id(self, id):
        host = BackgroundWMSDownloader.get_known_hosts()[id]
        return host

    def get_wms_server_id_from_url(self, url):
        index, host = BackgroundWMSDownloader.get_host_by_url(url)
        return index

    def get_threaded_wms_by_id(self, id):
        host = self.get_wms_server_by_id(id)
        return self.get_threaded_wms(host)

    def get_known_wms_names(self):
        return [s.name for s in BackgroundWMSDownloader.get_known_hosts()]

    def get_default_wms_id(self):
        index, host = BackgroundWMSDownloader.get_default_host()
        return index

    def remember_tile_servers(self, host=None):
        if host is not None:
            BackgroundTileDownloader.add_wms_host(host)
        hosts = BackgroundTileDownloader.get_known_hosts()
        self.window.application.save_json_data("tile_servers", hosts)

    def get_tile_cache_root(self):
        return os.path.join(self.window.application.cache_dir, "tiles")

    def get_tile_downloader(self, host=None):
        if host is None:
            host = BackgroundTileDownloader.get_known_hosts()[0]
        if host not in self.downloaders:
            cache_dir = self.get_tile_cache_root()
            ts = BackgroundTileDownloader(host, cache_dir)
            self.downloaders[host] = ts
        return self.downloaders[host]

    def get_tile_downloader_by_id(self, id):
        host = self.get_tile_server_by_id(id)
        return self.get_tile_downloader(host)

    def get_tile_server_by_id(self, id):
        host = BackgroundTileDownloader.get_known_hosts()[id]
        return host

    def get_tile_server_id_from_url(self, url):
        index, host = BackgroundTileDownloader.get_host_by_url(url)
        return index

    def get_known_tile_server_names(self):
        return [s.name for s in BackgroundTileDownloader.get_known_hosts()]

    def get_default_tile_server_id(self):
        index, host = BackgroundTileDownloader.get_default_host()
        return index
