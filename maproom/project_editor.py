# Standard library imports.
import sys
import os.path
import time
import json

# Major package imports.
import wx

# Enthought library imports.
from pyface.api import FileDialog, OK
from traits.api import Any
from traits.api import Bool
from traits.api import Dict
from traits.api import Float
from traits.api import Str
from traits.api import List
from traits.api import on_trait_change

from omnivore_framework.framework.editor import FrameworkEditor
from omnivore_framework.framework.errors import ProgressCancelError
import omnivore_framework.framework.clipboard as clipboard
from omnivore_framework.utils.wx.popuputil import PopupStatusBar
from omnivore_framework.utils.wx.tilemanager import TileManager
from omnivore_framework.templates import get_template

# Local imports.
from .errors import MapRoomError
from .layer_canvas import LayerCanvas
from .layer_manager import LayerManager
from . import renderer
from .layers import loaders
from .command import UndoStack, BatchStatus
from . import mouse_handler
from . import menu_commands as mec
from . import mouse_commands as moc
from . import toolbar
from .library.bsb_utils import extract_from_zip
from .library import apng
from . import panes
from .layer_tree_control import LayerTreeControl
from .ui.info_panels import LayerInfoPanel, SelectionInfoPanel
from .ui.triangle_panel import TrianglePanel
from .ui.merge_panel import MergePointsPanel
from .ui.undo_panel import UndoHistoryPanel

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class ProjectEditor(FrameworkEditor):
    """The wx implementation of a ProjectEditor.

    See the IProjectEditor interface for the API documentation.
    """

    printable = True

    imageable = True

    layer_manager = Any

    layer_zoomable = Bool

    layer_can_save = Bool

    layer_can_save_as = Bool

    layer_selected = Bool

    layer_above = Bool

    layer_below = Bool

    multiple_layers = Bool

    layer_has_points = Bool

    layer_has_selection = Bool

    layer_has_flagged = Bool

    layer_has_boundaries = Bool

    layer_is_groupable = Bool

    clickable_object_mouse_is_over = Any

    clickable_object_in_layer = Any

    last_refresh = Float(0.0)

    layer_visibility = Dict()

    # temporary variable set in load process and used in setting the initial
    # view
    loaded_project_extra_json = Any

    # can_paste_style is set by copy_style if there's a style that can be
    # applied
    can_paste_style = Bool(False)

    latest_movie = Any(False)

    # NOTE: Class attribute!
    clipboard_style = None

    # Force mouse mode toolbar to be blank so that the initial trait change
    # that occurs during initialization of this class doesn't match a real
    # mouse mode.  If it does match, the toolbar won't be properly adjusted
    # during the first trait change in response to update_layer_selection_ui
    # and there will be an empty between named toolbars
    mouse_mode_toolbar = Str("")

    mouse_mode_factory = Any(mouse_handler.SelectionMode)

    ###########################################################################
    # 'FrameworkEditor' interface.
    ###########################################################################

    def create(self, parent):
        self.control = self._create_control(parent)

    def init_blank_document(self, doc, **kwargs):
        # This gets called only when a new editor is created without a document
        # and is the only place to select an initial layer in this case.
        wx.CallAfter(self.layer_tree_control.select_initial_layer)

    def load_in_new_tab(self, metadata):
        # a default (blank) document is always created when opening a new
        # toplevel window. If this happens during app init, allow the initial
        # notebook tab to be replaced with the command line project or the
        # default project.
        if metadata.mime.startswith("application/x-maproom-project-"):
            return self.window.application.application_initialization_finished
        return False

    def load_omnivore_document(self, document, layer=False, **kwargs):
        """ Loads the data from the Omnivore document
        """
        metadata = document.metadata
        loader = loaders.get_loader(metadata)
        regime = kwargs.get("regime", 0)
        log.debug("load_omnivore_document: doc=%s loader=%s kwargs=%s" % (document, loader, str(kwargs)))
        if hasattr(loader, "load_project"):
            document = LayerManager.create(self)
            document.metadata = metadata.clone_traits()
            batch_flags = BatchStatus()
            # FIXME: Add load project command that clears all layers
            extra = loader.load_project(metadata, document, batch_flags)
            self.document = self.layer_manager = document
            self.create_layout(extra)
            self.parse_extra_json(extra, batch_flags)
            self.loaded_project_extra_json = extra
            log.debug("Clearing timeline")
            self.timeline.clear_marks()
            self.layer_tree_control.clear_all_items()
            self.layer_tree_control.rebuild()
            self.layer_tree_control.select_initial_layer()
            self.perform_batch_flags(None, batch_flags)

            # Clear modified flag
            self.layer_manager.undo_stack.set_save_point()
            self.dirty = self.layer_manager.undo_stack.is_dirty()
            self.mouse_mode_factory = mouse_handler.PanMode
            self.view_document(self.document)
        elif hasattr(loader, "iter_log"):
            line = 0
            batch_flags = BatchStatus()
            for cmd in loader.iter_log(metadata, self.layer_manager):
                line += 1
                errors = None
                try:
                    undo = self.process_batch_command(cmd, batch_flags)
                    if not undo.flags.success:
                        errors = undo.errors
                        break
                except Exception as e:
                    errors = [str(e)]
                    import traceback
                    print(traceback.format_exc())
                    break
            if errors is not None:
                header = [
                    "While restoring from the command log file:\n\n%s\n" % metadata.uri,
                    "an error occurred on line %d while processing" % line,
                    "the command '%s':" % cmd.short_name,
                    ""
                ]
                header.extend(errors)
                text = "\n".join(header)
                self.task.error(text, "Error restoring from command log")
            self.perform_batch_flags(None, batch_flags)
            self.view_document(self.document)
        else:
            if not hasattr(self, 'layer_tree_control'):
                self.create_layout({})
            log.debug("loading %s" % metadata)
            cmd = mec.LoadLayersCommand(metadata, regime)
            self.process_command(cmd)
            if 'regime' not in kwargs:
                layers = cmd.undo_info.affected_layers()
                if len(layers) == 1:
                    cmd = moc.ViewportCommand(layers[0])
                else:
                    center, units_per_pixel = self.layer_canvas.calc_zoom_to_layers(layers)
                    cmd = moc.ViewportCommand(None, center, units_per_pixel)
                self.process_command(cmd)
        document.read_only = document.metadata.check_read_only()

    def parse_extra_json(self, json, batch_flags):
        if json is None:
            json = {}

        # handle old version which was a two element list
        try:
            if len(json) == 2 and json[0] == "commands":
                json = {json[0]: json[1]}
        except KeyError:
            # a two element dict won't have a key of zero
            pass

        if "layer_visibility" in json:
            self.layer_visibility_from_json(json["layer_visibility"])

            # Just as a sanity check, make sure all layers have visibility
            # info. Very old json formats may not include visibility info for
            # all layers
            self.update_default_visibility(warn=True)
        else:
            self.layer_visibility = self.get_default_visibility()
        if "commands" in json:
            for cmd in self.layer_manager.undo_stack.unserialize_text(json["commands"], self.layer_manager):
                self.process_batch_command(cmd, batch_flags)
        if "timeline" in json:
            self.timeline.unserialize_json(json["timeline"])

    @property
    def current_extra_json(self):
        return {
            "layer_visibility": self.layer_visibility_to_json(),
            "projected_point_center": self.layer_canvas.projected_point_center,
            "projected_units_per_pixel": self.layer_canvas.projected_units_per_pixel,
            "timeline": self.timeline.serialize_json(),
            "tile_manager": self.control.calc_layout()["tile_manager"]
            }

    def layer_visibility_to_json(self):
        v = dict()
        for layer, vis in self.layer_visibility.items():
            v[layer.invariant] = vis
        return v

    def layer_visibility_from_json(self, json_data):
        lm = self.layer_manager
        v = dict()
        for invariant, vis in json_data.items():
            layer = lm.get_layer_by_invariant(int(invariant))
            if layer is not None:
                # Some layer visibility data from deleted layers may have been
                # saved, so only restore visibility for layers that are known
                v[layer] = vis
        self.layer_visibility = v

    def get_default_visibility(self):
        layer_visibility = dict()
        for layer in self.layer_manager.flatten():
            layer_visibility[layer] = layer.get_visibility_dict()
        return layer_visibility

    def update_default_visibility(self, warn=False):
        for layer in self.layer_manager.flatten():
            if layer not in self.layer_visibility:
                if warn:
                    log.warning("adding default visibility for layer %s" % layer)
                self.layer_visibility[layer] = layer.get_visibility_dict()

    def set_layer_visibility(self, visible_layers, layers=None):
        if layers is None:
            layers = self.layer_manager.flatten()
        for layer in layers:
            if layer.skip_on_insert:
                # skip static layers like scale and grid
                continue
            state = layer in visible_layers
            self.layer_visibility[layer]['layer'] = state
        self.layer_tree_control.update_checked_from_visibility()
        self.refresh()

    def is_layer_visible_at_current_time(self, layer):
        t = self.timeline.current_time
        if t is None:
            begin, end = self.timeline.selected_time_range
            return layer.is_visible_in_time_range(begin, end)
        else:
            return layer.is_visible_at_time(t)

    def rebuild_document_properties(self):
        self.layer_manager = self.document
        self.update_default_visibility()
        self.timeline.recalc_view()

    def init_view_properties(self):
        # Set default view
        self.layer_canvas.zoom_to_fit()

        # Override default view if provided in project file
        json = self.loaded_project_extra_json
        if json is not None:
            if "projected_units_per_pixel" in json:
                self.layer_canvas.set_units_per_pixel(json["projected_units_per_pixel"])
            if "projected_point_center" in json:
                self.layer_canvas.set_center(json["projected_point_center"])
        self.loaded_project_extra_json = None
        log.debug("using center: %s, upp=%f" % (str(self.layer_canvas.projected_point_center), self.layer_canvas.projected_units_per_pixel))

    def save(self, path=None, prompt=False):
        """ Saves the contents of the editor in a maproom project file
        """
        if path is None:
            path = self.document.uri
        if prompt or not path:
            default_dir = self.best_file_save_dir
            default_file = ""
            dialog = FileDialog(parent=self.window.control, action='save as', wildcard="MapRoom Project Files (*.maproom)|*.maproom", default_directory=default_dir, default_filename=default_file)
            if dialog.open() == OK:
                path = dialog.path
            else:
                return
        if not path:
            path = "%s.maproom" % self.name

        prefs = self.task.preferences
        if prefs.check_errors_on_save:
            if not self.check_all_layers_for_errors(True):
                return

        error = self.save_project(path)
        if error:
            self.task.error(error)
        else:
            self.layer_manager.undo_stack.set_save_point()

            # Update tab name.  Note that dirty must be changed in order for
            # the trait to be updated, so force a change if needed.  Also,
            # update the URI first because trait callbacks happen immediately
            # and because properties are used for the editor name, no trait
            # event gets called on updating the metadata URI.
            self.layer_manager.metadata.uri = path
            if not self.dirty:
                self.dirty = True
            self.dirty = self.layer_manager.undo_stack.is_dirty()

            # Push URL to top of recent files list
            self.window.application.successfully_saved_event = self.layer_manager.metadata.uri

            # refresh window name in case filename has changed
            self.task._active_editor_tab_change(None)

    def save_project(self, path):
        try:
            progress_log.info("START=Saving %s" % path)
            error = self.layer_manager.save_all(path, self.current_extra_json)
        except ProgressCancelError as e:
            error = str(e)
        finally:
            progress_log.info("END")

        return error

    def save_as_template(self, name):
        path = self.window.application.get_user_dir_filename("project_templates", name)
        if not os.path.exists(path) or self.task.confirm("Replace existing template %s?" % name, "Replace Template"):
            error = self.save_project(path)
            if error:
                self.task.error(error)
            else:
                self.task.templates_changed = True  # update template submenu

    def get_savepoint(self):
        layer = self.layer_tree_control.get_edit_layer()
        cmd = mec.SavepointCommand(layer, self.layer_canvas.get_zoom_rect())
        return cmd

    def save_log(self, path):
        """ Saves the command log to a text file
        """
        # Add temporary mec.SavepointCommand to command history so that it can be
        # serialized, but remove it after seriarization so it doesn't clutter
        # the history
        cmd = self.get_savepoint()
        self.layer_manager.undo_stack.add_command(cmd)
        serializer = self.layer_manager.undo_stack.serialize()
        try:
            fh = open(path, "wb")
            fh.write(str(serializer))
            fh.close()
        except IOError as e:
            self.task.error(str(e))
        self.layer_manager.undo_stack.pop_command()

    def save_layer(self, path, loader=None):
        """ Saves the contents of the current layer in an appropriate file
        """
        layer = self.layer_tree_control.get_edit_layer()
        if layer is None:
            return

        prefs = self.task.preferences
        if prefs.check_errors_on_save:
            if not self.check_all_layers_for_errors(True):
                return

        try:
            progress_log.info("START")
            error = self.layer_manager.save_layer(layer, path, loader)
        except ProgressCancelError as e:
            error = str(e)
        finally:
            progress_log.info("END")
        if error:
            self.task.error(error)
        else:
            self.window.application.successfully_loaded_event = path
        self.layer_metadata_changed(layer)
        self.update_layer_selection_ui()

    def get_numpy_image(self):
        # Deselect all layers because it's designed to be used as post-
        # processing image
        self.layer_tree_control.set_edit_layer(None)
        self.layer_canvas.render_callback(immediately=True)  # force update including deselected layer
        return self.layer_canvas.get_canvas_as_image()

    def print_preview(self):
        import os
        temp = os.path.join(self.window.application.cache_dir, "preview.pdf")
        self.save_as_pdf(temp)
        try:
            os.startfile(temp)
        except AttributeError:
            import subprocess
            if sys.platform == "darwin":
                file_manager = '/usr/bin/open'
            else:
                file_manager = 'xdg-open'
            subprocess.call([file_manager, temp])

    def print_page(self):
        self.print_preview()

    def save_as_pdf(self, path=None):
        self.layer_tree_control.set_edit_layer(None)
        pdf_canvas = renderer.PDFCanvas(project=self, path=path)
        pdf_canvas.copy_viewport_from(self.layer_canvas)
        pdf_canvas.update_renderers()
        pdf_canvas.render()

    def start_movie_recording(self):
        log.debug("Starting movie recording")
        self.latest_movie = apng.APNG()

    def stop_movie_recording(self):
        log.debug("Recorded %d frames" % len(self.latest_movie.frames))

    def save_latest_movie(self, path):
        if self.latest_movie is not None:
            self.latest_movie.save(path)

    def add_frame_to_movie(self, debug=False):
        if not self.timeline.timeline.is_beyond_playback_stop_value:
            log.debug("Recording image at %s" % time.strftime("%b %d %Y %H:%M", time.gmtime(self.timeline.current_time)))
            frame = self.get_numpy_image()
            if debug:
                frame_number = len(self.latest_movie.frames)
                h, w, depth = frame.shape
                image = wx.Image(w, h, frame)
                image.SaveFile("movie_frame_%03d.png" % frame_number, wx.BITMAP_TYPE_PNG)
            self.latest_movie.append(frame, delay=int(self.timeline.timeline.step_rate * 1000))
        else:
            log.debug("Skipping image at %s" % time.strftime("%b %d %Y %H:%M", time.gmtime(self.timeline.current_time)))

    @property
    def most_recent_uri(self):
        cmd = self.layer_manager.undo_stack.find_most_recent(mec.LoadLayersCommand)
        if cmd is None:
            return self.layer_manager.metadata.uri
        return cmd.metadata.uri

    def process_rnc_download(self, req, error):
        if not req.is_cancelled:
            if error is None:
                log.debug("loaded RNC map %s for map %s" % (req.path, req.extra_data))
                prefs = self.task.preferences
                kap = extract_from_zip(req.path, req.extra_data[0], prefs.bsb_directory)
                if kap:
                    self.window.application.load_file(kap, self.task, regime=req.extra_data[1])
                else:
                    self.task.error("The metadata in %s\nhas a problem: map %s doesn't exist" % (req.path, req.extra_data))
            else:
                self.task.error("Error downloading %s:\n%s" % (req.url, error))

    def check_rnc_map(self, url, filename, map_id):
        # Check for existence of already downloaded RNC map. This assumes the
        # layout BSB_ROOT/<map number>/<filename>
        prefs = self.task.preferences
        if prefs.bsb_directory:
            path = os.path.join(prefs.bsb_directory, "BSB_ROOT", map_id, filename)
            log.debug("checking for RNC file %s" % path)
            if os.path.exists(path):
                return path

    def download_rnc(self, url, filename, map_id, regime, confirm=False, name=None):
        kap = self.check_rnc_map(url, filename, map_id)
        if not kap:
            if confirm:
                if not self.task.confirm(f"Download RNC #{map_id}?\n\n{name}", "Confirm RNC Download"):
                    return
            self.download_file(url, None, self.process_rnc_download, (filename, regime))
        else:
            self.window.application.load_file(kap, self.task, regime=regime)

    def download_file(self, url, filename, callback, extra_data):
        if filename is None:
            filename = os.path.basename(url)
        req = self.download_control.request_download(url, filename, callback)
        req.extra_data = extra_data

    ###########################################################################
    # Private interface.
    ###########################################################################

    def _create_control(self, parent):
        """ Creates the toolkit-specific control for the widget. """

        panel = TileManager(parent)
        panel.Bind(TileManager.EVT_LAYOUT_CHANGED, self.on_layout_changed)

        self.document = self.layer_manager = LayerManager.create(self)
        self.layer_visibility = self.get_default_visibility()

        log.debug("LayerEditor: task=%s" % self.task)

        return panel

    def on_layout_changed(self, evt):
        layout = self.control.calc_layout()
        log.debug("on_layout_changed: new tilemanager layout {json.dumps(layout)}")

    def get_default_layout(self):
        try:
            data = get_template("%s.default_layout" % self.task.id)
        except OSError:
            log.error("no default layout")
            e = {}
        else:
            try:
                e = json.loads(data)
            except ValueError:
                log.error("invalid data in default layout")
                e = {}
        return e

    def create_layout(self, json):
        panel = self.control
        if "tile_manager" in json:
            layout = json
        else:
            layout = self.get_default_layout()
        panel.restore_layout(layout)

        # Mac can occasionally fail to get an OpenGL context, so creation of
        # the layer canvas can fail. Attempting to work around by giving it
        # more chances to work.
        attempts = 3
        while attempts > 0:
            attempts -= 1
            try:
                self.layer_canvas = LayerCanvas(panel, project=self)
                attempts = 0
            except wx.wxAssertionError:
                log.error("Failed initializing OpenGL context. Trying %d more times" % attempts)
                time.sleep(1)
        self.long_status = PopupStatusBar(self.layer_canvas)

        panel.add(self.layer_canvas, "layer_canvas", show_title=False, use_close_button=False)

        # Tree/Properties controls referenced from MapController
        self.layer_tree_control = LayerTreeControl(panel, self, size=(200, 300))
        panel.add(self.layer_tree_control, "layer_tree_control", use_close_button=False)

        self.layer_info = LayerInfoPanel(panel, self, size=(200, 200))
        panel.add(self.layer_info, "layer_info", use_close_button=False)

        self.selection_info = SelectionInfoPanel(panel, self, size=(200, 200))
        panel.add(self.selection_info, "selection_info", use_close_button=False)

        self.triangle_panel = TrianglePanel(panel, self.task)
        panel.add(self.triangle_panel, "triangle_panel", wx.RIGHT, sidebar=True, use_close_button=False)

        self.merge_points_panel = MergePointsPanel(panel, self.task)
        panel.add(self.merge_points_panel, "merge_points_panel", wx.RIGHT, sidebar=True, use_close_button=False)

        self.undo_history = UndoHistoryPanel(panel, self.task)
        panel.add(self.undo_history, "undo_history", wx.RIGHT, sidebar=True, use_close_button=False)

        self.flagged_control = panes.FlaggedPointPanel(panel, self.task)
        panel.add(self.flagged_control, "flagged_control", wx.RIGHT, sidebar=True, use_close_button=False)

        self.download_control = panes.DownloadPanel(panel, self.task)
        panel.add(self.download_control, "download_control", wx.RIGHT, sidebar=True, use_close_button=False)

        self.timeline = panes.TimelinePlaybackPanel(panel, self.task)
        panel.add_footer(self.timeline)

    # Traits event handlers

    @on_trait_change('layer_manager:layer_loaded')
    def layer_loaded(self, layer):
        log.debug("layer_loaded called for %s" % layer)
        self.layer_visibility[layer] = layer.get_visibility_dict()

    @on_trait_change('layer_manager:layers_changed')
    def layers_changed(self, batch_status):
        log.debug("layers_changed called!!!")
        try:
            collapse = batch_status.collapse
        except AttributeError:
            collapse = {}
        self.layer_tree_control.rebuild()
        self.layer_tree_control.collapse_layers(collapse)
        self.timeline.clear_marks()
        self.timeline.recalc_view()

    def update_layer_menu_ui(self, edit_layer):
        if edit_layer is not None:
            self.can_copy = edit_layer.can_copy()
            self.can_paste = True
            self.can_paste_style = self.clipboard_style is not None
            self.layer_can_save = edit_layer.can_save()
            self.layer_can_save_as = edit_layer.can_save_as()
            self.layer_selected = not edit_layer.is_root()
            self.layer_zoomable = edit_layer.is_zoomable()
            self.layer_above = self.layer_manager.is_raisable(edit_layer)
            self.layer_below = self.layer_manager.is_lowerable(edit_layer)
        else:
            self.can_copy = False
            self.can_paste = False
            self.can_paste_style = False
            self.layer_can_save = False
            self.layer_can_save_as = False
            self.layer_selected = False
            self.layer_zoomable = False
            self.layer_above = False
            self.layer_below = False

    def update_layer_selection_ui(self, edit_layer=None):
        if edit_layer is None:
            edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            # leave mouse_mode set to current setting
            self.mouse_mode_toolbar = edit_layer.mouse_mode_toolbar
            self.mouse_mode_factory = toolbar.get_valid_mouse_mode(self.mouse_mode_factory, self.mouse_mode_toolbar)
        else:
            self.mouse_mode_factory = mouse_handler.SelectionMode
        self.update_layer_menu_ui(edit_layer)
        self.layer_canvas.set_mouse_handler(self.mouse_mode_factory)
        self.multiple_layers = self.layer_manager.count_layers() > 1
        self.update_info_panels(edit_layer)
        self.update_layer_contents_ui(edit_layer)
        self.task.layer_selection_changed = edit_layer

    def update_layer_contents_ui(self, edit_layer=None):
        if edit_layer is None:
            edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            self.layer_has_points = edit_layer.has_points()
            self.layer_has_selection = edit_layer.has_selection()
            self.layer_has_flagged = edit_layer.has_flagged()
            self.layer_has_boundaries = edit_layer.has_boundaries()
            self.layer_is_groupable = edit_layer.has_groupable_objects()
            layer_name = edit_layer.name
        else:
            self.layer_has_points = False
            self.layer_has_selection = False
            self.layer_has_flagged = False
            self.layer_has_boundaries = False
            self.layer_is_groupable = False
            layer_name = "Current Layer"
        log.debug("has_points=%s, has_selection = %s, has_flagged=%s, has_boundaries = %s" % (self.layer_has_points, self.layer_has_selection, self.layer_has_flagged, self.layer_has_boundaries))
        self.layer_info.SetName(layer_name)
        self.update_undo_redo()

    def update_info_panels(self, layer, force=False):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer == layer:
            self.layer_info.display_panel_for_layer(self, layer, force)
            self.selection_info.display_panel_for_layer(self, layer, force)
        else:
            log.debug("Attempting to update panel for layer %s that isn't current", layer)

    def process_info_panel_keystroke(self, event, text):
        if self.layer_info.process_initial_key(event, text):
            return
        if self.selection_info.process_initial_key(event, text):
            return

    @on_trait_change('layer_manager:undo_stack_changed')
    def undo_stack_changed(self):
        log.debug("undo_stack_changed called!!!")
        self.refresh()

    @on_trait_change('layer_manager:layer_contents_changed')
    def layer_contents_changed(self, layer):
        log.debug("layer_contents_changed called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)

    @on_trait_change('layer_manager:layer_contents_changed_in_place')
    def layer_contents_changed_in_place(self, layer):
        log.debug("layer_contents_changed_in_place called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer, in_place=True)

    @on_trait_change('layer_manager:layer_contents_deleted')
    def layer_contents_deleted(self, layer):
        log.debug("layer_contents_deleted called!!! layer=%s" % layer)
        self.layer_canvas.rebuild_renderer_for_layer(layer)

    @on_trait_change('layer_manager:layer_metadata_changed')
    def layer_metadata_changed(self, layer):
        log.debug("layer_metadata_changed called!!! layer=%s" % layer)
        self.layer_tree_control.rebuild()

    @on_trait_change('layer_manager:refresh_needed')
    def refresh(self, batch_flags=None):
        log.debug("refresh called; batch_flags=%s" % batch_flags)
        if self.control is None:
            return
        if batch_flags is None or batch_flags is True:
            batch_flags = BatchStatus()

        # current control with focus is used to prevent usability issues with
        # text field editing in the calls to the info panel displays below.
        # Without checking for the current text field it is reformatted every
        # time, moving the cursor position to the beginning and generally
        # being annoying
        if batch_flags.editable_properties_changed:
            # except this prevents undo/redo from refreshing the state of the
            # control, and on undo text fields need to be refreshed regardless
            # of the cursor position.
            current = None
        else:
            current = self.window.control.FindFocus()

        # On Mac this is neither necessary nor desired.
        if not sys.platform.startswith('darwin'):
            self.control.Update()

        edit_layer = self.layer_tree_control.get_edit_layer()
        self.update_layer_contents_ui(edit_layer)
        self.update_layer_menu_ui(edit_layer)
        self.layer_info.display_panel_for_layer(self, edit_layer, batch_flags.editable_properties_changed, has_focus=current)
        self.selection_info.display_panel_for_layer(self, edit_layer, batch_flags.editable_properties_changed, has_focus=current)
        self.timeline.refresh_view()
        self.last_refresh = time.clock()
        self.control.Refresh()

    @on_trait_change('layer_manager:background_refresh_needed')
    def background_refresh(self):
        log.debug("background refresh called")
        t = time.clock()
        if t < self.last_refresh + 0.5:
            log.debug("refreshed too recently; skipping.")
            return
        self.refresh()

    @on_trait_change('layer_manager:threaded_image_loaded')
    def threaded_image_loaded(self, data):
        log.debug("threaded image loaded called")
        (layer, map_server_id), wms_request = data
        log.debug("event happed on %s for map server id %d" % (layer, map_server_id))
        log.debug(f"wms_request: {wms_request}")
        if layer.is_valid_threaded_result(map_server_id, wms_request):
            wx.CallAfter(self.layer_canvas.render)
        else:
            log.debug("Throwing away result from old map server id")

    # New Command processor

    def process_command(self, command, new_mouse_mode=None, override_editable_properties_changed=None):
        """Process a single command and immediately update the UI to reflect
        the results of the command.
        """
        try:
            # Fix for #702, crash adding a tile layer on MacOS. It seems that a
            # bunch of UI stuff gets called, cascading through a lot of trait
            # handlers and causing UI sizing and redrawing. Preventing the
            # immediate updates with the Freeze/Thaw pair seems to fix it.
            if sys.platform == "darwin":
                self.window.control.Freeze()

            b = BatchStatus()
            try:
                undo = self.process_batch_command(command, b)
            except MapRoomError as e:
                self.task.error(str(e), "Error Processing Command")
                if hasattr(e, 'error_points'):
                    layer = command.get_layer_in_layer_manager(self.layer_manager)
                    layer.highlight_exception(e)
                undo = None
            else:
                if override_editable_properties_changed is not None:
                    b.editable_properties_changed = override_editable_properties_changed
                self.perform_batch_flags(command, b)
                history = self.layer_manager.undo_stack.serialize()
                self.window.application.save_log(str(history), "command_log", ".mrc")
                if new_mouse_mode is not None:
                    self.mouse_mode_factory = new_mouse_mode
                    self.update_layer_selection_ui()
        finally:
            if sys.platform == "darwin":
                self.window.control.Thaw()
        return undo

    def process_flags(self, flags):
        b = BatchStatus()
        self.add_batch_flags(None, flags, b)
        self.perform_batch_flags(None, b)

    def process_batch_command(self, command, b):
        """Process a single command but don't update the UI immediately.
        Instead, update the batch flags to reflect the changes needed to
        the UI.

        """
        undo = self.layer_manager.undo_stack.perform(command, self)
        self.add_batch_flags(command, undo.flags, b)
        return undo

    def add_batch_flags(self, cmd, f, b):
        """Make additions to the batch flags as a result of the passed-in flags

        """
        # folders need bounds updating after child layers
        folder_bounds = []

        if f.layers_changed:
            # Only set this to True, never back to False once True
            b.layers_changed = True
        if f.refresh_needed:
            b.refresh_needed = True
        if f.immediate_refresh_needed:
            b.immediate_refresh_needed = True
        if f.errors:
            b.errors.append("When processing command '%s', the following errors were encountered:\n" % str(cmd))
            for e in f.errors:
                b.errors.append("- %s" % e)
            b.errors.append("")
        if f.message:
            b.messages.append("When processing command '%s', the following messages were generated:\n" % str(cmd))
            if isinstance(f.message, list):
                for m in f.message:
                    b.messages.append("- %s" % m)
            else:
                b.messages.append("- %s" % f.message)
            b.messages.append("")
        for lf in f.layer_flags:
            layer = lf.layer
            if layer in b.layers:
                log.debug("layer %s already in batch flags" % layer)
            else:
                b.layers.append(layer)
            if lf.layer_items_moved:
                b.editable_properties_changed = True
                if lf.indexes_of_points_affected is not None:
                    rebuild_layer = layer.update_affected_points(lf.indexes_of_points_affected)
                    if rebuild_layer is not None:
                        b.need_rebuild[rebuild_layer] = True
                else:
                    b.need_rebuild[layer] = True
                    if layer.is_folder():
                        folder_bounds.append((self.layer_manager.get_multi_index_of_layer(layer), layer))
                    else:
                        layer.update_bounds()
            if lf.layer_display_properties_changed:
                b.need_rebuild[layer] = False
                b.refresh_needed = True
                b.editable_properties_changed = True
            if lf.layer_contents_added or lf.layer_contents_deleted:
                b.need_rebuild[layer] = False
                b.immediate_refresh_needed = True
            # Hidden layer check only displayed in current window, not any others
            # that are displaying this project
            if lf.hidden_layer_check:
                vis = self.layer_visibility[layer]['layer']
                if not vis:
                    b.messages.append("Warning: operating on hidden layer %s" % layer.name)
            if lf.select_layer:
                # only the last layer in the list will be selected
                b.select_layer = layer
            if lf.layer_loaded:
                self.layer_manager.layer_loaded = layer
                b.layers_changed = True
            if lf.layer_metadata_changed:
                b.metadata_changed = True
            if lf.collapse:
                b.collapse[layer] = True

        # Update the folders after all the children are moved. Also, need to
        # update the folders from children to parent because child folder
        # bounds affect the parent.
        folder_bounds.sort()
        for _, layer in reversed(folder_bounds):
            layer.update_bounds()

    def perform_batch_flags(self, cmd, b):
        """Perform the UI updates given the BatchStatus flags

        """
        log.debug("perform_batch_flags layers affected: %s" % str(b.layers))
        for layer in b.layers:
            layer.increment_change_count()
            if layer.transient_edit_layer:
                affected_layer = layer.update_transient_layer(cmd)
                if affected_layer is not None:
                    b.need_rebuild[affected_layer] = True
                    b.need_rebuild[layer] = True

        # Use LayerManager events to trigger updates in all windows that are
        # displaying this project
        for layer, in_place in b.need_rebuild.items():
            if in_place:
                self.layer_manager.layer_contents_changed_in_place = layer
            else:
                self.layer_manager.layer_contents_changed = layer

        if b.layers_changed:
            self.layer_manager.layers_changed = b
        if b.metadata_changed:
            self.layer_manager.layer_metadata_changed = True

        overlay_affected = self.layer_manager.recalc_overlay_bounds()
        if overlay_affected:
            b.refresh_needed = True
            b.immediate_refresh_needed = False

        if b.immediate_refresh_needed:
            self.layer_canvas.render(immediately=True)
        if b.refresh_needed:
            self.layer_manager.refresh_needed = b
        if b.select_layer:
            self.layer_tree_control.set_edit_layer(b.select_layer)

        if b.errors:
            self.task.error("\n".join(b.errors))
        if b.messages:
            self.task.information("\n".join(b.messages), "Messages")

        self.undo_history.update_history()

    supported_clipboard_data_objects = [wx.CustomDataObject("maproom")]

    @property
    def clipboard_data_format(self):
        return "maproom"

    def copy_selection_to_clipboard(self, name):
        focused = self.control.FindFocus()
        if hasattr(focused, "AppendText"):
            try:
                text = focused.GetValue()
            except AttributeError:
                try:
                    text = focused.GetText()
                except AttributeError:
                    text = None
        else:
            text = None

        if text is not None:
            data_obj = wx.TextDataObject()
            data_obj.SetText(text)
            retval = "%d characters" % len(text)
        else:
            edit_layer = self.layer_tree_control.get_edit_layer()
            if edit_layer is not None:
                json_data = edit_layer.serialize_json(-999, children=True)
                text = json.dumps(json_data, indent=4)
                # print("clipboard object: json data", text)
                data_obj = wx.CustomDataObject("maproom")
                data_obj.SetData(text.encode('utf-8'))
                retval = "layer %s" % edit_layer.name
            else:
                data_obj = None
                retval = "Error: unable to copy a layer"
        clipboard.set_clipboard_object(data_obj)
        return retval

    def paste(self, cmd_cls=None):
        """ Pastes the current clipboard at the current insertion point or over
        the current selection
        """
        try:
            data_obj = clipboard.get_paste_data_object(self)
            self.process_paste_data_object(data_obj)
        except clipboard.ClipboardError as e:
            self.task.error(str(e), "Paste Error")

    def process_paste_data_object(self, data_obj, cmd_cls=None):
        # print("Found data object %s" % data_obj)
        text = clipboard.get_data_object_value(data_obj, "maproom")
        # print("value:", text)
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = mec.PasteLayerCommand(edit_layer, text, self.layer_canvas.world_center)
            self.process_command(cmd)

    def copy_style(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            self.__class__.clipboard_style = edit_layer.style.get_copy()
            self.can_paste_style = True

    def paste_style(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            style = self.clipboard_style
            if style is not None:
                cmd = moc.StyleChangeCommand(edit_layer, style)
                self.process_command(cmd)

    def clear_selection(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            edit_layer.clear_all_selections()
            self.update_layer_contents_ui()
            self.refresh()

    def delete_selection(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            cmd = edit_layer.delete_all_selected_objects()
            self.process_command(cmd)

    def clear_all_flagged(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            edit_layer.clear_flagged(refresh=False)
            self.update_layer_contents_ui()
            self.refresh()

    def select_all_flagged(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is not None:
            edit_layer.select_flagged(refresh=False)
            self.update_layer_contents_ui()
            self.refresh()

    def select_boundary(self):
        edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is None:
            return

        error = None
        try:
            progress_log.info("START=Finding outer boundary for %s" % edit_layer.name)
            status = edit_layer.select_outer_boundary()
        except ProgressCancelError:
            error = "cancel"
        except Exception:
            error = "Can't determine boundary"
        finally:
            progress_log.info("END")

        if error == "cancel":
            return
        elif error is not None:
            self.task.error(error, edit_layer.name)
        elif status is None:
            self.task.error("No complete boundary", edit_layer.name)
        else:
            self.update_layer_contents_ui()
            self.refresh()

    def point_tool_selected(self):
        for layer in self.layer_manager.flatten():
            layer.clear_all_line_segment_selections()
        self.refresh()

    def point_tool_deselected(self):
        pass

    def line_tool_selected(self):
        n = 0
        for layer in self.layer_manager.flatten():
            n += layer.get_num_points_selected()
        if (n > 1):
            for layer in self.layer_manager.flatten():
                layer.clear_all_point_selections()
            self.refresh()

    def line_tool_deselected(self):
        pass

    def clear_all_selections(self, refresh=True):
        for layer in self.layer_manager.flatten():
            layer.clear_all_selections()
            layer.clear_visibility_when_deselected(self.layer_visibility[layer])
        if refresh:
            self.refresh()

    def clickable_object_is_ugrid_point(self):
        return self.layer_canvas.picker.is_ugrid_point(self.clickable_object_mouse_is_over)

    def clickable_object_is_ugrid_line(self):
        return self.layer_canvas.picker.is_ugrid_line(self.clickable_object_mouse_is_over)

    def clickable_object_is_interior(self):
        return self.layer_canvas.picker.is_interior(self.clickable_object_mouse_is_over)

    def clickable_object_is_polygon_point(self):
        return self.layer_canvas.picker.is_polygon_point(self.clickable_object_mouse_is_over)

    def delete_selected_layer(self, layer=None):
        if layer is None:
            layer = self.layer_tree_control.get_edit_layer()
        if layer is None:
            self.window.status_bar.message = "Selected layer to delete!."
            return

        if layer.is_root():
            m = "The root node of the layer tree is selected. This will delete all layers in the tree."
        else:
            m = None

        if m is not None and not self.task.confirm(m):
            return

        cmd = mec.DeleteLayerCommand(layer)
        self.process_command(cmd)

    def check_for_errors(self, edit_layer=None, save_message=False):
        error = None
        if edit_layer is None:
            edit_layer = self.layer_tree_control.get_edit_layer()
        if edit_layer is None:
            return

        try:
            progress_log.info("START=Checking layer %s" % edit_layer.name)
            error = self.layer_manager.check_layer(edit_layer, self.window)
        except ProgressCancelError:
            error = "cancel"
        finally:
            progress_log.info("END")

        self.update_layer_contents_ui()
        all_ok = True
        if error == "cancel":
            all_ok = False
        elif error is not None:
            edit_layer.highlight_exception(error)
            if save_message:
                all_ok = self.task.confirm(str(error), "Layer Contains Problems; Save Anyway?")
            else:
                self.task.error(str(error), "Layer Contains Problems")
        else:
            edit_layer.clear_flagged(refresh=True)
            self.task.information("Layer %s OK" % edit_layer.name, "No Problems Found")
        log.debug(f"check_for_errors in {edit_layer}: {error}")
        self.flagged_control.recalc_view()
        return all_ok

    def check_all_layers_for_errors(self, save_message=False):
        messages = []
        error = None
        try:
            progress_log.info("START=Checking all layers for errors")
            for layer in self.layer_manager.flatten():
                progress_log.info("TITLE=Checking layer %s" % layer.name)
                error = self.layer_manager.check_layer(layer, self.window)
                if error is not None:
                    messages.append("%s: %s" % (layer.name, str(error)))
                    layer.highlight_exception(error)
                    self.control.Refresh()
                    error = None
        except ProgressCancelError:
            error = "cancel"
        finally:
            progress_log.info("END")

        self.update_layer_contents_ui()
        all_ok = True
        if error == "cancel":
            all_ok = False
        elif messages:
            if save_message:
                msg = "Layers Contains Problems; Save Anyway?"
                all_ok = self.task.confirm("\n\n".join(messages), msg, no_label="Don't Save", yes_label="Save")
            else:
                msg = "Layers With Problems"
                self.task.information("\n\n".join(messages), msg)
        else:
            for layer in self.layer_manager.flatten():
                layer.clear_flagged()
            self.layer_manager.refresh_needed = None
            if not save_message:
                self.task.information("Layers OK", "No Problems Found")
        return all_ok

    def editor_summary(self):
        lines = [FrameworkEditor.editor_summary(self)]
        lines.append("layer_manager summary:")
        lines.append(self.layer_manager.debug_structure("    "))
        lines.append(self.layer_canvas.debug_structure("    "))
        lines.append(self.layer_manager.undo_stack.debug_structure("    "))
        return "\n".join(lines)
