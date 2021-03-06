import os
import glob
import tempfile
import shutil

from maproom.app_framework.filesystem import fsopen as open
from maproom.app_framework.errors import ProgressCancelError

from ..library.Boundary import Boundaries, PointsError
from ..command import UndoInfo

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


class BaseLoader(object):
    mime = None

    # List of supported filename extensions, including the leading ".".  If
    # multiple extensions are supported, put the most common one first so that
    # the file dialog will display that as the default.
    extensions = []

    # map of extension to pretty name
    extension_desc = {}

    name = "Abstract loader"

    def can_load(self, metadata):
        return metadata.mime == self.mime

    def can_save_layer(self, layer):
        return False

    def can_save_project(self):
        return False

    def extension_name(self, ext):
        if ext in self.extension_desc:
            return self.extension_desc[ext]
        return self.name

    def is_valid_extension(self, extension):
        return extension.lower() in self.extensions

    def get_pretty_extension_list(self):
        return ", ".join(self.extensions)

    def get_file_dialog_wildcard(self, requested_ext=None):
        # Using only the first extension
        wildcards = []
        if self.extensions:
            for ext in self.extensions:
                if ext is None or ext == requested_ext:
                    name = self.extension_name(ext)
                    wildcards.append("%s (*%s)|*%s" % (name, ext, ext))
        return "|".join(wildcards)

    def load_layers_from_uri(self, uri, manager, **kwargs):
        undo = UndoInfo()
        saved_invariant = manager.next_invariant
        try:
            layers = self.load_layers(uri, manager, **kwargs)
        except ProgressCancelError as e:
            undo.flags.success = False
            undo.flags.errors = [str(e)]
        except IOError as e:
            undo.flags.success = False
            undo.flags.errors = [str(e)]

        if undo.flags.success:
            if layers is None:
                undo.flags.success = False
                undo.flags.errors = [f"Invalid layer data in {uri} for {self.name} layer"]
            else:
                errors = []
                warnings = []
                for layer in layers:
                    if layer.load_error_string != "":
                        errors.append(layer.load_error_string)
                    if layer.load_warning_string != "":
                        warnings.append(layer.load_warning_string)
                if errors:
                    undo.flags.success = False
                    undo.flags.errors = errors
                if warnings:
                    undo.flags.message = warnings

            if undo.flags.success:
                manager.add_layers(layers, None)
                first = True
                for layer in layers:
                    lf = undo.flags.add_layer_flags(layer)
                    if first:
                        lf.select_layer = True
                        first = False
                    lf.layer_loaded = True

                undo.flags.layers_changed = True
                undo.flags.refresh_needed = True
                undo.data = (layers, saved_invariant)

        return undo


class BaseLayerLoader(BaseLoader):
    layer_types = []

    points_per_tick = 5000

    def can_save_layer(self, layer):
        return layer.type in self.layer_types

    def load_layers(self, metadata, manager, **kwargs):
        raise NotImplementedError

    def save_layer(self, uri, layer):
        from maproom.layers import state

        # Save the layer inside an empty directory because we can't assume that
        # the layer will only overwrite a single file. ESRI shapefile data is
        # spread among at least 3 file, for instance. Everything that's created
        # in the temp directory is copied back to the real location.
        if uri is None:
            uri = layer.file_path
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(uri))
        log.debug(f"save_layer: temp_file = {temp_file}")

        try:
            saved_file = self.save_to_local_file(temp_file, layer)
            if saved_file is not None:
                temp_file = saved_file
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            if hasattr(e, "error_points") and e.error_points is not None:
                layer.clear_all_selections(state.FLAGGED)
                for p in e.error_points:
                    layer.select_point(p, state.FLAGGED)
                layer.manager.project.refresh(True)
            error = str(e)
        else:
            error = None

        if not error:
            if temp_file and os.path.exists(temp_file):
                if layer.get_num_points_selected(state.FLAGGED):
                    layer.clear_all_selections(state.FLAGGED)
                    layer.manager.project.refresh(True)
                try:
                    self.gather_save_files(temp_dir, uri)
                    layer.file_path = uri
                except Exception as e:
                    import traceback

                    error = "Unable to save file to disk. Make sure you have write permissions to the file.\n\nSystem error was: %s" % str(e)
                    print(traceback.format_exc())
            else:
                error = "No file produced. Hmmm."
        return error

    def save_to_local_file(self, filename, layer):
        fh = open(filename, "w")
        try:
            self.save_to_fh(fh, layer)
        except Exception:
            raise
        finally:
            fh.close()

    def save_to_fh(self, fh, layer):
        raise NotImplementedError

    def gather_save_files(self, temp_dir, uri):
        """Move generated save file(s) to the uri specified.

        Everything in the directory temp_dir will have been generated by the
        save routine, so must copy everything to the directory of the
        destination file.
        """
        uri_base = os.path.dirname(uri)
        log.debug(f"save_layers: uri={uri} uri_base={uri_base}")
        for path in glob.glob(os.path.join(temp_dir, "*")):
            filename = os.path.basename(path)
            dest_uri = uri_base + "/" + filename
            with open(dest_uri, "wb") as fh:
                data = open(path, "rb").read()
                fh.write(data)


    def get_boundaries(self, layer):
        boundaries = Boundaries(layer, allow_branches=False)
        errors, error_points = boundaries.check_errors()
        if errors:
            raise PointsError("Problems with boundaries:\n\n%s" % "\n\n".join(errors), error_points)

        # normalize windings on rings
        for (boundary_index, boundary) in enumerate(boundaries):
            # if the outer boundary's area is positive, then reverse its
            # points so that they're wound counter-clockwise
            if boundary_index == 0:
                if boundary.area > 0.0:
                    boundary = reversed(boundary)
            # if any other boundary has a negative area, then reverse its
            # points so that they're wound clockwise
            elif boundary.area < 0.0:
                boundary = reversed(boundary)

        return boundaries
