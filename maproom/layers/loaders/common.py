import os
import glob
import tempfile
import shutil

from maproom.layers import constants
from maproom.library.Boundary import Boundaries, PointsError

class BaseLoader(object):
    mime = None
    
    # List of supported filename extensions, including the leading ".".  If
    # multiple extensions are supported, put the most common one first so that
    # the file dialog will display that as the default.
    extensions = []
    
    name = "Abstract loader"
    
    def can_load(self, metadata):
        return metadata.mime == self.mime
    
    def can_save_layer(self, layer):
        return False
    
    def can_save_project(self):
        return False
    
    def is_valid_extension(self, extension):
        return extension.lower() in self.extensions
    
    def get_pretty_extension_list(self):
        return ", ".join(self.extensions)
    
    def get_file_dialog_wildcard(self):
        # Using only the first extension
        wildcards = []
        if self.extensions:
            ext = self.extensions[0]
            wildcards.append("%s (*%s)|*%s" % (self.name, ext, ext))
        return "|".join(wildcards)

class BaseLayerLoader(BaseLoader):
    layer_types = []
    
    points_per_tick = 5000
    
    def can_save_layer(self, layer):
        return layer.type in self.layer_types
    
    def load_layers(self, metadata, manager):
        raise NotImplementedError
    
    def save_layer(self, uri, layer):
        if uri is None:
            uri = layer.file_path
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(uri))

        try:
            error = self.save_to_local_file(temp_file, layer)
        except Exception as e:
            import traceback
            print traceback.format_exc(e)
            if hasattr(e, "points") and e.points is not None:
                layer.clear_all_selections(constants.STATE_FLAGGED)
                for p in e.points:
                    layer.select_point(p, constants.STATE_FLAGGED)
                layer.manager.dispatch_event('refresh_needed')
            error = e.message
        
        if (not error and temp_file and os.path.exists(temp_file)):
            if layer.get_num_points_selected(constants.STATE_FLAGGED):
                layer.clear_all_selections(constants.STATE_FLAGGED)
                layer.manager.dispatch_event('refresh_needed')
            try:
                # copy all the files that have been created in that directory;
                # e.g. ESRI shapefile data is spread among 3 files.
                uri_base = os.path.dirname(uri)
                for path in glob.glob(os.path.join(temp_dir, "*")):
                    filename = os.path.basename(path)
                    shutil.copy(path, os.path.join(uri_base, filename))
                layer.file_path = uri
            except Exception as e:
                import traceback
            
                error = "Unable to save file to disk. Make sure you have write permissions to the file.\n\nSystem error was: %s" % e.message
                print traceback.format_exc(e)
        return error
    
    def save_to_local_file(self, filename, layer):
        fh = open(filename, "w")
        error = ""
        had_error = False
        try:
            self.save_to_fh(fh, layer)
        except Exception:
            raise
        finally:
            fh.close()
        return error
    
    def save_to_fh(self, fh, layer):
        raise NotImplementedError

    def get_boundaries(self, layer):
        boundaries = Boundaries(layer, allow_branches=False)
        errors, error_points = boundaries.check_errors()
        if errors:
            raise PointsError("Problems with boundaries:\n\n%s" % "\n\n".join(errors), error_points)
        
        # normalize windings on polygons
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
