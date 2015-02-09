import os
import tempfile
import shutil

from maproom.layers import constants

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
    
    def can_save_layer(self, layer):
        return layer.type in self.layer_types
    
    def load_layers(self, metadata, manager):
        raise NotImplementedError
    
    def save_layer(self, uri, layer):
        if uri is None:
            uri = layer.file_path
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(uri))

        error = self.save_to_local_file(temp_file, layer)
        
        if (not error and temp_file and os.path.exists(temp_file)):
            if layer.get_num_points_selected(constants.STATE_FLAGGED):
                layer.clear_all_selections(constants.STATE_FLAGGED)
                layer.manager.dispatch_event('refresh_needed')
            try:
                shutil.copy(temp_file, uri)
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
        except Exception as e:
            import traceback
            
            print traceback.format_exc(e)
            if hasattr(e, "points") and e.points is not None:
                layer.clear_all_selections(constants.STATE_FLAGGED)
                for p in e.points:
                    layer.select_point(p, constants.STATE_FLAGGED)
                layer.manager.dispatch_event('refresh_needed')
            error = e.message
        finally:
            fh.close()
        return error
    
    def save_to_fh(self, fh, layer):
        raise NotImplementedError
