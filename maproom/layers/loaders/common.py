import os
import tempfile
import shutil

from maproom.layers import constants

class PointsError(Exception):
    def __init__(self, message, points=None):
        Exception.__init__(self, message)
        self.points = points

class BaseLoader(object):
    mime = None
    
    layer_type = ""
    
    name = "Abstract loader"
    
    def can_load(self, metadata):
        return metadata.mime == self.mime
    
    def load(self, metadata, manager):
        raise NotImplementedError
    
    def can_save(self, layer):
        return layer.type == self.layer_type
    
    def check(self, layer):
        pass
    
    def save(self, uri, layer):
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
            if hasattr(e, "points") and e.points != None:
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
