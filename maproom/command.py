import os

from serializer import Serializer

class UndoStack(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.insert_index = 0
        self.batch = self
    
    def perform(self, cmd, editor):
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.perform(editor)
        if undo_info.flags.success:
            self.add_command(cmd)
        cmd.last_flags = undo_info.flags
        return undo_info

    def can_undo(self):
        return self.insert_index > 0
    
    def get_undo_command(self):
        if self.can_undo():
            return self[self.insert_index - 1]
    
    def undo(self, editor):
        cmd = self.get_undo_command()
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.undo(editor)
        if undo_info.flags.success:
            self.insert_index -= 1
        cmd.last_flags = undo_info.flags
        return undo_info
    
    def can_redo(self):
        return self.insert_index < len(self)
    
    def get_redo_command(self):
        if self.can_redo():
            return self[self.insert_index]
    
    def redo(self, editor):
        cmd = self.get_redo_command()
        if cmd is None:
            return UndoInfo()
        undo_info = cmd.perform(editor)
        if undo_info.flags.success:
            self.insert_index += 1
        cmd.last_flags = undo_info.flags
        return undo_info
    
    def start_batch(self):
        if self.batch == self:
            self.batch = Batch()
    
    def end_batch(self):
        self.batch = self

    def add_command(self, command):
        self.batch.insert_at_index(command)
        print self.batch.history_list()
        s = self.serialize()
        print s
    
    def insert_at_index(self, command):
        last = self.get_undo_command()
        if last is not None and last.coalesce(command):
            return
        if command.is_recordable():
            self[self.insert_index:] = [command]
            self.insert_index += 1
    
    def history_list(self):
        h = [str(c) for c in self]
        return h
    
    def serialize(self):
        s = Serializer()
        for c in self:
            s.add(c)
        return s


class LayerStatus(object):
    def __init__(self, layer):
        self.layer = layer  # FIXME: should be a weakref (or maybe token)
        
        # True if items within the layer have changed position only
        self.layer_items_moved = False
        
        # True if items have been added to the layer
        self.layer_contents_added = False
        
        # True if items have been removed from the layer
        self.layer_contents_deleted = False
        
        # True if drawing properties changed (color, line width, etc.)
        self.layer_display_properties_changed = False
        
        # True if layer properties changed (layer name, etc.)
        self.layer_metadata_changed = False
        
        # True if this layer was loaded and needs to be broadcast to all views
        self.layer_loaded = False
        
        # True if a message should be displayed if it isn't the top layer and
        # it was edited
        self.hidden_layer_check = False
        
        # ...needs to be selected after the command finishes
        self.select_layer = False
        
        # True if layer should be zoomed
        self.zoom_to_layer = False
    

class CommandStatus(object):
    def __init__(self):
        # True if command successfully completes, must set to False on failure
        self.success = True
        
        # List of errors encountered
        self.errors = []
        
        # Message displayed to the user
        self.message = None
        
        # True if screen redraw needed
        self.refresh_needed = False
        
        # True if projection changed on any layer
        self.projection_changed = False
        
        # True if layers have been added, removed, renamed or reordered in the
        # layer manager
        self.layers_changed = False
        
        self.layer_flags = []
    
    def add_layer_flags(self, layer):
        lf = LayerStatus(layer)
        self.layer_flags.append(lf)
        return lf


class UndoInfo(object):
    def __init__(self):
        self.index = -1
        self.data = None
        self.flags = CommandStatus()
    
    def __str__(self):
        return "index=%d, flags=%s" % (self.index, str(dir(self.flags)))

class Command(object):
    serialize_order = [
        ('layer', 'layer'),
        ]
    
    def __init__(self, layer=None):
        # FIXME: should be some invariant that uniquely identifies the layer
        # without being a reference to the layer itself.  When a layer gets
        # removed and then re-added (by an undo then redo) a new layer is
        # created for that command.  The layers saved in redos beyond that
        # command are pointing to the old layer, not the newly created layer,
        # so stepping into the future operates on the old layer that isn't
        # displayed.
        self.layer = layer
        
        self.undo_info = None
        self.last_flags = None

    def __str__(self):
        return "<unnamed command>"
    
    def coalesce(self, next_command):
        return False
    
    def is_recordable(self):
        return True
    
    def perform(self, editor):
        pass
    
    def undo(self, editor):
        pass


class Batch(UndoStack):
    """A batch is immutable once created, so there's no need to allow
    intermediate index points.
    """
    def __str__(self):
        return "<batch>"
    
    def perform(self, editor):
        for c in self:
            c.perform(editor)
    
    def undo(self, editor):
        for c in reversed(self):
            c.undo(editor)
