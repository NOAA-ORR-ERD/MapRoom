import os


class UndoStack(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.insert_index = 0
        self.batch = self
    
    def perform(self, cmd, editor):
        if cmd is None:
            return CommandStatus()
        undo_info = cmd.perform(editor)
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
            return CommandStatus()
        undo_info = cmd.undo(editor)
        cmd.last_flags = undo_info.flags
        self.insert_index -= 1
        return undo_info
    
    def can_redo(self):
        return self.insert_index < len(self)
    
    def get_redo_command(self):
        if self.can_redo():
            return self[self.insert_index]
    
    def redo(self, editor):
        cmd = self.get_redo_command()
        if cmd is None:
            return CommandStatus()
        undo_info = cmd.perform(editor)
        cmd.last_flags = undo_info.flags
        self.insert_index += 1
        return undo_info
    
    def start_batch(self):
        if self.batch == self:
            self.batch = Batch()
    
    def end_batch(self):
        self.batch = self

    def add_command(self, command):
        self.batch.insert_at_index(command)
        print self.batch.history_list()
    
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


class CommandStatus(object):
    def __init__(self):
        # True if command successfully completes, must set to False on failure
        self.success = True
        
        # Message displayed to the user
        self.message = None
        
        # True if screen redraw needed
        self.refresh_needed = False
        
        # True if projection changed on any layer
        self.projection_changed = False
        
        # True if layers have been added, removed, renamed or reordered in the
        # layer manager
        self.layers_changed = False
        
        ##### Set the following to the layer object if ...
        
        # ...any items within the layer have changed position only
        self.layer_items_moved = None
        
        # ...any items have been added to the layer
        self.layer_contents_added = None
        
        # ...any items have been removed from the layer
        self.layer_contents_deleted = None
        
        # ...any drawing properties changed (color, line width, etc.)
        self.layer_display_properties_changed = None
        
        # ...any layer properties changed (layer name, etc.)
        self.layer_metadata_changed = None
        
        # ...this layer was loaded and needs to be broadcast to all views
        self.layer_loaded = None
        
        # ...a message should be displayed if it isn't the top layer and it was edited
        self.hidden_layer_check = None
        
        # ...needs to be selected after the command finishes
        self.select_layer = None


class UndoInfo(object):
    def __init__(self):
        self.index = -1
        self.data = None
        self.flags = CommandStatus()
    
    def __str__(self):
        return "index=%d, flags=%s" % (self.index, str(dir(self.flags)))

class Command(object):
    def __init__(self):
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
