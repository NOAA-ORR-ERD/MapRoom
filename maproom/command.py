import os


class UndoStack(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
        self.insert_index = 0
        self.batch = self
    
    def can_undo(self):
        return self.insert_index > 0
    
    def get_undo_command(self):
        if self.can_undo():
            return self[self.insert_index - 1]
    
    def undo(self, editor):
        cmd = self.get_undo_command()
        if cmd is not None:
            undo_info = cmd.undo(editor)
            self.insert_index -= 1
            return undo_info
        return None
    
    def can_redo(self):
        return self.insert_index < len(self)
    
    def get_redo_command(self):
        if self.can_redo():
            return self[self.insert_index]
    
    def redo(self, editor):
        cmd = self.get_redo_command()
        if cmd is not None:
            undo_info = cmd.perform(editor)
            self.insert_index += 1
            return undo_info
        return None
    
    def start_batch(self):
        if self.batch == self:
            self.batch = Batch()
    
    def end_batch(self):
        self.batch = self

    def add_command(self, command):
        self.batch.insert_at_index(command)
        self.batch.list_contents()
    
    def insert_at_index(self, command):
        self[self.insert_index:] = [command]
        self.insert_index += 1
    
    def list_contents(self):
        for i, command in enumerate(self):
            print "%03d" % i, command


class CommandStatus(object):
    def __init__(self):
        self.message = None
        self.refresh_needed = False
        self.layer_contents_changed = False
        self.layer_contents_deleted = False
        self.layer_metadata_changed = False
        self.projection_changed = False
        
        self.select_layer = None


class UndoInfo(object):
    def __init__(self):
        self.index = -1
        self.data = None
        self.flags = CommandStatus()

class Command(object):
    def __str__(self):
        return "<unnamed command>"
    
    def perform(self, editor):
        pass
    
    def undo(self, editor):
        pass


class Batch(Command):
    """A batch is immutable once created, so there's no need to allow
    intermediate index points.
    """
    def __init__(self, *args, **kwargs):
        self.commands = []
    
    def __str__(self):
        return "<batch>"
    
    def insert_at_index(self, command):
        self.commands.append(command)
    
    def perform(self, editor):
        for c in self.commands:
            c.perform(editor)
    
    def undo(self, editor):
        for c in reversed(self.commands):
            c.undo(editor)

