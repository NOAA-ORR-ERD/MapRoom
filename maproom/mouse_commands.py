import numpy as np

from command import Command, UndoInfo

class InsertPointCommand(Command):
    def __init__(self, layer, world_point):
        self.layer = layer
        self.world_point = world_point
        self.undo_info = None
    
    def __str__(self):
        return "Add Point #%d" % self.undo_info.index
    
    def perform(self, editor):
        self.undo_info = self.layer.insert_point(self.world_point)
        self.layer.select_point(self.undo_info.index)
        vis = editor.layer_visibility[self.layer]['layer']
        if not vis:
            self.undo_info.message = "Added point to hidden layer %s" % layer.name
        return self.undo_info

    def undo(self, editor):
        undo_info = self.layer.delete_point(self.undo_info.index)
        return undo_info

class MovePointsCommand(Command):
    def __init__(self, layer, indexes, dx, dy):
        self.layer = layer
        self.indexes = indexes
        self.dx = dx
        self.dy = dy
        self.undo_info = None
    
    def __str__(self):
        if len(self.indexes) == 1:
            return "Move Point #%d" % self.indexes[0]
        return "Move %d Points" % len(self.indexes)
    
    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and np.array_equal(next_command.indexes, self.indexes):
                self.dx += next_command.dx
                self.dy += next_command.dy
                return True
    
    def is_recordable(self):
        return len(self.indexes) > 0
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        old_x = np.copy(self.layer.points.x[self.indexes])
        old_y = np.copy(self.layer.points.y[self.indexes])
        undo.data = (old_x, old_y)
        undo.flags.refresh_needed = True
        undo.flags.layer_items_moved = self.layer
        undo.flags.layer_contents_added = self.layer
        self.layer.points.x[self.indexes] += self.dx
        self.layer.points.y[self.indexes] += self.dy
        print "dx=%f, dy=%f" % (self.dx, self.dy)
        print self.indexes
        print undo.data
        return self.undo_info

    def undo(self, editor):
        (old_x, old_y) = self.undo_info.data
        self.layer.points.x[self.indexes] = old_x
        self.layer.points.y[self.indexes] = old_y
        return self.undo_info
