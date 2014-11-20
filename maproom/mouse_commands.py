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
        self.undo_info.flags.hidden_layer_check = self.layer
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

class ChangeDepthCommand(Command):
    def __init__(self, layer, indexes, depth):
        self.layer = layer
        self.indexes = indexes
        self.depth = depth
        self.undo_info = None
    
    def __str__(self):
        return "Set Depth to %s" % str(self.depth)
    
    def coalesce(self, next_command):
        if next_command.__class__ == self.__class__:
            if next_command.layer == self.layer and np.array_equal(next_command.indexes, self.indexes):
                self.depth = next_command.depth
                return True
    
    def is_recordable(self):
        return len(self.indexes) > 0
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        old_depths = np.copy(self.layer.points.z[self.indexes])
        undo.data = old_depths
        undo.flags.refresh_needed = True
        undo.flags.layer_items_moved = self.layer
        self.layer.points.z[self.indexes] = self.depth
        return self.undo_info

    def undo(self, editor):
        (old_depths) = self.undo_info.data
        self.layer.points.z[self.indexes] = old_depths
        return self.undo_info

class InsertLineCommand(Command):
    def __init__(self, layer, index, world_point):
        self.layer = layer
        self.index = index
        self.world_point = world_point
        self.undo_point = None
        self.undo_line = None
    
    def __str__(self):
        return "Line From Point #%d" % self.index
    
    def perform(self, editor):
        self.undo_point = self.layer.insert_point(self.world_point)
        self.undo_line = self.layer.insert_line_segment(self.undo_point.index, self.index)
        self.layer.select_point(self.undo_point.index)
        self.undo_point.flags.hidden_layer_check = self.layer
        # FIXME: merge undo status
        return self.undo_point

    def undo(self, editor):
        # FIXME: merge undo status
        undo_info = self.layer.delete_line_segment(self.undo_line.index)
        undo_info = self.layer.delete_point(self.undo_point.index)
        return undo_info

class DeleteLinesCommand(Command):
    def __init__(self, layer, point_indexes, line_indexes):
        self.layer = layer
        self.point_indexes = point_indexes
        self.line_indexes = line_indexes
        self.undo_point = None
        self.undo_line = None
    
    def __str__(self):
        if len(self.line_indexes) == 1:
            return "Delete Line #%d" % self.line_indexes[0]
        return "Delete %d Lines" % len(self.line_indexes)
    
    def perform(self, editor):
        self.undo_info = undo = UndoInfo()
        old_line_indexes = self.layer.get_lines_connected_to_points(self.point_indexes)
        if self.line_indexes is not None:
            old_line_indexes = np.unique(np.append(old_line_indexes, self.line_indexes))
        old_line_segments = np.copy(self.layer.line_segment_indexes[old_line_indexes])
        old_points = np.copy(self.layer.points[self.point_indexes])
        undo.data = (old_points, old_line_segments, old_line_indexes)
        print "DeleteLinesCommand: %s" % str(undo.data)
        undo.flags.refresh_needed = True
        undo.flags.layer_items_moved = self.layer
        undo.flags.layer_contents_deleted = self.layer
        self.layer.remove_points_and_lines(self.point_indexes, old_line_indexes)
        return self.undo_info

    def undo(self, editor):
        """
        Using the batch numpy.insert, it expects the point indexes to be
        relative to the current state of the array, not the original indexes.

        >>> a=np.arange(10)
        >>> a
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> indexes=[2,5,7,9]
        >>> b=np.delete(a,indexes,0)
        >>> b
        array([0, 1, 3, 4, 6, 8])
        >>> np.insert(b, indexes, indexes)
        IndexError: index 12 is out of bounds for axis 1 with size 10
        >>> fixed = indexes - np.arange(4)
        >>> np.insert(b, fixed, indexes)
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
        old_points, old_line_segments, old_line_indexes = self.undo_info.data
        offset = np.arange(len(self.point_indexes))
        indexes = self.point_indexes - offset
        self.layer.points = np.insert(self.layer.points, indexes, old_points).view(np.recarray)
        offset = np.arange(len(old_line_indexes))
        indexes = old_line_indexes - offset
        self.layer.line_segment_indexes = np.insert(self.layer.line_segment_indexes, indexes, old_line_segments).view(np.recarray)
        undo = UndoInfo()
        undo.flags.refresh_needed = True
        undo.flags.layer_items_moved = self.layer
        undo.flags.layer_contents_added = self.layer
        return undo

class CropRectCommand(Command):
    def __init__(self, layer, world_rect):
        self.layer = layer
        self.world_rect = world_rect
        self.undo_info = None
    
    def __str__(self):
        return "Crop"
    
    def perform(self, editor):
        self.undo_info = self.layer.crop_rectangle(self.world_rect)
        return self.undo_info

    def undo(self, editor):
        old_state = self.undo_info.data
        undo_info = self.layer.set_state(old_state)
        return undo_info
