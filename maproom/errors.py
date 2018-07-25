class MapRoomError(RuntimeError):
    pass


class PointsError(MapRoomError):
    def __init__(self, message, error_points=None):
        self.message = message
        self.error_points = error_points

    def __str__(self):
        return self.message
