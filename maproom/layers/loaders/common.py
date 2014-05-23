class PointsError(Exception):
    def __init__(self, message, points=None):
        Exception.__init__(self, message)
        self.points = points
