import os
import math
import time
import numpy as np
from osgeo import gdal, gdal_array, osr
import pyproj
from maproom.library.accumulator import accumulator

from maproom.layers import PolygonLayer

from common import BaseLoader

class BNALoader(BaseLoader):
    mime = "application/x-maproom-bna"
    
    layer_type = "polygon"
    
    name = "BNA"
    
    def load(self, metadata, manager):
        layer = PolygonLayer(manager=manager)
        
        (layer.load_error_string,
         f_polygon_points,
         f_polygon_starts,
         f_polygon_counts,
         f_polygon_types,
         f_polygon_identifiers) = load_bna_file(metadata.uri)
        if (layer.load_error_string == ""):
            layer.set_data(f_polygon_points, f_polygon_starts, f_polygon_counts,
                           f_polygon_types, f_polygon_identifiers)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.type = self.mime
        return layer
    
    def check(self, layer):
        raise RuntimeError("Not abte to check BNA files")
    
    def save_to_file(self, f, layer):
        return "Can't save to BNA yet."


def update_status(message):
    import wx
    tlw = wx.GetApp().GetTopWindow()
    tlw.SetStatusText(message)
    wx.SafeYield()


def load_bna_file(file_path):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:
    
    ( load_error_string, polygon_points, polygon_starts, polygon_counts, polygon_types, polygon_identifiers )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        polygon_points = numpy array (type = 2 x np.float32)
        polygon_starts = numpy array (type = 1 x np.uint32)
        polygon_counts = numpy array (type = 1 x np.uint32)
        polygon_types = numpy array (type = 1 x np.uint32) (these are the BNA feature codes)
        polygon_identifiers = list
    """

    print "******** START"
    t0 = time.clock()
    f = file(file_path)
    s = f.read()
    f.close()
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "read in {0} seconds".format(t)

    # arr = np.fromstring(str, dtype=np.float32, sep=' ')
    t0 = time.clock()
    length = len(s)
    print "length = " + str(length)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "length in {0} seconds".format(t)

    t0 = time.clock()
    nr = s.count("\r")
    print "num \\r = = " + str(nr)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "count in {0} seconds".format(t)

    t0 = time.clock()
    nn = s.count("\n")
    print "num \\n = = " + str(nn)
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "count in {0} seconds".format(t)

    if (nr > 0 and nn > 0):
        t0 = time.clock()
        s = s.replace("\r", "")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "replace \\r with empty in {0} seconds".format(t)
        nr = 0

    if (nr > 0):
        t0 = time.clock()
        s = s.replace("\r", "\n")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        print "replace \\r with \\n in {0} seconds".format(t)
        nr = 0

    t0 = time.clock()
    lines = s.split("\n")
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print lines[0]
    print lines[1]
    print "split in {0} seconds".format(t)

    polygon_points = accumulator(block_shape=(2,), dtype=np.float32)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_types = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []

    t0 = time.clock()
    update_interval = .1
    update_every = 1000
    t_update = t0 + update_interval
    total_points = 0
    i = 0
    num_lines = len(lines)
    while True:
        if (i >= num_lines):
            break
        if (i % update_every) == 0:
            t = time.clock()
            if t > t_update:
                update_status("Loading %d of %d data points." % (i, num_lines))
                t_update = t + update_interval
        line = lines[i].strip()
        i += 1
        if (len(line) == 0):
            continue

        # fixme -- this will break if there are commas in any of the fields!
        pieces = line.split(",")
        if len(pieces) != 3:
            return ("The .bna file {0} is invalid. Error at line {1}.".format(file_path, i), None, None, None, None, None)
        polygon_identifiers.append(pieces[0].strip('"'))
        try:
            feature_code = int(pieces[1].strip('"'))
        except ValueError:
            feature_code = 0
        num_points = int(pieces[2])
        original_num_points = num_points

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        is_polygon = False
        if num_points < 3:
            num_points = abs(num_points)
        else:
            is_polygon = True

        # TODO: for now we just assume it's a polygon (could be a polyline or a point)
        # fixme: should we be adding polylines and points?
        # or put them somewhere separate -- particularly points!
        first_point = ()
        for j in xrange(num_points):
            line = lines[i].strip()
            i += 1
            pieces = line.split(",")
            p = (float(pieces[0]), float(pieces[1]))
            if (j == 0):
                first_point = p
            # if the last point is a duplicate of the first point, remove it
            if (j == (num_points - 1) and p[0] == first_point[0] and p[1] == first_point[1]):
                num_points -= 1
                continue
            polygon_points.append(p)

        polygon_starts.append(total_points)
        polygon_counts.append(num_points)
        polygon_types.append(feature_code)
        total_points += num_points

    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    print "loop in {0} seconds".format(t)
    print "******** END"
    update_status("Loaded %d data points in %.2fs." % (num_lines, t))

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            np.asarray(polygon_types)[:, 0],
            polygon_identifiers)
