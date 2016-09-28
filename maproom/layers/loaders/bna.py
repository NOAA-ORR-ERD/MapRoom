import os
import math
import time

from fs.opener import fsopen
import numpy as np
from osgeo import gdal, gdal_array, osr
import pyproj

from maproom.library.accumulator import accumulator
from maproom.layers import PolygonLayer, RNCLoaderLayer

from common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

class BNALoader(BaseLayerLoader):
    mime = "application/x-maproom-bna"
    
    layer_types = ["polygon"]
    
    extensions = [".bna"]
    
    name = "BNA"
    
    layer_class = PolygonLayer

    def load_layers(self, metadata, manager):
        layer = self.layer_class(manager=manager)
        
        (layer.load_error_string,
         f_polygon_points,
         f_polygon_starts,
         f_polygon_counts,
         f_polygon_identifiers) = load_bna_file(metadata.uri)
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_data(f_polygon_points, f_polygon_starts, f_polygon_counts,
                           f_polygon_identifiers)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]
    
    def save_to_fh(self, fh, layer):
        save_bna_file(fh, layer)

class RNCLoader(BNALoader):
    mime = "application/x-maproom-rncloader"
    
    layer_types = ["rncloader"]
    
    extensions = [".bna"]
    
    name = "RNCLoader"
    
    layer_class = RNCLoaderLayer


def load_bna_file(uri):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:
    
    ( load_error_string, polygon_points, polygon_starts, polygon_counts, polygon_types, polygon_identifiers )
    
    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        polygon_points = numpy array (type = 2 x np.float64)
        polygon_starts = numpy array (type = 1 x np.uint32)
        polygon_counts = numpy array (type = 1 x np.uint32)
        polygon_types = numpy array (type = 1 x np.uint32) (these are the BNA feature codes)
        polygon_identifiers = list
    """

    log.debug("******** START")
    t0 = time.clock()
    f = fsopen(uri, "r")
    s = f.read()
    f.close()
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug("read in {0} seconds".format(t))

    # arr = np.fromstring(str, dtype=np.float64, sep=' ')
    t0 = time.clock()
    length = len(s)
    log.debug("length = " + str(length))
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug("length in {0} seconds".format(t))

    t0 = time.clock()
    nr = s.count("\r")
    log.debug("num \\r = = " + str(nr))
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug("count in {0} seconds".format(t))

    t0 = time.clock()
    nn = s.count("\n")
    log.debug("num \\n = = " + str(nn))
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug("count in {0} seconds".format(t))

    if (nr > 0 and nn > 0):
        t0 = time.clock()
        s = s.replace("\r", "")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        log.debug("replace \\r with empty in {0} seconds".format(t))
        nr = 0

    if (nr > 0):
        t0 = time.clock()
        s = s.replace("\r", "\n")
        t = time.clock() - t0  # t is wall seconds elapsed (floating point)
        log.debug("replace \\r with \\n in {0} seconds".format(t))
        nr = 0

    t0 = time.clock()
    lines = s.split("\n")
    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug(lines[0])
    log.debug(lines[1])
    log.debug("split in {0} seconds".format(t))

    polygon_points = accumulator(block_shape=(2,), dtype=np.float64)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []

    t0 = time.clock()
    update_every = 1000
    total_points = 0
    i = 0
    num_lines = len(lines)
    progress_log.info("TICKS=%d" % num_lines)
    progress_log.info("Loading BNA...")
    while True:
        if (i >= num_lines):
            break
        if (i % update_every) == 0:
            progress_log.info("TICK=%d" % i)
        line = lines[i].strip()
        i += 1
        if (len(line) == 0):
            continue

        # fixme -- this will break if there are commas in any of the fields!
        pieces = line.split(",")
        if len(pieces) != 3:
            return ("The .bna file {0} is invalid. Error at line {1}.".format(file_path, i), None, None, None, None, None)
        try:
            feature_code = int(pieces[1].strip('"'))
        except ValueError:
            feature_code = 0
        name = pieces[0].strip('"')
        if name.lower() in ['map bounds', 'mapbounds']:
            feature_code = 4
        elif name.lower() in ['spillable area', 'spillablearea']:
            feature_code = 5
        polygon_identifiers.append(
            {'name': name,
             'feature_code': feature_code}
            )

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
        total_points += num_points
    progress_log.info("TICK=%d" % num_lines)

    t = time.clock() - t0  # t is wall seconds elapsed (floating point)
    log.debug("loop in {0} seconds".format(t))
    log.debug("******** END")

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            polygon_identifiers)


def save_bna_file(f, layer):
    update_every = 1000
    ticks = 0
    progress_log.info("TICKS=%d" % np.alen(layer.points))
    progress_log.info("Saving BNA...")
    for i, p in enumerate(layer.iter_polygons()):
        print "polygon #%d" % i
        polygon = p[0]
        count = np.alen(polygon)
        ident = p[1]
        print ident
        f.write('"%s","%s", %d\n' % (ident['name'], ident['feature_code'], count))
        for j in range(count):
            f.write("%s,%s\n" % (polygon[j][0], polygon[j][1]))
            ticks += 1
                
            if (ticks % update_every) == 0:
                progress_log.info("TICK=%d" % ticks)
    progress_log.info("TICK=%d" % ticks)
    progress_log.info("Saved BNA")
