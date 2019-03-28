import os

from sawx.filesystem import fsopen as open

import numpy as np
from shapely.geometry import Polygon, LineString

from maproom.library.shapely_utils import add_maproom_attributes_to_shapely_geom
from maproom.layers import PolygonLayer, RNCLoaderLayer, PolygonShapefileLayer

from .common import BaseLayerLoader
from .shapefile import BNAShapefileLoader, write_layer_as_shapefile

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def identify_loader(file_guess):
    if not file_guess.is_binary and file_guess.uri.lower().endswith(".bna"):
        lines = file_guess.sample_data.splitlines()
        if b".KAP" in lines[0]:
            return dict(mime="application/x-maproom-rncloader", loader=RNCLoader())
        return dict(mime="application/x-maproom-bna", loader=BNAShapefileLoader())


class RNCLoader(BaseLayerLoader):
    mime = "application/x-maproom-rncloader"

    layer_types = ["rncloader"]

    extensions = [".bna"]

    name = "RNCLoader"

    layer_class = RNCLoaderLayer

    def load_layers(self, uri, manager, **kwargs):
        layer = self.layer_class(manager=manager)

        (layer.load_error_string,
         f_ring_points,
         f_ring_starts,
         f_ring_counts,
         f_ring_identifiers) = load_bna_file(uri, regimes=[0, 360])
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_data(f_ring_points, f_ring_starts, f_ring_counts,
                           f_ring_identifiers)
            layer.file_path = uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]

    def save_to_fh(self, fh, layer):
        save_bna_file(fh, layer)


def parse_bna_file(uri):
    f = open(uri, "r")
    s = f.read()
    f.close()
    lines = s.splitlines()
    items = []

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
            raise RuntimeError("Error at line {0}. Expecting line with 3 items: primary name, secondary name & point count.".format(i))
        try:
            feature_code = int(pieces[1].strip('"'))
        except ValueError:
            feature_code = 0
        name = pieces[0].strip('"')
        if name.lower() in ['map bounds', 'mapbounds']:
            feature_code = 4
        elif name.lower() in ['spillable area', 'spillablearea']:
            feature_code = 5
        feature_type = pieces[1].strip('"')
        num_points = int(pieces[2])

        # A negative num_points value indicates that this is a line
        # rather than a polygon. And if a "polygon" only has 1 or 2
        # points, it's not a polygon.
        is_polygon = False
        if num_points < 3:
            num_points = abs(num_points)
        else:
            is_polygon = True

        polygon_points = np.zeros((num_points, 2), dtype=np.float64)
        first_point = ()
        for j in range(num_points):
            line = lines[i].strip()
            i += 1
            pieces = line.split(",")
            if len(pieces) != 2:
                raise RuntimeError("Error at line {0}. Expecting line with 2 items: longitude, latitude.".format(i))
            p = (float(pieces[0]), float(pieces[1]))
            if (j == 0):
                first_point = tuple(p)
            # if the last point is a duplicate of the first point, remove it
            if (j == (num_points - 1) and p[0] == first_point[0] and p[1] == first_point[1]):
                num_points -= 1
                continue
            polygon_points[j, :] = p
        total_points += num_points

        item = [name, feature_type, feature_code, num_points, is_polygon, polygon_points]
        items.append(item)
    progress_log.info("TICK=%d" % num_lines)
    return items, total_points


def load_bna_file(uri, regimes=None):
    """
    used by the code below, to separate reading the file from creating the special maproom objects.
    reads the data in the file, and returns:

    ( load_error_string, polygon_points, polygon_starts, polygon_counts, polygon_types, ring_identifiers )

    where:
        load_error_string = string descripting the loading error, or "" if there was no error
        polygon_points = numpy array (type = 2 x np.float64)
        polygon_starts = numpy array (type = 1 x np.uint32)
        polygon_counts = numpy array (type = 1 x np.uint32)
        polygon_types = numpy array (type = 1 x np.uint32) (these are the BNA feature codes)
        ring_identifiers = list
    """
    items, total_points = parse_bna_file(uri)
    num_polygons = len(items)

    if regimes is None:
        regimes = [0]
    total_points *= len(regimes)
    num_polygons *= len(regimes)
    all_polygon_points = np.zeros((total_points, 2), dtype=np.float64)
    polygon_starts = np.zeros((num_polygons,), dtype=np.uint32)
    polygon_counts = np.zeros((num_polygons,), dtype=np.uint32)
    ring_identifiers = []

    polygon_index = 0
    start_index = 0
    for regime in regimes:
        for name, feature_type, feature_code, num_points, is_polygon, item_points in items:
            ring_identifiers.append(
                {'name': name,
                 'feature_code': feature_code}
            )
            last_index = start_index + num_points
            p = item_points[0:num_points]
            p[:,0] += regime
            all_polygon_points[start_index:last_index, :] = p
            polygon_starts[polygon_index] = start_index
            polygon_counts[polygon_index] = num_points
            polygon_index += 1
            total_points += num_points
            start_index = last_index

    return ("",
            all_polygon_points,
            polygon_starts,
            polygon_counts,
            ring_identifiers)


def save_bna_file(f, layer):
    update_every = 1000
    ticks = 0
    progress_log.info("TICKS=%d" % np.alen(layer.points))
    progress_log.info("Saving BNA...")
    for i, p in enumerate(layer.iter_rings()):
        polygon = p[0]
        count = np.alen(polygon)
        ident = p[1]
        f.write('"%s","%s", %d\n' % (ident['name'], ident['feature_code'], count + 1))  # extra point for closed polygon
        for j in range(count):
            f.write("%s,%s\n" % (polygon[j][0], polygon[j][1]))
            ticks += 1

            if (ticks % update_every) == 0:
                progress_log.info("TICK=%d" % ticks)
        # duplicate first point to create a closed polygon
        f.write("%s,%s\n" % (polygon[0][0], polygon[0][1]))
    progress_log.info("TICK=%d" % ticks)
    progress_log.info("Saved BNA")
