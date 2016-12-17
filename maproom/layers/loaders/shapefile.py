import os
import math
import time

from fs.opener import opener
import numpy as np
from osgeo import ogr, osr
import pyproj

from maproom.library.accumulator import accumulator
from maproom.layers import PolygonLayer

from common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")

class ShapefileLoader(BaseLayerLoader):
    mime = "application/x-maproom-shapefile"
    
    layer_types = ["polygon", "line"]
    
    extensions = [".shp"]
    
    name = "Shapefile"
    
    layer_class = PolygonLayer

    def load_layers(self, metadata, manager):
        layer = self.layer_class(manager=manager)
        
        (layer.load_error_string,
         f_polygon_points,
         f_polygon_starts,
         f_polygon_counts,
         f_polygon_identifiers) = load_shapefile(metadata.uri)
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_data(f_polygon_points, f_polygon_starts, f_polygon_counts,
                           f_polygon_identifiers)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]
    
    def save_to_local_file(self, filename, layer):
        boundaries = self.get_boundaries(layer)
        write_boundaries_as_shapefile(filename, layer, boundaries)

def get_dataset(uri):
    """Get OGR Dataset, performing URI to filename conversion since OGR
    doesn't support URIs, only files on the local filesystem
    """

    fs, relpath = opener.parse(uri)
    print "OGR:", relpath
    print "OGR:", fs
    if not fs.hassyspath(relpath):
        raise RuntimeError("Only file URIs are supported for OGR: %s" % metadata.uri)
    file_path = fs.getsyspath(relpath)
    if file_path.startswith("\\\\?\\"):  # OGR doesn't support extended filenames
        file_path = file_path[4:]
    dataset = ogr.Open(str(file_path))

    if (dataset is None):
        return ("Unable to load the shapefile " + file_path, None)

    if (dataset.GetLayerCount() < 1):
        return ("No vector layers in shapefile " + file_path, None)

    return "", dataset


def load_shapefile(uri):
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

    error, dataset = get_dataset(uri)
    if error:
        return (error, None)

    polygon_points = accumulator(block_shape=(2,), dtype=np.float64)
    polygon_starts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_counts = accumulator(block_shape=(1,), dtype = np.uint32)
    polygon_identifiers = []
    scoping_hack = [0]

    def add_polygon(points, name, feature_code):
        if len(points) < 1:
            return
        example = points[0]
        if len(example) > 2:
            points = [(p[0], p[1]) for p in points]
        num_points = len(points)
        polygon_points.extend(points)
        polygon_starts.append(scoping_hack[0])
        polygon_counts.append(num_points)
        scoping_hack[0] += num_points
        polygon_identifiers.append(
            {'name': name,
             'feature_code': feature_code}
            )

    layer = dataset.GetLayer()
    progress_log.info("TICKS=%d" % dataset.GetLayerCount())
    progress_log.info("Loading Shapefile...")

    for feature in layer:
        feature_code = 0
        name = "shapefile"

        geom = feature.GetGeometryRef()
        geo_type = geom.GetGeometryName()

        if geo_type == 'MULTIPOLYGON':
            for i in range(geom.GetGeometryCount()):
                poly = geom.GetGeometryRef(i)
                ring = poly.GetGeometryRef(i)
                points = ring.GetPoints()
                print geo_type, i, points
                add_polygon(points, geo_type, feature_code)
        elif geo_type == 'POLYGON':
            poly = geom.GetGeometryRef(0)
            points = poly.GetPoints()
            add_polygon(points, geo_type, feature_code)
        elif geo_type == 'LINESTRING':
            points = geom.GetPoints()
            # polygon layer doesn't currently support lines, so fake it by
            # reversing the points and taking the line back on itself
            backwards = list(points)
            backwards.reverse()
            points.extend(backwards)
            add_polygon(points, geo_type, feature_code)
        elif geo_type == 'POINT':
            points = geom.GetPoints()
            # polygon layer doesn't currently support points, so fake it by
            # creating tiny little triangles for each point
            x, y = points[0]
            polygon = [(x, y), (x + 0.0005, y + .001), (x + 0.001, y)]
            add_polygon(polygon, geo_type, feature_code)
        else:
            print 'unknown type: ', geo_type

    progress_log.info("TICK=%d" % 1)

    return ("",
            np.asarray(polygon_points),
            np.asarray(polygon_starts)[:, 0],
            np.asarray(polygon_counts)[:, 0],
            polygon_identifiers)

def write_boundaries_as_shapefile(filename, layer, boundaries):
    # with help from http://www.digital-geography.com/create-and-edit-shapefiles-with-python-only/
    srs = osr.SpatialReference()
    srs.ImportFromProj4(layer.manager.project.layer_canvas.projection.srs)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapefile = driver.CreateDataSource(filename)
    shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbPolygon)

    poly = ogr.Geometry(ogr.wkbPolygon)
    file_point_index = 0
    points = layer.points
    for (boundary_index, boundary) in enumerate(boundaries):
        ring = ogr.Geometry(ogr.wkbLinearRing)

        for point_index in boundary:
            ring.AddPoint(points.x[point_index], points.y[point_index])
            file_point_index += 1
            
            if file_point_index % BaseLayerLoader.points_per_tick == 0:
                progress_log.info("Saved %d points" % file_point_index)

        print "adding ring", ring
        poly.AddGeometry(ring)
        print 'Polygon area =', poly.GetArea()

    feature_index = 0
    layer_defn = shapefile_layer.GetLayerDefn()
    feature = ogr.Feature(layer_defn)
    feature.SetGeometry(poly)
    feature.SetFID(feature_index)
    shapefile_layer.CreateFeature(feature)
    feature.Destroy()
    feature = None

    # ## lets add now a second point with different coordinates:
    # point.AddPoint(474598, 5429281)
    # feature_index = 1
    # feature = osgeo.ogr.Feature(layer_defn)
    # feature.SetGeometry(point)
    # feature.SetFID(feature_index)
    # layer.CreateFeature(feature)
    shapefile.Destroy()
    shapefile = None  # garbage collection = save

if __name__ == "__main__":
    # from http://gis.stackexchange.com/questions/43436/why-does-this-simple-python-ogr-code-create-an-empty-polygon
    #create simple square polygon shapefile:
    from osgeo import ogr
    driver = ogr.GetDriverByName('ESRI Shapefile')

    datasource = driver.CreateDataSource('square.shp')
    layer = datasource.CreateLayer('layerName',geom_type=ogr.wkbPolygon)

    #create polygon object:
    myRing = ogr.Geometry(type=ogr.wkbLinearRing)
    myRing.AddPoint(0.0, 0.0)#LowerLeft
    myRing.AddPoint(0.0, 10.0)#UpperLeft
    myRing.AddPoint(10.0, 10.0)#UpperRight
    myRing.AddPoint(10.0, 0.0)#Lower Right
    myRing.AddPoint(0.0, 0.0)#close ring
    myPoly = ogr.Geometry(type=ogr.wkbPolygon)
    myPoly.AddGeometry(myRing)
    print ('Polygon area =',myPoly.GetArea() )#returns correct area of 100.0

    #create feature object with point geometry type from layer object:
    feature = ogr.Feature( layer.GetLayerDefn() )
    feature.SetGeometry(myPoly)
    layer.CreateFeature(feature)

    #flush memory
    feature.Destroy()
    datasource.Destroy()
