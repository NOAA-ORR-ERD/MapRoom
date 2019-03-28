import os

from osgeo import ogr, osr
import numpy as np

from sawx.filesystem import filesystem_path
from sawx.utils.fileutil import ExpandZip

from maproom.library.shapefile_utils import load_shapefile, load_bna_items
from maproom.layers import PolygonParentLayer
from ..renderer import data_types
from .common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def identify_loader(file_guess):
    if file_guess.is_zipfile and file_guess.zipfile_contains_extension(".shp"):
            return dict(mime="application/x-maproom-shapefile-zip", loader=ZipShapefileLoader())
    try:
        file_path = file_guess.filesystem_path
    except OSError:
        log.debug(f"{file_guess.uri} not on local filesystem, GDAL won't load it.")
        return None
    if file_path.startswith("\\\\?\\"):  # GDAL doesn't support extended filenames
        file_path = file_path[4:]
    try:
        dataset = ogr.Open(file_path)
    except RuntimeError:
        log.debug("OGR can't open %s; not an image")
        return None
    if dataset is not None and dataset.GetLayerCount() > 0:
        # check to see if there are any valid layers because some CSV files
        # seem to be recognized as having layers but have no geometry.
        count = 0
        for layer_index in range(dataset.GetLayerCount()):
            layer = dataset.GetLayer(layer_index)
            for feature in layer:
                ogr_geom = feature.GetGeometryRef()
                print(f"ogr_geom for {layer} = {ogr_geom}")
                if ogr_geom is None:
                    continue
                count += 1
        if count > 0:
            return dict(mime="image/x-maproom-shapefile", loader=ShapefileLoader())
    return None


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

        poly.AddGeometry(ring)

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


def write_layer_as_shapefile(filename, layer, driver_name):
    srs = osr.SpatialReference()
    srs.ImportFromProj4(layer.manager.project.layer_canvas.projection.srs)
    write_geometry_as_shapefile(filename, layer.name.encode("UTF-8"), driver_name, layer.geometry, srs)


def write_geometry_as_shapefile(filename, layer_name, driver_name, geometry_list, srs=None):
    driver = ogr.GetDriverByName(driver_name)
    shapefile = driver.CreateDataSource(filename)
    shapefile_layer = shapefile.CreateLayer(layer_name, srs, ogr.wkbUnknown)
    layer_defn = shapefile_layer.GetLayerDefn()
    feature_index = 0

    # shapefiles can't contain geometry collections, so to put different
    # geometries in the same file you have to use different features for each.
    # From http://invisibleroads.com/tutorials/gdal-shapefile-geometries-fields-save-load.html
    for geom in geometry_list:
        feature = ogr.Feature(layer_defn)
        g = ogr.CreateGeometryFromWkt(geom.wkt)
        feature.SetGeometry(g)
        feature.SetFID(feature_index)
        shapefile_layer.CreateFeature(feature)
        feature.Destroy()
        feature_index += 1


class ShapefileLoader(BaseLayerLoader):
    mime = "application/x-maproom-shapefile"

    layer_types = ["shapefile"]

    extensions = [".shp", ".kml", ".json", ".geojson"]

    extension_desc = {
        ".shp": "ESRI Shapefile",
        ".kml": "KML",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
    }

    name = "Shapefile"

    def load_uri_as_items(self, uri):
        return load_shapefile(uri)

    def load_layers(self, uri, manager, **kwargs):
        """May return one or two layers; points are placed in a separate layer
        so they can be rendered and operated on like a regular points layer
        """
        file_path = uri

        layers = []
        parent = PolygonParentLayer(manager=manager)
        # parent.grouped = True
        parent.grouped = False
        parent.file_path = uri
        parent.name = os.path.split(file_path)[1]
        parent.mime = self.mime
        layers.append(parent)

        parent.load_error_string, geometry_list, point_list = self.load_uri_as_items(uri)
        geom_type = geometry_list[0]
        items = geometry_list[1:]
        if log.isEnabledFor(logging.DEBUG):
            print(geom_type)
            for item in geometry_list[1:]:
                print(item)
            print()
        if (parent.load_error_string == ""):
            parent.set_geometry(point_list, geometry_list)
        return layers

    def save_to_local_file(self, filename, layer):
        _, ext = os.path.splitext(filename)
        desc = self.extension_desc[ext]
        if ext == ".bna":
            write_rings_as_bna(filename, layer, layer.points, layer.rings, layer.point_adjacency_array, layer.manager.project.layer_canvas.projection)
        else:
            write_rings_as_shapefile(filename, layer, layer.points, layer.rings, layer.ring_adjacency, layer.manager.project.layer_canvas.projection)


class ZipShapefileLoader(ShapefileLoader):
    mime = "application/x-maproom-shapefile-zip"

    def load_uri_as_items(self, uri):
        expanded_zip = ExpandZip(uri)
        filename = expanded_zip.find_extension(".shp")
        return load_shapefile(filename)


def write_rings_as_shapefile(filename, layer, points, rings, adjacency, projection):
    # with help from http://www.digital-geography.com/create-and-edit-shapefiles-with-python-only/
    srs = osr.SpatialReference()
    srs.ImportFromProj4(projection.srs)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapefile = driver.CreateDataSource(filename)
    print(f"writing {filename}, srs={srs}")
    shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbPolygon)

    file_point_index = 0
    ring_index = 0
    feature_index = 0

    def add_poly():
        nonlocal poly
        if poly is not None:
            layer_defn = shapefile_layer.GetLayerDefn()
            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(poly)
            feature.SetFID(feature_index)
            shapefile_layer.CreateFeature(feature)
            feature = None
        poly = ogr.Geometry(ogr.wkbPolygon)

    poly = None
    while ring_index < len(rings):
        point_index = first_index = int(rings.start[ring_index])
        count = -adjacency[first_index]['point_flag']
        feature_code = 0
        dup_first_point = False
        if count < 2:
            dest_ring = ogr.Geometry(ogr.wkbPoint)
        elif count == 2:
            dest_ring = ogr.Geometry(ogr.wkbLineString)
        else:
            feature_code = adjacency[first_index+1]['state']
            dest_ring = ogr.Geometry(ogr.wkbLinearRing)
            dup_first_point = True
        if feature_code >= 0:
            add_poly()
        
        x, y = projection(points.x[first_index:first_index+count], points.y[first_index:first_index+count])
        for index in range(0, count):
            dest_ring.AddPoint(x[index], y[index])

            file_point_index += 1
            if file_point_index % BaseLayerLoader.points_per_tick == 0:
                progress_log.info("Saved %d points" % file_point_index)
        if dup_first_point:
            dest_ring.AddPoint(x[0], y[0])
        poly.AddGeometry(dest_ring)
        ring_index += 1
    add_poly()

    # ## lets add now a second point with different coordinates:
    # point.AddPoint(474598, 5429281)
    # feature_index = 1
    # feature = osgeo.ogr.Feature(layer_defn)
    # feature.SetGeometry(point)
    # feature.SetFID(feature_index)
    # layer.CreateFeature(feature)
    shapefile = None  # garbage collection = save


def write_rings_as_bna(filename, layer, points, rings, adjacency, projection):
    update_every = 1000
    ticks = 0
    progress_log.info("TICKS=%d" % np.alen(points))
    progress_log.info("Saving BNA...")
    with open(filename, "w") as fh:
        file_point_index = 0
        ring_index = 0
        feature_index = 0
        while ring_index < len(rings):
            point_index = int(rings.start[ring_index])
            count = 0
            geom = layer.geometry_list[ring_index]
            fh.write('"%s","%s", %d\n' % (geom.name, geom.feature_name, rings.count[ring_index] + 1))  # extra point for closed polygon
            # print(f"starting ring={ring_index}, start={point_index} {type(point_index)} count={rings.count[ring_index]}")
            while count < rings.count[ring_index]:
                # print(f"ring:{ring_index}, point_index={point_index} x={points.x[point_index]} y={points.y[point_index]}")
                fh.write("%s,%s\n" % (points.x[point_index], points.y[point_index]))
                count += 1
                point_index = adjacency.next[point_index]

                file_point_index += 1
                if file_point_index % BaseLayerLoader.points_per_tick == 0:
                    progress_log.info("Saved %d points" % file_point_index)

            # duplicate first point to create a closed polygon
            point_index = int(rings.start[ring_index])
            fh.write("%s,%s\n" % (points.x[point_index], points.y[point_index]))
            ring_index += 1


class BNAShapefileLoader(ShapefileLoader):
    mime = "application/x-maproom-bna"

    extensions = [".bna"]

    extension_desc = {
        ".bna": "BNA text file",
    }

    def load_uri_as_items(self, uri):
        return load_bna_items(uri)


if __name__ == "__main__":
    # from http://gis.stackexchange.com/questions/43436/why-does-this-simple-python-ogr-code-create-an-empty-polygon
    # create simple square polygon shapefile:
    # driver = ogr.GetDriverByName('ESRI Shapefile')

    # datasource = driver.CreateDataSource('square.shp')
    # layer = datasource.CreateLayer('layerName',geom_type=ogr.wkbPolygon)

    # #create polygon object:
    # myRing = ogr.Geometry(type=ogr.wkbLinearRing)
    # myRing.AddPoint(0.0, 0.0)#LowerLeft
    # myRing.AddPoint(0.0, 10.0)#UpperLeft
    # myRing.AddPoint(10.0, 10.0)#UpperRight
    # myRing.AddPoint(10.0, 0.0)#Lower Right
    # myRing.AddPoint(0.0, 0.0)#close ring
    # myPoly = ogr.Geometry(type=ogr.wkbPolygon)
    # myPoly.AddGeometry(myRing)
    # print ('Polygon area =',myPoly.GetArea() )#returns correct area of 100.0

    # #create polygon object:
    # hole = ogr.Geometry(type=ogr.wkbLinearRing)
    # hole.AddPoint(3.0, 3.0)#LowerLeft
    # hole.AddPoint(7.0, 3.0)#Lower Right
    # hole.AddPoint(7.0, 7.0)#UpperRight
    # hole.AddPoint(3.0, 7.0)#UpperLeft
    # hole.AddPoint(3.0, 3.0)#close ring
    # myPoly.AddGeometry(hole)
    # print ('Polygon area =',myPoly.GetArea() )#returns correct area of 100.0

    # #create feature object with point geometry type from layer object:
    # feature = ogr.Feature( layer.GetLayerDefn() )
    # feature.SetGeometry(myPoly)
    # layer.CreateFeature(feature)

    # #flush memory
    # feature.Destroy()
    # datasource.Destroy()

    error, geom_list = load_shapely('/noaa/maproom/TestData/Shapefiles/20160516_0013d.shp')
    print(geom_list)
    write_geometry_as_shapefile("a.kml", "test", "KML", geom_list)
    write_geometry_as_shapefile("a.shp", "test", "ESRI Shapefile", geom_list)
    write_geometry_as_shapefile("a.json", "test", "GeoJSON", geom_list)
