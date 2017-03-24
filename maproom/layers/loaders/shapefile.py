import os

from osgeo import ogr, osr

from maproom.library.shapely_utils import load_shapely
from maproom.layers import PolygonShapefileLayer

from common import BaseLayerLoader

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


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

    layer_class = PolygonShapefileLayer

    def load_layers(self, metadata, manager):
        layer = self.layer_class(manager=manager)

        layer.load_error_string, geometry_list = load_shapely(metadata.uri)
        progress_log.info("Creating layer...")
        if (layer.load_error_string == ""):
            layer.set_geometry(geometry_list)
            layer.file_path = metadata.uri
            layer.name = os.path.split(layer.file_path)[1]
            layer.mime = self.mime
        return [layer]

    def save_to_local_file(self, filename, layer):
        _, ext = os.path.splitext(filename)
        desc = self.extension_desc[ext]
        write_layer_as_shapefile(filename, layer, desc)


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
    print geom_list
    write_geometry_as_shapefile("a.kml", "test", "KML", geom_list)
    write_geometry_as_shapefile("a.shp", "test", "ESRI Shapefile", geom_list)
    write_geometry_as_shapefile("a.json", "test", "GeoJSON", geom_list)
