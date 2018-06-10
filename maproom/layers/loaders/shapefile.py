import os

from osgeo import ogr, osr

from maproom.library.shapely_utils import load_shapely2
from maproom.layers import PolygonParentLayer, PolygonBoundaryLayer, PointLayer

from .common import BaseLayerLoader

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


class ReallyOldShapefileLoader(BaseLayerLoader):
    mime = "application/x-maproom-shapefile-not-appearing-in-this-film"

    layer_types = ["shapefile"]

    extensions = [".shp", ".kml", ".json", ".geojson"]

    extension_desc = {
        ".shp": "ESRI Shapefile",
        ".kml": "KML",
        ".geojson": "GeoJSON",
        ".json": "GeoJSON",
    }

    name = "Shapefile"

    def load_layers(self, metadata, manager, **kwargs):
        """May return one or two layers; points are placed in a separate layer
        so they can be rendered and operated on like a regular points layer
        """
        layers = []
        layer = PolygonShapefileLayer(manager=manager)

        layer.load_error_string, geometry_list, point_list = load_shapely(metadata.uri)
        if (layer.load_error_string == ""):
            if geometry_list:
                progress_log.info("Creating polygon layer...")
                layer.set_geometry(geometry_list)
                layer.file_path = metadata.uri
                layer.name = os.path.split(layer.file_path)[1]
                layer.mime = self.mime
                layers.append(layer)
            if len(point_list) > 0:
                progress_log.info("Creating point layer...")
                layer = PointLayer(manager=manager)
                layer.set_data(point_list)
                layer.file_path = metadata.uri
                layer.name = os.path.split(layer.file_path)[1]
                layer.mime = self.mime
                layers.append(layer)
        else:
            layers.append(layer)  # to return the load error string
        return layers

    def save_to_local_file(self, filename, layer):
        _, ext = os.path.splitext(filename)
        desc = self.extension_desc[ext]
        write_layer_as_shapefile(filename, layer, desc)


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

    def load_layers(self, metadata, manager, **kwargs):
        """May return one or two layers; points are placed in a separate layer
        so they can be rendered and operated on like a regular points layer
        """
        file_path = metadata.uri

        layers = []
        parent = PolygonParentLayer(manager=manager)
        parent.grouped = True
        parent.file_path = metadata.uri
        parent.name = os.path.split(file_path)[1]
        parent.mime = self.mime
        layers.append(parent)

        def create_layer(points, layer_cls, name, verify_winding=False, positive_winding=True):
            layer = layer_cls(manager=manager)
            layer.name = name
            layer.set_data_from_boundary_points(points)
            if verify_winding:
                layer.verify_winding(positive_winding)
            layer.file_path = metadata.uri
            layer.mime = self.mime
            return layer

        def create_polygon(geom_sublist, name):
            points = geom_sublist[0]
            layer = create_layer(points, PolygonBoundaryLayer, name, True, True)
            layer.verify_winding()
            layers.append(layer)

            log.debug(f"geom_sublist: {geom_sublist}")
            for j, points in enumerate(geom_sublist[1:]):
                layer = create_layer(points, PolygonBoundaryLayer, f"Hole #{j+1}", True, False)
                layers.append(layer)

        parent.load_error_string, geometry_list = load_shapely2(metadata.uri)
        if (parent.load_error_string == ""):
            if geometry_list:
                progress_log.info("Creating polygon layer...")
                for i, item in enumerate(geometry_list):
                    item_type = item[0]
                    print(f"item {i}: {item_type}, {len(item[1])} points")
                    if item_type == "MultiPolygon":
                        for j, poly in enumerate(item[1:]):
                            create_polygon(poly, "MultiPolygon #{i+1}.{j+1}")
                    else:
                        if item_type == "Point":
                            points = item[1]
                            layer = create_layer(points, PointLayer, f"Points #{i+1}")
                            layers.append(layer)
                        else:
                            create_polygon(item[1:], f"Polygon #{i+1}")
        return layers

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
    print(geom_list)
    write_geometry_as_shapefile("a.kml", "test", "KML", geom_list)
    write_geometry_as_shapefile("a.shp", "test", "ESRI Shapefile", geom_list)
    write_geometry_as_shapefile("a.json", "test", "GeoJSON", geom_list)
