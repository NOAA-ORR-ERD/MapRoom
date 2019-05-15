import os
import glob

from osgeo import ogr, osr
import numpy as np

from sawx.filesystem import filesystem_path
from sawx.utils.fileutil import ExpandZip, save_to_flat_zip

from maproom.library.shapefile_utils import load_shapefile, load_bna_items
from maproom.layers import PolygonParentLayer
from ..renderer import data_types
from .common import BaseLayerLoader
from .bna import save_bna_file

import logging
log = logging.getLogger(__name__)
progress_log = logging.getLogger("progress")


def identify_loader(file_guess):
    if file_guess.is_text and file_guess.uri.lower().endswith(".bna"):
        lines = file_guess.sample_lines
        if b".KAP" not in lines[0]:
            return dict(mime="application/x-maproom-bna", loader=BNAShapefileLoader())
    if file_guess.is_zipfile:
        if file_guess.zipfile_contains_extension(".shp"):
            return dict(mime="application/x-maproom-shapefile-zip", loader=ZipShapefileLoader())
        else:
            # should we bail if it's an unknown zipfile? Can OGR open zipped data?
            pass
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


class ShapefileLoader(BaseLayerLoader):
    mime = "application/x-maproom-shapefile"

    layer_types = ["shapefile", "annotation", "polyline_obj", "polygon_obj", "rectangle_obj", "ellipse_obj", "circle_obj", "line_obj"]

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
        log.debug(f"load_layers: num points={len(point_list)}, geometry list={geometry_list}")
        if (parent.load_error_string == ""):
            geom_type = geometry_list[0]
            items = geometry_list[1:]
            if log.isEnabledFor(logging.DEBUG):
                print(geom_type)
                for item in geometry_list[1:]:
                    print(item)
                print()
            parent.set_geometry(point_list, geometry_list)
        else:
            log.error(parent.load_error_string)
        return layers

    def save_to_local_file(self, filename, layer):
        _, ext = os.path.splitext(filename)
        feature_list = layer.calc_output_feature_list()
        write_feature_list_as_shapefile(filename, feature_list, layer.manager.project.layer_canvas.projection)


class ZipShapefileLoader(ShapefileLoader):
    mime = "application/x-maproom-shapefile-zip"

    extensions = [".zip"]

    extension_desc = {
        ".zip": "Zipped ESRI Shapefile",
    }

    name = "Zipped Shapefile"

    def load_uri_as_items(self, uri):
        expanded_zip = ExpandZip(uri)
        filename = expanded_zip.find_extension(".shp")
        return load_shapefile(filename)

    def save_to_local_file(self, filename, layer):
        filename, ext = os.path.splitext(filename)
        filename += ".shp"  # force OGR to use ESRI Shapefile
        super().save_to_local_file(filename, layer)
        return filename

    def gather_save_files(self, temp_dir, uri):
        # instead of moving files, zip them up and store at uri
        files = glob.glob(os.path.join(temp_dir, "*"))
        if len(files) == 1 and os.path.is_dir(files[0]):
            files = glob.glob(os.path.join(files[0], "*"))
        save_to_flat_zip(uri, files)
        log.debug(f"gather_save_files: saved to zip {uri}")


ext_to_driver_name = {
    ".shp": "ESRI Shapefile",
    ".json": "GeoJSON",
    ".geojson": "GeoJSON",
    ".kml": "KML",
}

need_projection = set(["ESRI Shapefile"])

shape_restriction = set(["ESRI Shapefile"])


def write_feature_list_as_shapefile(filename, feature_list, projection):
    # with help from http://www.digital-geography.com/create-and-edit-shapefiles-with-python-only/
    srs = osr.SpatialReference()
    srs.ImportFromProj4(projection.srs)

    _, ext = os.path.splitext(filename)
    try:
        driver_name = ext_to_driver_name[ext]
    except KeyError:
        raise RuntimeError(f"Unknown shapefile extension '{ext}'")

    using_projection = driver_name in need_projection
    single_shape = driver_name in shape_restriction

    driver = ogr.GetDriverByName(driver_name)
    shapefile = driver.CreateDataSource(filename)
    log.debug(f"writing {filename}, driver={driver}, srs={srs}")

    file_point_index = 0

    def fill_ring(dest_ring, points, geom_info, dup_first_point=True):
        nonlocal file_point_index
        if geom_info.count == "boundary":
            # using a Boundary object
            boundary = geom_info.start_index
            index_iter = boundary.point_indexes
            first_index = index_iter[0]
            if dup_first_point and boundary.is_closed:
                dup_first_point = False
            # temporarily shadow x & y with boundary points
            if using_projection:
                rx, ry = projection(boundary.points.x, boundary.points.y)
            else:
                rx, ry = boundary.points.x, boundary.points.y
        else:
            if using_projection:
                rx, ry = projection(points.x, points.y)
            else:
                rx, ry = points.x, points.y
            if geom_info.count == "indexed":
                # using a list of point indexes
                index_iter = geom_info.start_index
                first_index = index_iter[0]
            else:
                first_index = geom_info.start_index
                index_iter = range(first_index, first_index + geom_info.count)
        # print(first_index, geom_info.count)
        for index in index_iter:
            # print(index)
            dest_ring.AddPoint(rx[index], ry[index])

            file_point_index += 1
            if file_point_index % BaseLayerLoader.points_per_tick == 0:
                progress_log.info("Saved %d points" % file_point_index)
        if dup_first_point:
            dest_ring.AddPoint(rx[first_index], ry[first_index])

    last_geom_type = None
    shapefile_layer = None
    for feature_index, feature in enumerate(feature_list):
        geom_type = feature[0]
        points = feature[1]
        if last_geom_type is None:
            last_geom_type = geom_type
        elif single_shape and last_geom_type != geom_type:
            raise RuntimeError(f"Only one geometry type may be saved to a {driver_name}. Starting writing {last_geom_type}, found {geom_type}")
        log.debug(f"writing: {geom_type}, {feature[1:]}")
        if geom_type == "Polygon":
            if shapefile_layer is None:
                shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbPolygon)
            poly = ogr.Geometry(ogr.wkbPolygon)
            for geom_info in feature[2:]:
                dest_ring = ogr.Geometry(ogr.wkbLinearRing)
                fill_ring(dest_ring, points, geom_info)
                poly.AddGeometry(dest_ring)

            layer_defn = shapefile_layer.GetLayerDefn()
            f = ogr.Feature(layer_defn)
            f.SetGeometry(poly)
            f.SetFID(feature_index)
            shapefile_layer.CreateFeature(f)
            f = None
            poly = None
        elif geom_type == "LineString":
            if shapefile_layer is None:
                shapefile_layer = shapefile.CreateLayer("test", srs, ogr.wkbLineString)
            poly = ogr.Geometry(ogr.wkbLineString)
            geom_info = feature[2]
            fill_ring(poly, points, geom_info, False)
            layer_defn = shapefile_layer.GetLayerDefn()
            f = ogr.Feature(layer_defn)
            f.SetGeometry(poly)
            f.SetFID(feature_index)
            shapefile_layer.CreateFeature(f)
            f = None
            poly = None
        elif geom_type == "Point":
            raise RuntimeError("Points should be saved as particle layers")

    # ## lets add now a second point with different coordinates:
    # point.AddPoint(474598, 5429281)
    # feature_index = 1
    # feature = osgeo.ogr.Feature(layer_defn)
    # feature.SetGeometry(point)
    # feature.SetFID(feature_index)
    # layer.CreateFeature(feature)
    shapefile = None  # garbage collection = save


def write_feature_list_as_bna(filename, feature_list, projection):
    update_every = 1000
    ticks = 0

    def write(feature_list, fh=None):
        file_point_index = 0

        def write_poly(points, geom_info, dup_first_point=True):
            nonlocal file_point_index

            if geom_info.count == "boundary":
                # using a Boundary object
                boundary = geom_info.start_index
                count = len(boundary)
                index_iter = boundary.point_indexes
                first_index = index_iter[0]
                if dup_first_point and boundary.is_closed:
                    dup_first_point = False
                # temporarily shadow x & y with boundary points
                rx, ry = boundary.points.x, boundary.points.y
            else:
                rx, ry = points.x, points.y
                if geom_info.count == "indexed":
                    # using a list of point indexes
                    index_iter = geom_info.start_index
                    count = len(index_iter)
                    first_index = index_iter[0]
                else:
                    first_index = geom_info.start_index
                    count = geom_info.count
                    index_iter = range(first_index, first_index + count)
            if dup_first_point:
                count += 1  # extra point for closed polygon

            if fh is None:
                file_point_index += count
            else:
                fh.write(f'"{geom_info.name}","{geom_info.feature_name}", {count}\n')

                # print(first_index, geom_info.count)
                for index in index_iter:
                    fh.write(f"{rx[index]},{ry[index]}\n")
                    file_point_index += 1
                    if file_point_index % BaseLayerLoader.points_per_tick == 0:
                        progress_log.info("Saved %d points" % file_point_index)

                # duplicate first point to create a closed polygon
                if dup_first_point:
                    fh.write(f"{rx[first_index]},{ry[first_index]}\n")

        for feature_index, feature in enumerate(feature_list):
            geom_type = feature[0]
            points = feature[1]
            if fh is not None:
                log.debug(f"writing: {geom_type}, {feature[2:]}")
            if geom_type == "Polygon":
                for geom_info in feature[2:]:
                    write_poly(points, geom_info)
            else:
                geom_info = feature[2]
                write_poly(points, geom_info, False)

        return file_point_index

    total = write(feature_list)
    progress_log.info(f"TICKS={total}")
    progress_log.info("Saving BNA...")
    with open(filename, "w") as fh:
        write(feature_list, fh)


class BNAShapefileLoader(ShapefileLoader):
    mime = "application/x-maproom-bna"

    extensions = [".bna"]

    extension_desc = {
        ".bna": "BNA text file",
    }

    def load_uri_as_items(self, uri):
        return load_bna_items(uri)

    def save_to_local_file(self, filename, layer):
        # write_rings_as_bna(filename, layer, layer.points, layer.rings, layer.point_adjacency_array, layer.manager.project.layer_canvas.projection)
        feature_list = layer.calc_output_feature_list()
        write_feature_list_as_bna(filename, feature_list, layer.manager.project.layer_canvas.projection)


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
