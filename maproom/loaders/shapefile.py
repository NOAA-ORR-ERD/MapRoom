import os
import glob

from osgeo import ogr, osr
import numpy as np

from sawx.filesystem import filesystem_path
from sawx.utils.fileutil import ExpandZip, save_to_flat_zip

from maproom.library.shapefile_utils import load_shapefile, write_feature_list_as_shapefile
from maproom.library.bna_utils import load_bna_items, write_feature_list_as_bna
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
        write_feature_list_as_shapefile(filename, feature_list, self.points_per_tick)


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
        write_feature_list_as_bna(filename, feature_list, self.points_per_tick)


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
