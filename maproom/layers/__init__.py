# flake8: noqa

from base import Layer, EmptyLayer
from line import LineLayer, LineEditLayer
from folder import RootLayer, Folder
from grid import Grid
from scale import Scale
from compass_rose import CompassRose
from triangle import TriangleLayer
from polygon import PolygonLayer, RNCLoaderLayer
from raster import RasterLayer
from wms import WMSLayer
from tiles import TileLayer
from particles import ParticleLayer
from vector_object import LineVectorObject, RectangleVectorObject, EllipseVectorObject, CircleVectorObject, OverlayScalableImageObject, OverlayTextObject, OverlayIconObject, PolylineObject, PolygonObject, AnnotationLayer, ArrowTextBoxLayer, ArrowTextIconLayer
from shapefile import PolygonShapefileLayer
from style import LayerStyle, parse_styles_from_json, styles_to_json
import state


# List for style defaults: each class of object has its own default style
styleable_vector_objects = [LineVectorObject(), RectangleVectorObject(), EllipseVectorObject(), PolylineObject(), PolygonObject(), OverlayTextObject(), OverlayIconObject(), ArrowTextBoxLayer(), ArrowTextIconLayer()]
