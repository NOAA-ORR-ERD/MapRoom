# flake8: noqa

from .base import Layer, EmptyLayer
from .point import PointLayer
from .line import LineLayer, LineEditLayer
from .folder import RootLayer, Folder
from .grid import Graticule
from .scale import Scale
from .compass_rose import CompassRose
from .noaa_logo import NOAALogo
from .timestamp import Timestamp
from .triangle import TriangleLayer
from .polygon import PolygonLayer, RNCLoaderLayer
from .polygon2 import PolygonParentLayer, RingEditLayer
from .raster import RasterLayer
from .wms import WMSLayer
from .tiles import TileLayer
from .particles import ParticleLayer, valid_legend_types
from .vector_object import LineVectorObject, RectangleVectorObject, EllipseVectorObject, CircleVectorObject, OverlayScalableImageObject, OverlayTextObject, OverlayIconObject, PolylineObject, PolygonObject, AnnotationLayer, ArrowTextBoxLayer, ArrowTextIconLayer
from .shapefile import PolygonShapefileLayer
from . import state


# List for style defaults: each class of object has its own default style
styleable_layers = [LineVectorObject, PolylineObject, RectangleVectorObject, EllipseVectorObject, CircleVectorObject, PolygonObject, OverlayTextObject, OverlayIconObject, ArrowTextBoxLayer, ArrowTextIconLayer, ParticleLayer]
