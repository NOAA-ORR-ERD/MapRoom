"""
The :mod:`maproomlib.plugin` module contains built-in plugins for various
functionality including layers, loading files, rendering, etc.

When writing your own plugin, it's easiest to model the plugin after a
built-in plugin of the same plugin type. See
:class:`maproomlib.utility.Plugin_loader` for more information on how plugins
are discovered and loaded.

.. autoclass:: maproomlib.plugin.Point_set_layer
.. autoclass:: maproomlib.plugin.Selected_point_set_layer
.. autoclass:: maproomlib.plugin.Line_set_layer
.. autoclass:: maproomlib.plugin.Selected_line_set_layer
.. autoclass:: maproomlib.plugin.Selected_whole_layer
.. autoclass:: maproomlib.plugin.Polygon_set_layer
.. autoclass:: maproomlib.plugin.Tile_set_layer
.. autoclass:: maproomlib.plugin.Label_set_layer
.. autoclass:: maproomlib.plugin.Depth_label_set_layer
.. autoclass:: maproomlib.plugin.Triangle_set_layer
.. autoclass:: maproomlib.plugin.Line_point_layer
.. autoclass:: maproomlib.plugin.Verdat_loader
.. autoclass:: maproomlib.plugin.Buoy_loader
.. autoclass:: maproomlib.plugin.Verdat_saver
.. autoclass:: maproomlib.plugin.Triangulation_error
.. autoclass:: maproomlib.plugin.Polygon_point_layer
.. autoclass:: maproomlib.plugin.Bna_loader
.. autoclass:: maproomlib.plugin.Load_polygon_error
.. autoclass:: maproomlib.plugin.Gdal_raster_layer
.. autoclass:: maproomlib.plugin.Gdal_loader
.. autoclass:: maproomlib.plugin.Load_gdal_error
.. autoclass:: maproomlib.plugin.Transformation_error
.. autoclass:: maproomlib.plugin.Dnc_loader
.. autoclass:: maproomlib.plugin.Ogr_loader
.. autoclass:: maproomlib.plugin.Maproom_loader
.. autoclass:: maproomlib.plugin.Maproom_saver
.. autoclass:: maproomlib.plugin.Selection_layer
.. autoclass:: maproomlib.plugin.Layer_selection_layer
.. autoclass:: maproomlib.plugin.Opengl_renderer
.. autoclass:: maproomlib.plugin.Composite_layer
.. autoclass:: maproomlib.plugin.Load_layer_error
.. autoclass:: maproomlib.plugin.Flag_layer
.. autoclass:: maproomlib.plugin.Tri_poly_loader
.. autoclass:: maproomlib.plugin.Moss_loader
.. autoclass:: maproomlib.plugin.Le_loader
"""

from Point_set_layer import Point_set_layer
from Selected_point_set_layer import Selected_point_set_layer
from Line_set_layer import Line_set_layer
from Selected_line_set_layer import Selected_line_set_layer
from Selected_whole_layer import Selected_whole_layer
from Polygon_set_layer import Polygon_set_layer
from Tile_set_layer import Tile_set_layer
from Label_set_layer import Label_set_layer
from Depth_label_set_layer import Depth_label_set_layer
from Triangle_set_layer import Triangle_set_layer
from Line_point_layer import Line_point_layer, Triangulation_error
from Verdat_loader import Verdat_loader
from Buoy_loader import Buoy_loader
from Verdat_saver import Verdat_saver
from Polygon_point_layer import Polygon_point_layer, Load_polygon_error
from Bna_loader import Bna_loader
from Gdal_raster_layer import Gdal_raster_layer, Load_gdal_error, Transformation_error
from Gdal_loader import Gdal_loader
from Dnc_loader import Dnc_loader
from Ogr_loader import Ogr_loader
from Maproom_loader import Maproom_loader
from Maproom_saver import Maproom_saver
from Selection_layer import Selection_layer
from Layer_selection_layer import Layer_selection_layer
from Opengl_renderer import Opengl_renderer
from Composite_layer import Composite_layer, Load_layer_error, Save_layer_error
from Flag_layer import Flag_layer
from Tri_poly_loader import Tri_poly_loader
from Moss_loader import Moss_loader
from Le_loader import Le_loader
