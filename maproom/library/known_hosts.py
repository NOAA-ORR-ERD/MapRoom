"""Collection of known hosts for WMS and tile services

"""
from host_utils import WMSHost, TileHost, LocalTileHost, OpenTileHost

default_tile_hosts = [
    # LocalTileHost("Blank"),
    # ESRI services listed here: http://server.arcgisonline.com/ArcGIS/rest/services/
    OpenTileHost("USGS Topo", ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/"], suffix="", reverse_coords=True),
    OpenTileHost("ESRI Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI USA Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/USA_Topo_Maps/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Ocean Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Ocean Reference", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Terrain Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Satellite Imagery", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Street Map", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI NatGeo Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Shaded Relief", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/"], reverse_coords=True),

    OpenTileHost("MapQuest", ["http://otile1.mqcdn.com/tiles/1.0.0/osm/", "http://otile2.mqcdn.com/tiles/1.0.0/osm/", "http://otile3.mqcdn.com/tiles/1.0.0/osm/", "http://otile4.mqcdn.com/tiles/1.0.0/osm/"]),
    OpenTileHost("MapQuest Satellite", ["http://otile1.mqcdn.com/tiles/1.0.0/sat/", "http://otile2.mqcdn.com/tiles/1.0.0/sat/", "http://otile3.mqcdn.com/tiles/1.0.0/sat/", "http://otile4.mqcdn.com/tiles/1.0.0/sat/"]),
    OpenTileHost("OpenStreetMap", ["http://a.tile.openstreetmap.org/", "http://b.tile.openstreetmap.org/", "http://c.tile.openstreetmap.org/"]),
    OpenTileHost("Navionics", ["http://backend.navionics.io/tile/"], suffix="?LAYERS=config_2_20.00_0&TRANSPARENT=FALSE&UGC=TRUE&navtoken=TmF2aW9uaWNzX2ludGVybmFscHVycG9zZV8wMDAwMSt3ZWJhcHAubmF2aW9uaWNzLmNvbQ%3D%3D"),
    ]

default_wms_hosts = [
#    WMSHost("USGS National Atlas 1 Million", "http://webservices.nationalatlas.gov/wms/1million?", "1.3.0", "1 Million Scale - "),
    WMSHost("NOAA RNC", "http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/ImageServer/WMSServer?", "1.3.0"),
    WMSHost("NOAA Maritime Charts", "http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?", "1.3.0"),
    WMSHost("USACE Inland ENC", "http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?", "1.3.0", default_layer_indexes=[1]),
    WMSHost("OpenStreetMap WMS Deutschland", "http://ows.terrestris.de/osm/service?", "1.1.1"),
    WMSHost("USGS Topo", "http://basemap.nationalmap.gov/arcgis/services/USGSTopo/MapServer/WMSServer?", "1.3.0"),
    WMSHost("USGS Topo Large", "http://services.nationalmap.gov/arcgis/services/USGSTopoLarge/MapServer/WMSServer?", "1.3.0"),
    WMSHost("USGS Imagery Topo Large", "http://services.nationalmap.gov/arcgis/services/USGSImageryTopoLarge/MapServer/WMSServer?", "1.3.0"),
    WMSHost("USGS National Atlas Map Reference", "http://webservices.nationalatlas.gov/wms/map_reference?", "1.3.0", "Map Reference - "),
    WMSHost("USGS National Atlas 1 Million", "http://webservices.nationalatlas.gov/wms/1million?", "1.3.0", "1 Million Scale - "),
    WMSHost("NRL", "http://geoint.nrlssc.navy.mil/nrltileserver/wms/fast?", "1.1.1"),
    ]
