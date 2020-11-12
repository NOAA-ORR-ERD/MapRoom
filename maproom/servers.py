import os

from maproom.app_framework import persistence

from .library.thread_utils import BackgroundWMSDownloader
from .library.tile_utils import BackgroundTileDownloader
from .library.host_utils import OpenTileHost, WMSHost

import logging
log = logging.getLogger(__name__)


default_tile_hosts = [
    # LocalTileHost("Blank"),
    # ESRI services listed here: http://server.arcgisonline.com/ArcGIS/rest/services/
    OpenTileHost("USGS Topo", ["https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/"], suffix="", reverse_coords=True),
    OpenTileHost("ESRI Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI USA Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/USA_Topo_Maps/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Ocean Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Ocean Reference", ["http://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Reference/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Terrain Base", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Satellite Imagery", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"], reverse_coords=True, default=True),
    OpenTileHost("ESRI Street Map", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI NatGeo Topographic", ["http://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/"], reverse_coords=True),
    OpenTileHost("ESRI Shaded Relief", ["http://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/"], reverse_coords=True),

    OpenTileHost("OpenStreetMap", ["http://a.tile.openstreetmap.org/", "http://b.tile.openstreetmap.org/", "http://c.tile.openstreetmap.org/"]),
    OpenTileHost("Navionics", ["http://backend.navionics.io/tile/"], suffix="?LAYERS=config_2_20.00_0&TRANSPARENT=FALSE&UGC=TRUE&navtoken=TmF2aW9uaWNzX2ludGVybmFscHVycG9zZV8wMDAwMSt3ZWJhcHAubmF2aW9uaWNzLmNvbQ%3D%3D"),
]

default_wms_hosts = [
    #    WMSHost("USGS National Atlas 1 Million", "http://webservices.nationalatlas.gov/wms/1million?", "1.3.0", "1 Million Scale - "),
    WMSHost("NOAA RNC", "http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/ImageServer/WMSServer?", "1.3.0", default=True),
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


downloaders = {}

def stop_threaded_processing():
    global downloaders

    log.debug("Stopping threaded services...")
    while len(downloaders) > 0:
        url, wms = downloaders.popitem()
        log.debug("Stopping threaded downloader %s" % wms)
        wms.stop_threads()
    log.debug("Stopped threaded services.")

    import threading
    for thread in threading.enumerate():
        log.debug("thread running: %s" % thread.name)


# WMS Servers

def get_threaded_wms(host=None):
    if host is None:
        host = BackgroundWMSDownloader.get_known_hosts()[0]
    if host.url not in downloaders:
        wms = BackgroundWMSDownloader(host)
        downloaders[host.url] = wms
    return downloaders[host.url]

def get_wms_server_by_id(id):
    host = BackgroundWMSDownloader.get_known_hosts()[id]
    return host

def get_wms_server_id_from_url(url):
    index, host = BackgroundWMSDownloader.get_host_by_url(url)
    return index

def get_threaded_wms_by_id(id):
    host = get_wms_server_by_id(id)
    return get_threaded_wms(host)

def get_known_wms_names():
    return [s.name for s in BackgroundWMSDownloader.get_known_hosts()]

def get_default_wms_id():
    index, host = BackgroundWMSDownloader.get_default_host()
    return index

def remember_wms():
    hosts = BackgroundWMSDownloader.get_known_hosts()
    persistence.save_json_data("wms_servers", hosts)


# Tile servers

def get_tile_cache_root():
    return persistence.get_cache_dir("tiles")

def get_tile_downloader(host=None):
    if host is None:
        host = BackgroundTileDownloader.get_known_hosts()[0]
    if host not in downloaders:
        cache_dir = get_tile_cache_root()
        ts = BackgroundTileDownloader(host, cache_dir)
        downloaders[host] = ts
    return downloaders[host]

def get_tile_downloader_by_id(id):
    host = get_tile_server_by_id(id)
    return get_tile_downloader(host)

def get_tile_server_by_id(id):
    host = BackgroundTileDownloader.get_known_hosts()[id]
    return host

def get_tile_server_id_from_url(url):
    index, host = BackgroundTileDownloader.get_host_by_url(url)
    return index

def get_known_tile_server_names():
    return [s.name for s in BackgroundTileDownloader.get_known_hosts()]

def get_default_tile_server_id():
    index, host = BackgroundTileDownloader.get_default_host()
    return index

def remember_tile_servers():
    hosts = BackgroundTileDownloader.get_known_hosts()
    persistence.save_json_data("tile_servers", hosts)

# persistence

def restore_from_last_time():
    hosts = persistence.get_json_data("wms_servers")
    if hosts is None:
        hosts = default_wms_hosts
    BackgroundWMSDownloader.set_known_hosts(hosts)

    hosts = persistence.get_json_data("tile_servers")
    if hosts is None:
        hosts = default_tile_hosts
    BackgroundTileDownloader.set_known_hosts(hosts)

def remember_for_next_time():
    remember_wms()
    remember_tile_servers()
