""" URL/Host utilities

"""
import os
import math
import glob
from copy import deepcopy

import jsonpickle

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class HostCache(object):
    cached_known_hosts = None

    def __init__(self, host, cache_root=None):
        self.host = host
        self.server = None
        if cache_root is not None and not os.path.exists(cache_root):
            try:
                os.makedirs(cache_root)
            except os.error:
                cache_root = None
        self.cache_root = cache_root

    @classmethod
    def get_known_hosts(cls):
        if cls.cached_known_hosts is None:
            cls.cached_known_hosts = []
        return cls.cached_known_hosts

    @classmethod
    def get_host_by_name(cls, name):
        for h in cls.get_known_hosts():
            if h.name == name:
                return h
        return None

    @classmethod
    def get_host_by_url(cls, url):
        for i, h in enumerate(cls.get_known_hosts()):
            if h.is_in_url_list(url):
                return i, h
        return None, None

    @classmethod
    def add_host(cls, host):
        cls.get_known_hosts()  # ensure the list has been created
        cls.cached_known_hosts.append(host)

    @classmethod
    def set_known_hosts(cls, hostlist):
        cls.cached_known_hosts = hostlist

    def get_server_config(self):
        raise NotImplementedError

    def get_server(self):
        return self.server

    def is_valid(self):
        return self.server.is_valid()


class SortableHost(object):
    def __init__(self, name="", default=False):
        self.name = name
        self.default = default

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        """Custom jsonpickle state restore routine

        The use of jsonpickle to recreate objects doesn't go through __init__,
        so there will be missing attributes when restoring old versions of the
        json. Once a version gets out in the wild and additional attributes are
        added to a segment, a default value should be applied here.
        """
        self.default = state.pop('default', False)
        self.__dict__.update(state)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.url == other.url

    def label_helper(self):
        extra = " (default)" if self.default else ""
        return self.name + extra

    def default_helper(self, value=None):
        if value is not None:
            self.default = value
        return self.default


class WMSHost(SortableHost):
    def __init__(self, name="", url="", version="1.3.0", strip_prefix="", default_layer_indexes=None):
        SortableHost.__init__(self, name)
        if url.endswith("?"):
            url = url[:-1]
        self.url = url
        self.version = version
        self.strip_prefix = strip_prefix
        self.strip_prefix_len = len(strip_prefix)
        self.default_layer_indexes = default_layer_indexes

    def __hash__(self):
        return hash(self.url)

    def __str__(self):
        return " ".join([self.name, self.url, self.version])

    def __repr__(self):
        return "<" + " ".join([self.__class__.__name__, self.name, self.url, self.version]) + ">\n"

    def is_in_url_list(self, url):
        return url == self.url

    def convert_title(self, title):
        if self.strip_prefix:
            if title.startswith(self.strip_prefix):
                return title[self.strip_prefix_len:]
        return title

    def get_default_layer_indexes(self):
        if self.default_layer_indexes is not None:
            return self.default_layer_indexes
        return [0]


class TileHost(SortableHost):
    known_suffixes = ['.png', '']

    def __init__(self, name="host", url_list=[], strip_prefix="", tile_size=256, suffix=".png", reverse_coords=False):
        SortableHost.__init__(self, name)
        self.urls = []
        for url in url_list:
            if url.endswith("?"):
                url = url[:-1]
            if url.endswith("/"):
                url = url[:-1]
            self.urls.append(url)
        self.url_index = 0
        self.num_urls = len(self.urls)
        self.strip_prefix = strip_prefix
        self.strip_prefix_len = len(strip_prefix)
        self.tile_size = tile_size
        self.suffix = suffix
        self.reverse_coords = reverse_coords

    def __str__(self):
        return " ".join([self.name, "%d urls" % self.num_urls, str(self.default)])

    def __repr__(self):
        return " ".join([self.name, "%d urls" % self.num_urls, str(self.default)])

    def __hash__(self):
        return hash(self.urls[0])

    def __eq__(self, other):
        return set(self.urls) == set(other.urls)

    def is_in_url_list(self, url):
        return url in self.urls

    @property
    def url_format(self):
        return "z/y/x" if self.reverse_coords else "z/x/y"

    @url_format.setter
    def url_format(self, value):
        self.reverse_coords = (value == "z/y/x")

    @classmethod
    def copy_helper(cls, src):
        dest = deepcopy(src)
        dest.name = "Copy of %s" % src.name
        return dest

    # Reference for tile number calculations:
    # http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames

    def world_to_tile_num(self, zoom, lon, lat):
        zoom = int(zoom)
        if zoom == 0:
            return (0, 0)
        lat_rad = lat * math.pi / 180.0
        n = 2 << (zoom - 1)
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)

        # x values not clamped to allow wrapping across the dateline
        ytile = max(ytile, 0)
        ytile = min(ytile, n - 1)

        return (xtile, ytile)

    rad2deg = 180.0 / math.pi

    def tile_num_to_world_lb_rt(self, zoom, x, y):
        zoom = int(zoom)
        if zoom == 0:
            return ((-180.0, -85.0511287798066), (180.0, 85.0511287798066))
        n = 2 << (zoom - 1)
        lon1 = (x * 360.0 / n) - 180.0
        lon2 = ((x + 1) * 360.0 / n) - 180.0
        lat1 = math.atan(math.sinh(math.pi * (1.0 - (2.0 * (y + 1) / n)))) * self.rad2deg
        lat2 = math.atan(math.sinh(math.pi * (1.0 - (2.0 * y / n)))) * self.rad2deg
        return ((lon1, lat1), (lon2, lat2))

    def get_tile_init_request(self, cache_root):
        raise NotImplementedError

    def get_next_url(self):
        # mostly round robin URL index.  If multiple threads hit this at the
        # same time the same URLs might be used in each thread, but not worth
        # thread locking
        self.url_index = (self.url_index + 1) % self.num_urls
        url = self.urls[self.url_index]
        return url

    def get_tile_url(self, zoom, x, y):
        url = self.get_next_url()
        if self.reverse_coords:
            x, y = y, x
        n = 2 << (zoom - 1)
        x = x % n  # use modulo to wrap around
        return "%s/%s/%s/%s%s" % (url, zoom, x, y, self.suffix)

    def get_tile_cache_dir(self, cache_root):
        # >>> ".".join("http://a.tile.openstreetmap.org/".split("//")[1].split("/")[0].rsplit(".", 2)[-2:])
        # 'openstreetmap.org'
        domain = ".".join(self.urls[0].split("//")[1].split("/")[0].rsplit(".", 2)[-2:])
        name = domain + "--" + "".join(x for x in self.name if x.isalnum())
        return "%s/%s" % (cache_root, name)

    def get_tile_cache_file_template(self, cache_root):
        name = self.get_tile_cache_dir(cache_root)
        template = "%s/%%s/%%s/%%s%s" % (name, self.suffix)
        return template

    def get_tile_cache_file(self, cache_root, zoom, x, y):
        template = self.get_tile_cache_file_template(cache_root)
        n = 2 << (zoom - 1)
        x = x % n  # use modulo to wrap around
        path = template % (zoom, x, y)
        return path

    def clear_cache(self, cache_root):
        template = self.get_tile_cache_file_template(cache_root)
        path = template % ("*", "*", "*")
        # delete each file individually so there's no possibility of some wild
        # recursive delete running out of control due to some bad cache path
        for image in glob.glob(path):
            os.unlink(image)  # throws exception to calling function and let them handle it


class LocalTileHost(TileHost):
    request_type = "local"

    def __init__(self, name, tile_size=256):
        TileHost.__init__(self, name, [""], tile_size=tile_size)

    def __hash__(self):
        return hash(self.name)


class OpenTileHost(TileHost):
    request_type = "url"


class WMTSTileHost(TileHost):
    request_type = "wmts"

# jsonpickle doesn't handle the case of an object being serialized without
# __getstate__ to then be restored with __setstate__. __setstate__ is never
# called because the json data does not have the "py/state" key. Only by adding
# a custom handler can we support both formats to go through __setstate__
class ExtraAttributeHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        data["py/state"] = obj.__getstate__()
        return data

    def restore(self, data):
        # use jsonpickle convenience function to find the class
        cls = jsonpickle.unpickler.loadclass(data["py/object"])
        restored = cls.__new__(cls)

        # restore using py/state if serialized with __getstate__, or from the
        # flattened dict if the data was written out by a version of the class
        # before using __getstate__
        if "py/state" in data:
            state = data["py/state"]
        else:
            state = {k:v for k,v in data.iteritems() if not k.startswith("py/")}
        restored.__setstate__(state)
        return restored

ExtraAttributeHandler.handles(SortableHost)
ExtraAttributeHandler.handles(TileHost)
ExtraAttributeHandler.handles(LocalTileHost)
ExtraAttributeHandler.handles(OpenTileHost)
ExtraAttributeHandler.handles(WMTSTileHost)
ExtraAttributeHandler.handles(WMSHost)
