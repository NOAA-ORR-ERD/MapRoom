"""Thread utilities

"""
from requests.exceptions import HTTPError

from owslib.wms import WebMapService
from owslib.util import ServiceException

from omnivore.utils.background_http import BackgroundHttpDownloader, BaseRequest, UnskippableURLRequest

from numpy_images import get_numpy_from_data

import rect
from host_utils import WMSHost, HostCache

import logging
log = logging.getLogger(__name__)


blank_png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00@\x00\x00\x00@\x08\x06\x00\x00\x00\xaaiq\xde\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xdf\t\x02\x10/\x0b\x11M\xec5\x00\x00\x00\x1diTXtComment\x00\x00\x00\x00\x00Created with GIMPd.e\x07\x00\x00\x00mIDATx\xda\xed\xdb\xb1\x01\x00 \x08\xc4@\xb1\xfa\xfdg\xa5\xc7=\xe42\xc2\xf5\xa9N\xe6ln3@'s\xcf\xf2\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0G\xb5}\x9f\x7f\x96\x0b\x08\x89\xb7w\x1e\xe3\x00\x00\x00\x00IEND\xaeB`\x82"


class BackgroundWMSDownloader(BackgroundHttpDownloader, HostCache):
    def __init__(self, wmshost):
        HostCache.__init__(self, wmshost)
        BackgroundHttpDownloader.__init__(self)

    def get_server_config(self):
        self.server = WMSRequestServer(self.host)
        self.send_request(self.server)

    def request_map(self, world_rect, proj_rect, image_size, layer=None, event=None, event_data=None):
        req = WMSRequest(self.server, world_rect, proj_rect, image_size, layer, event, event_data)
        self.send_request(req)
        return req


class WMSRequestServer(UnskippableURLRequest):
    def __init__(self, wmshost):
        self.wmshost = wmshost
        UnskippableURLRequest.__init__(self, wmshost.url)
        self.current_layer = None
        self.layer_keys = []
        self.world_bbox_rect = None

    def get_wms(self):
        wms = WebMapService(self.url, self.wmshost.version)
        return wms

    def get_data_from_server(self):
        # if True:  # To test error handling, uncomment this
        #     import time
        #     time.sleep(1)
        #     self.error = "Test error"
        #     return
        try:
            wms = self.get_wms()
            self.setup(wms)
        except ServiceException, e:
            self.error = e
        except HTTPError, e:
            log.error("Error contacting %s: %s" % (self.url, e))
            self.error = e
        except AttributeError, e:
            log.error("Bad response from server %s: %s" % (self.url, e))
            self.error = e
        except Exception, e:
            log.error("Server error %s: %s" % (self.url, e))
            self.error = e

    def is_valid(self):
        return self.current_layer is not None

    def setup(self, wms):
        self.wms = wms
        self.layer_keys = self.wms.contents.keys()
        self.layer_keys.sort()
        self.current_layer = self.layer_keys[0]
        self.world_bbox_rect = self.get_global_bbox()
        self.debug()

    def get_global_bbox(self):
        bbox = ((None, None), (None, None))
        for name in self.layer_keys:
            b = self.wms[name].boundingBoxWGS84
            r = ((b[0], b[1]), (b[2], b[3]))
            bbox = rect.accumulate_rect(bbox, r)
        return bbox

    def debug(self):
        wms = self.wms
        print wms
        print "identification: Title: ", wms.identification.title
        print "identification: Abstract: ", wms.identification.abstract
        print "contents", wms.contents
        layer = self.current_layer
        print "layer index", layer
        print "title", wms[layer].title
        print "bounding box", wms[layer].boundingBoxWGS84
        print "crsoptions", wms[layer].crsOptions
        print "styles", wms[layer].styles
        #    {'pseudo_bright': {'title': 'Pseudo-color image (Uses IR and Visual bands,
        #    542 mapping), gamma 1.5'}, 'pseudo': {'title': '(default) Pseudo-color
        #    image, pan sharpened (Uses IR and Visual bands, 542 mapping), gamma 1.5'},
        #    'visual': {'title': 'Real-color image, pan sharpened (Uses the visual
        #    bands, 321 mapping), gamma 1.5'}, 'pseudo_low': {'title': 'Pseudo-color
        #    image, pan sharpened (Uses IR and Visual bands, 542 mapping)'},
        #    'visual_low': {'title': 'Real-color image, pan sharpened (Uses the visual
        #    bands, 321 mapping)'}, 'visual_bright': {'title': 'Real-color image (Uses
        #    the visual bands, 321 mapping), gamma 1.5'}}

        # Available methods, their URLs, and available formats::

        print [op.name for op in wms.operations]
        print wms.getOperationByName('GetMap').methods

        # The NOAA server returns a bad URL (not fully specified or maybe just old),
        # so replace it with the server URL used above.  This prevents patching the
        # wms code.
        all_methods = wms.getOperationByName('GetMap').methods
        for m in all_methods:
            if m['type'].lower() == 'get':
                m['url'] = self.url
                break
        print wms.getOperationByName('GetMap').methods
        print wms.getOperationByName('GetMap').formatOptions

        for name in self.layer_keys:
            print "layer:", name, "title", self.wmshost.convert_title(self.wms[name].title), "crsoptions", wms[layer].crsOptions

    def get_layer_info(self):
        layer_info = []
        for name in self.layer_keys:
            layer_info.append((name, self.wmshost.convert_title(self.wms[name].title)))
        return layer_info

    def get_default_layers(self):
        host_default = self.wmshost.get_default_layer_indexes()
        return [self.layer_keys[i] for i in host_default if i < len(self.layer_keys)]

    def get_bbox(self, layers, wr, pr):
        types = [("102100", "p"),
                 ("102113", "p"),
                 ("3857", "p"),
                 ("900913", "p"),
                 ("4326", "w"),
                 ]
        # FIXME: only using the first layer for coord sys.  Can different
        # layers have different allowed coordinate systems?
        layer = layers[0]
        bbox = None
        c = {t.split(":", 1)[1]: t for t in self.wms[layer].crsOptions}
        for crs, which in types:
            if crs in c:
                if which == "p":
                    bbox = (pr[0][0], pr[0][1], pr[1][0], pr[1][1])
                else:
                    bbox = (wr[0][0], wr[0][1], wr[1][0], wr[1][1])
                break
        if bbox is None:
            bbox = (wr[0][0], wr[0][1], wr[1][0], wr[1][1])
        return c[crs], bbox

    def get_image(self, wr, pr, size, layers=None):
        if layers is None:
            layers = self.get_default_layers()
        corrected = []
        for name in layers:
            if not name:
                name = self.current_layer
            corrected.append(name)
        if self.is_valid():
            crs, bbox = self.get_bbox(corrected, wr, pr)
            img = self.wms.getmap(layers=corrected,
                                  #                             styles=styles,
                                  srs=crs,
                                  bbox=bbox,
                                  size=size,
                                  format='image/png',
                                  transparent=True
                                  )
            data = img.read()
        else:
            data = blank_png
        return data


class WMSRequest(BaseRequest):
    def __init__(self, server, world_rect, proj_rect, image_size, layers=None, manager=None, event_data=None):
        BaseRequest.__init__(self)
        self.url = "%s image @%s from %s" % (image_size, world_rect, server.url)
        self.server = server
        self.world_rect = world_rect
        self.proj_rect = proj_rect
        self.image_size = image_size
        self.layers = layers
        self.manager = manager
        self.event_data = event_data

    def get_data_from_server(self):
        try:
            self.data = self.server.get_image(self.world_rect, self.proj_rect, self.image_size, self.layers)

            if not rect.intersects(self.world_rect, self.server.world_bbox_rect):
                self.error = "Outside WMS boundary of %s" % rect.pretty_format(self.server.world_bbox_rect)
        except ServiceException, e:
            self.error = e
        except Exception, e:
            self.error = e
        if self.manager is not None:
            self.manager.threaded_image_loaded = (self.event_data, self)

    def get_image_array(self):
        try:
            return get_numpy_from_data(self.data)
        except (IOError, TypeError):
            # some WMSes return HTML data instead of an image on an error
            # (usually see this when outside the bounding box)
            return get_numpy_from_data(blank_png)


if __name__ == "__main__":
    import sys
    import time

    wr = ((-126.59861836927804, 45.49049794230259), (-118.90005638437373, 50.081373712237856))
    pr = ((-14092893.732, 5668589.93218), (-13235893.732, 6427589.93218))
    size = (857, 759)

# http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?BBOX=-8556942.2885109,4566851.4970803,-8551142.6289909,4570907.4368929&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0
# Capabilities: http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0

    # FAILS:
    # http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?LAYERS=0&STYLES=&WIDTH=800&FORMAT=image%2Fpng&CRS=EPSG%3A4326&REQUEST=GetMap&HEIGHT=800&BGCOLOR=0xFFFFFF&VERSION=1.3.0&BBOX=41.886646386289456%2C-130.71649285948095%2C51.256696088097456%2C-115.31936888967233&EXCEPTIONS=XML&TRANSPARENT=TRUE
    # WORKS:
    # http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?BBOX=-8556942.2885109,4566851.4970803,-8551142.6289909,4570907.4368929&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0
    # FAILS:
    # http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?LAYERS=0&STYLES=&WIDTH=800&FORMAT=image%2Fpng&REQUEST=GetMap&HEIGHT=800&BGCOLOR=0xFFFFFF&VERSION=1.3.0&EXCEPTIONS=XML&TRANSPARENT=TRUE&BBOX=-8556942.2885109,4566851.4970803,-8551142.6289909,4570907.4368929&SRS=EPSG%3A102113

# http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?BBOX=-9151321.3960644,4688981.1460582,-9128122.757984,4705204.9053088&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0
    # Capabilities: http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0
# crsoptions ['EPSG:102100']

    url = "http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/ImageServer/WMSServer?"
    #url = "http://webservices.nationalatlas.gov/wms/map_reference?"
    #url = "http://geoint.nrlssc.navy.mil/nrltileserver/wms/fast?"

    for version in ['1.3.0', '1.1.1']:
        host = WMSHost("test", url, version)
        print host
        downloader = BackgroundWMSDownloader(host)
        while True:
            if downloader.server.is_finished:
                break
            time.sleep(.1)
            print "Waiting for server config..."
        if downloader.server.is_valid():
            break

    if not downloader.server.is_valid():
        print downloader.server.error
    sys.exit()

    h = WMSHost.get_wms_by_name("USACE Inland ENC")
#    h = WMSHost.get_wms_by_name("NOAA RNC")
    downloader = BackgroundWMSDownloader(h)

    test = downloader.request_map(wr, pr, size)
    test = downloader.request_map(wr, pr, size)
    test = downloader.request_map(wr, pr, size, layer=["0", "1", "2", "3", "4", "5", "6", "7"])
    test = downloader.request_map(wr, pr, size, layer=['10', '11', '12', '14', '15', '17', '18', '19', '20'])
    test = downloader.request_map(wr, pr, size)
    while True:
        if test.is_finished:
            break
        time.sleep(1)
        print "Waiting for test..."

    if test.error:
        print "Error!", test.error
    else:
        print "world bbox", downloader.server.world_bbox_rect
        outfile = 'wmstest.png'
        out = open(outfile, 'wb')
        out.write(test.data)
        out.close()
        print "Generated image", outfile

    downloader = None
