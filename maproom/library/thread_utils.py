"""Thread utilities

"""
from requests.exceptions import HTTPError

from owslib.wms import WebMapService
from owslib.util import ServiceException

from peppy2.utils.background_http import BackgroundHttpDownloader, BaseRequest, UnskippableURLRequest

from numpy_images import get_numpy_from_data

blank_png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00@\x00\x00\x00@\x08\x06\x00\x00\x00\xaaiq\xde\x00\x00\x00\x06bKGD\x00\xff\x00\xff\x00\xff\xa0\xbd\xa7\x93\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x07tIME\x07\xdf\t\x02\x10/\x0b\x11M\xec5\x00\x00\x00\x1diTXtComment\x00\x00\x00\x00\x00Created with GIMPd.e\x07\x00\x00\x00mIDATx\xda\xed\xdb\xb1\x01\x00 \x08\xc4@\xb1\xfa\xfdg\xa5\xc7=\xe42\xc2\xf5\xa9N\xe6ln3@'s\xcf\xf2\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0G\xb5}\x9f\x7f\x96\x0b\x08\x89\xb7w\x1e\xe3\x00\x00\x00\x00IEND\xaeB`\x82"


class BackgroundWMSDownloader(BackgroundHttpDownloader):
    def __init__(self, url, version='1.1.1'):
        self.url = url
        self.version = version
        BackgroundHttpDownloader.__init__(self)

    def get_server_config(self):
        self.wms = WMSInitRequest(self.url, self.version)
        self.send_request(self.wms)
    
    def request_map(self, world_rect, proj_rect, image_size, layer=None, event=None, event_data=None):
        req = WMSRequest(self.wms, world_rect, proj_rect, image_size, layer, event, event_data)
        self.send_request(req)
        return req


class WMSInitRequest(UnskippableURLRequest):
    def __init__(self, url, version='1.1.1'):
        if url.endswith("?"):
            url = url[:-1]
        UnskippableURLRequest.__init__(self, url)
        self.version = version
        self.current_layer = None
        self.layer_keys = []
        
    def get_data_from_server(self):
        try:
            wms = WebMapService(self.url, self.version)
            self.setup(wms)
        except ServiceException, e:
            self.error = e
        except HTTPError, e:
            print "Error contacting", self.url, e
            self.error = e
    
    def is_valid(self):
        return self.current_layer is not None
    
    def setup(self, wms):
        self.wms = wms
        self.layer_keys = self.wms.contents.keys()
        self.layer_keys.sort()
        self.current_layer = self.layer_keys[0]
        self.debug()
    
    def debug(self):
        wms = self.wms
        print wms
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

        #Available methods, their URLs, and available formats::

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
    
    def get_bbox(self, layers, wr, pr):
        types = [("102100", "p"),
                 ("102113", "p"),
                 ("3857", "p"),
                 ("900913" "p"),
                 ("4326" "w"),
                 ]
        # FIXME: only using the first layer for coord sys.  Can different
        # layers have different allowed coordinate systems?
        layer = layers[0]
        bbox = None
        c = {t.split(":",1)[1]: t for t in self.wms[layer].crsOptions}
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
            layers = [self.current_layer]
            styles = ['']
        else:
            styles = ['' for a in layers]
        corrected = []
        for name in layers:
            if not name:
                name = self.current_layer
            corrected.append(name)
        if self.is_valid():
            crs, bbox = self.get_bbox(corrected, wr, pr)
            img = self.wms.getmap(layers=corrected,
                             styles=styles,
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
    def __init__(self, wms, world_rect, proj_rect, image_size, layers=None, manager=None, event_data=None):
        BaseRequest.__init__(self)
        self.url = "%s image @%s from %s" % (image_size, world_rect, wms.url)
        self.wms = wms
        self.world_rect = world_rect
        self.proj_rect = proj_rect
        self.image_size = image_size
        self.layers = layers
        self.manager = manager
        self.event_data = event_data
    
    def get_data_from_server(self):
        try:
            self.data = self.wms.get_image(self.world_rect, self.proj_rect, self.image_size, self.layers)
            if self.manager is not None:
                self.manager.threaded_image_loaded = (self.event_data, self)
        except ServiceException, e:
            self.error = e
    
    def get_image_array(self):
        try:
            return get_numpy_from_data(self.data)
        except IOError:
            # some WMSes return HTML data instead of an image on an error
            # (usually see this when outside the bounding box)
            return get_numpy_from_data(blank_png)


if __name__ == "__main__":
    import time
    
    wr = ((-126.59861836927804, 45.49049794230259), (-118.90005638437373, 50.081373712237856))
    pr = ((-14092893.732, 5668589.93218), (-13235893.732, 6427589.93218))
    size = (857, 759)
    
    #wms = BackgroundWMSDownloader('http://egisws02.nos.noaa.gov/ArcGIS/services/RNC/NOAA_RNC/ImageServer/WMSServer?')
    #wms = BackgroundWMSDownloader('http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WMSServer?', "1.3.0")
    #wms = BackgroundWMSDownloader('http://ows.terrestris.de/osm/service?')
    #wms = BackgroundWMSDownloader('http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/ImageServer/WMSServer?', "1.3.0")
    #wms = BackgroundWMSDownloader('http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?', "1.3.0")
    wms = BackgroundWMSDownloader('http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?', "1.3.0")
    
#http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?BBOX=-8556942.2885109,4566851.4970803,-8551142.6289909,4570907.4368929&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0
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
    
    test = wms.request_map(wr, pr, size)
    test = wms.request_map(wr, pr, size)
    test = wms.request_map(wr, pr, size, layer=["0","1","2","3","4","5","6","7"])
    test = wms.request_map(wr, pr, size)
    while True:
        if test.is_finished:
            break
        time.sleep(1)
        print "Waiting for test..."

    out = open('wmstest.png', 'wb')
    out.write(test.data)
    out.close()
            
    wms = None
    