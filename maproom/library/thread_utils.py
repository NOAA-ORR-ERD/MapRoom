"""Thread utilities

"""
from requests.exceptions import HTTPError

from owslib.wms import WebMapService, ServiceException

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
    
    def request_map(self, world_rect, image_size, event=None, event_data=None):
        req = WMSRequest(self.wms, world_rect, image_size, event, event_data)
        self.send_request(req)
        return req


class WMSInitRequest(UnskippableURLRequest):
    def __init__(self, url, version='1.1.1'):
        if url.endswith("?"):
            url = url[:-1]
        UnskippableURLRequest.__init__(self, url)
        self.version = version
        self.current_layer = None
        
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
        k = self.wms.contents.keys()
        k.sort()
        self.current_layer = k[0]
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
    
    def get_image(self, wr, size):
        if self.is_valid():
            img = self.wms.getmap(layers=[self.current_layer],
                             styles=[''],
                             srs='EPSG:4326',
                             bbox=(wr[0][0], wr[0][1], wr[1][0], wr[1][1]),
                             size=size,
                             format='image/png',
                             transparent=True
                             )
            data = img.read()
        else:
            data = blank_png
        return data


class WMSRequest(BaseRequest):
    def __init__(self, wms, world_rect, image_size, manager=None, event_data=None):
        BaseRequest.__init__(self)
        self.url = "%s image @%s from %s" % (image_size, world_rect, wms.url)
        self.wms = wms
        self.world_rect = world_rect
        self.image_size = image_size
        self.manager = manager
        self.event_data = event_data
    
    def get_data_from_server(self):
        try:
            self.data = self.wms.get_image(self.world_rect , self.image_size)
            if self.manager is not None:
                self.manager.threaded_image_loaded = (self.event_data, self)
        except ServiceException, e:
            self.error = e
    
    def get_image_array(self):
        return get_numpy_from_data(self.data)


if __name__ == "__main__":
    import time
    
    wms = BackgroundWMSDownloader('http://egisws02.nos.noaa.gov/ArcGIS/services/RNC/NOAA_RNC/ImageServer/WMSServer?')
    
    test = wms.request_map(((-130.71649285948095, 41.886646386289456), (-115.31936888967233, 51.256696088097456)), (800, 800))
    test = wms.request_map(((-130.71649285948095, 41.886646386289456), (-115.31936888967233, 51.256696088097456)), (800, 800))
    test = wms.request_map(((-130.71649285948095, 41.886646386289456), (-115.31936888967233, 51.256696088097456)), (800, 800))
    while True:
        if test.is_finished:
            break
        time.sleep(1)
        print "Waiting for test..."

    out = open('wmstest.png', 'wb')
    out.write(test.data)
    out.close()
            
    wms = None
    