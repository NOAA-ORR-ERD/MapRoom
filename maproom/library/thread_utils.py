"""Thread utilities

"""

from owslib.wms import WebMapService, ServiceException

from peppy2.utils.background_http import BackgroundHttpDownloader, BaseRequest, UnskippableURLRequest

from numpy_images import get_numpy_from_data


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
        img = self.wms.getmap(layers=[self.current_layer],
                         styles=[''],
                         srs='EPSG:4326',
                         bbox=(wr[0][0], wr[0][1], wr[1][0], wr[1][1]),
                         size=size,
                         format='image/png',
                         transparent=True
                         )
        data = img.read()
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
    