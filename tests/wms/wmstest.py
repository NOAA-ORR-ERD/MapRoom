from owslib.wms import WebMapService
server_url = 'http://egisws02.nos.noaa.gov/ArcGIS/services/RNC/NOAA_RNC/ImageServer/WMSServer?'
wms = WebMapService(server_url, version='1.1.1')
#wms = WebMapService('https://nos-pgeoweb04.noaa.gov/ArcGIS/services/RNC/NOAA_RNC/ImageServer/WMSServer?', version='1.1.1')
print wms.identification.type
print wms.identification.version
print wms.identification.title
print wms.identification.abstract

#Available layers::

print wms.contents
layer = '0'
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
        m['url'] = server_url[:-1]  # without the ?
        break
print wms.getOperationByName('GetMap').methods
print wms.getOperationByName('GetMap').formatOptions

#That's everything needed to make a request for imagery::

img = wms.getmap(   layers=[layer],
                    styles=[''],
                    srs='EPSG:4326',
                    bbox=(-130.71649285948095, 41.886646386289456, -115.31936888967233, 51.256696088097456),
                    size=(800, 800),
                    format='image/png',
                    transparent=True
                    )
out = open('samplewms.png', 'wb')
out.write(img.read())
out.close()
