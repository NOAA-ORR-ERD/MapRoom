# Find out what a WMTS has to offer. Service metadata:

from owslib.wmts import WebMapTileService
#wmts = WebMapTileService("http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi")
wmts = WebMapTileService("http://map1.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi")
print wmts.identification.type
#'OGC WMTS'
print wmts.identification.version
#'1.0.0'
print wmts.identification.title
#'NASA Global Imagery Browse Services for EOSDIS'
print str.strip(wmts.identification.abstract)
#'Near real time imagery from multiple NASA instruments'
print wmts.identification.keywords
#['World', 'Global']


#Service Provider:

print wmts.provider.name
#'National Aeronautics and Space Administration'
print wmts.provider.url
#'http://earthdata.nasa.gov/'


#Available Layers:

print len(wmts.contents.keys()) > 0
#True
print sorted(list(wmts.contents))[0]
#'AIRS_CO_Total_Column_Day'


# Fetch a tile (using some defaults):

tile = wmts.gettile(layer='MODIS_Terra_CorrectedReflectance_TrueColor', tilematrixset='EPSG4326_250m', tilematrix='0', row=0, column=0, format="image/jpeg")
out = open('nasa_modis_terra_truecolour.jpg', 'wb')
bytes_written = out.write(tile.read())
out.close()
