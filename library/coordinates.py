# coding=utf8

import unittest

import app_globals

def float_to_degrees_minutes_seconds( value, directions = None ):
    direction = ""
    if directions:
        if value > 0:
            direction = directions[ 0 ]
        elif value < 0:
            direction = directions[ 1 ]
    
    value = abs( value )
    degrees = int( value )
    value = ( value - degrees ) * 60
    minutes = int( value )
    seconds = int( ( value - minutes ) * 60 )
    
    if degrees == 0 and minutes == 0 and seconds == 0:
        direction = ""
    
    return ( degrees, minutes, seconds, direction )

def float_to_degrees_minutes( value, directions = None ):
    direction = ""
    if directions:
        if value > 0:
            direction = directions[ 0 ]
        elif value < 0:
            direction = directions[ 1 ]
    
    value = abs( value )
    degrees = int( value )
    minutes = ( value - degrees ) * 60
    
    return ( degrees, minutes, direction )

def float_to_degrees( value, directions = None ):
    direction = ""
    if directions:
        if value > 0:
            direction = directions[ 0 ]
        elif value < 0:
            direction = directions[ 1 ]
    
    degrees = abs( value )
    
    return ( degrees, direction )
    
def degrees_minutes_seconds_to_float( degrees ):
    values = degrees.strip().split(" ")
    dir = ""
    if len(values) == 3:
        degrees, minutes, seconds = values
    else:
        degrees, minutes, seconds, dir = values
    
    result = float(degrees.strip())
    result += float(minutes.strip()) / 60.0
    result += float(seconds.strip()) / 3600.0

    if dir in ["W", "S"]:
        result *= -1
    
    return result
    
def degrees_minutes_to_float( degrees ):
    degrees, minutes, dir = degrees.strip().split(" ")
    
    result = float(degrees.strip())
    result += float(minutes.strip()) / 60.0

    if dir in ["W", "S"]:
        result *= -1
    
    return result
    
def degrees_to_float( degrees ):
    degrees, dir = degrees.strip().split(" ")
    result = float(degrees.strip())
    dir = dir.strip()
    if dir in ["W", "S"]:
        result *= -1
        
    return result    

def format_lat_lon_degrees_minutes_seconds( longitude, latitude ):
    lon = u"%d° %d′ %d″ %s" % \
        float_to_degrees_minutes_seconds( longitude, directions = ( "E", "W" ) )
    lat = u"%d° %d′ %d″ %s" % \
        float_to_degrees_minutes_seconds( latitude, directions = ( "N", "S" ) )
    
    return u"%s, %s" % ( lat, lon )

def format_lat_lon_degrees_minutes( longitude, latitude ):
    lon = u"%d° %0.4f′ %s" % \
        float_to_degrees_minutes( longitude, directions = ( "E", "W" ) )
    lat = u"%d° %0.4f′ %s" % \
        float_to_degrees_minutes( latitude, directions = ( "N", "S" ) )
    
    return u"%s, %s" % ( lat, lon )

def format_lat_lon_degrees( longitude, latitude ):
    lon = u"%0.6f° %s" % \
        float_to_degrees( longitude, directions = ( "E", "W" ) )
    lat = u"%0.6f° %s" % \
        float_to_degrees( latitude, directions = ( "N", "S" ) )
    
    return u"%s, %s" % ( lat, lon )

def format_coords_for_display( longitude, latitude ):
    display_format = app_globals.preferences["Coordinate Display Format"]
    if display_format == "degrees minutes seconds":
        return format_lat_lon_degrees_minutes_seconds( longitude, latitude )
    elif display_format == "decimal degrees":
        return format_lat_lon_degrees( longitude, latitude )
    else:
        if display_format != "degrees decimal minutes":
            print "ERROR: Unknown format %s specified when formatting coordinates" % display_format
        return format_lat_lon_degrees_minutes( longitude, latitude )

def format_lat_line_label( latitude ):
    ( degrees, minutes, direction ) = \
        float_to_degrees_minutes( latitude, directions = ( "N", "S" ) )
    
    minutes = round( minutes )
    if minutes == 60:
        minutes = 0
        degrees += 1
    
    return u" %d° %d' %s " % ( degrees, minutes, direction )

def format_lon_line_label( longitude ):
    ( degrees, minutes, direction ) = \
        float_to_degrees_minutes( longitude, directions = ( "E", "W" ) )
    
    minutes = round( minutes )
    if minutes == 60:
        minutes = 0
        degrees += 1
    
    return u" %d° %d' %s " % ( degrees, minutes, direction )

def lat_lon_from_degrees_minutes( lat_lon_string ):
    lat_lon_string = lat_lon_string.replace(u"°", "").replace(u"′", "")
    lon, lat = lat_lon_string.split(",")
    try:
        return (degrees_minutes_to_float(lat), degrees_minutes_to_float(lon))
    except Exception, e:
        import traceback
        print traceback.format_exc(e)
        return (-1, -1)
    
def lat_lon_from_degrees_minutes_seconds( lat_lon_string ):
    lat_lon_string = lat_lon_string.replace(u"°", "").replace(u"″", "").replace(u"′", "")
    lon, lat = lat_lon_string.split(",")
    try:
        return (degrees_minutes_seconds_to_float(lat), degrees_minutes_seconds_to_float(lon))
    except Exception, e:
        import traceback
        print traceback.format_exc(e)
        return (-1, -1)
    
def lat_lon_from_decimal_degrees( lat_lon_string ):
    lat_lon_string = lat_lon_string.replace(u"°", "")
    try:
        lon, lat = lat_lon_string.split(",")
        return (degrees_to_float(lat), degrees_to_float(lon))
    except Exception, e:
        import traceback
        print traceback.format_exc(e)
        return (-1, -1)
    
def lat_lon_from_format_string( lat_lon_string ):
    if lat_lon_string.find(u"″") != -1:
        return lat_lon_from_degrees_minutes_seconds( lat_lon_string )
    elif lat_lon_string.find(u"′") != -1:
        return lat_lon_from_degrees_minutes( lat_lon_string )

    return lat_lon_from_decimal_degrees( lat_lon_string )
    
    

class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.coord = (-62.242, 12.775)
    
    def testToDecimalDegrees(self):
        result = u"12.775000° N, 62.242000° W"
        self.assertEquals(result, format_lat_lon_degrees(self.coord[0], self.coord[1]))
        
    def testToDegreesMinutesSeconds(self):
        result = u"12° 46′ 30″ N, 62° 14′ 31″ W"
        self.assertEquals(result, format_lat_lon_degrees_minutes_seconds(self.coord[0], self.coord[1]))
        
    def testToDegreesDecimalMinutes(self):
        result = u"12° 46.5000′ N, 62° 14.5200′ W"
        self.assertEquals(result, format_lat_lon_degrees_minutes(self.coord[0], self.coord[1]))
        
    def testFromFormatStringDecimalDegrees(self):
        lat_lon_string = u"12.775000° N, 62.242000° W"
        self.assertEquals(self.coord, lat_lon_from_format_string(lat_lon_string))
        
    def testFromFormatStringDegreesMinutesSeconds(self):
        lat_lon_string = u"12° 46′ 30″ N, 62° 14′ 31″ W"
        result = lat_lon_from_format_string(lat_lon_string)
        self.assertEquals(self.coord, (round(result[0], 3), round(result[1], 3)))
        equator_string = u"0° 0′ 0″, 62° 14′ 31″ W"
        result = lat_lon_from_format_string(equator_string)
        self.assertEquals((self.coord[0], 0.0), (round(result[0], 3), round(result[1], 3)))
        
    def testFromFormatStringDegreesMinutes(self):
        lat_lon_string = u"12° 46.5000′ N, 62° 14.5200′ W"
        result = lat_lon_from_format_string(lat_lon_string)
        self.assertEquals(self.coord, (round(result[0], 3), round(result[1], 3)))
        
    def testInvalidFormatStringToLatLon(self):
        self.assertEquals((-1, -1),  lat_lon_from_format_string("Hello!"))
        
def getTestSuite():
    return unittest.makeSuite(CoordinateTests)