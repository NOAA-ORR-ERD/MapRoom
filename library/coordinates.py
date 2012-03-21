# coding=utf8

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
