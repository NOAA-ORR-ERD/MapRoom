# coding=utf8


def float_to_degrees_minutes_seconds( value, signs = None ):
    sign = ""
    if signs:
        if value > 0:
            sign = signs[ 0 ]
        elif value < 0:
            sign = signs[ 1 ]

    value = abs( value )
    degrees = int( value )
    value = ( value - degrees ) * 60
    minutes = int( value )
    seconds = int( ( value - minutes ) * 60 )

    if degrees == 0 and minutes == 0 and seconds == 0:
        sign = ""

    return ( degrees, minutes, seconds, sign )


def float_to_degrees_minutes( value, signs = None ):
    sign = ""
    if signs:
        if value > 0:
            sign = signs[ 0 ]
        elif value < 0:
            sign = signs[ 1 ]

    value = abs( value )
    degrees = int( value )
    minutes = ( value - degrees ) * 60

    return ( degrees, minutes, sign )


def float_to_degrees( value, signs = None ):
    sign = ""
    if signs:
        if value > 0:
            sign = signs[ 0 ]
        elif value < 0:
            sign = signs[ 1 ]

    degrees = abs( value )

    return ( degrees, sign )


def format_lat_long_degrees_minutes_seconds( longitude, latitude ):
    long = u"%d° %d′ %d″ %s" % \
        float_to_degrees_minutes_seconds( longitude, signs = ( "E", "W" ) )
    lat = u"%d° %d′ %d″ %s" % \
        float_to_degrees_minutes_seconds( latitude, signs = ( "N", "S" ) )

    return u"%s, %s" % ( lat, long )

    
def format_lat_long_degrees_minutes( longitude, latitude ):
    long = u"%d° %0.4f′ %s" % \
        float_to_degrees_minutes( longitude, signs = ( "E", "W" ) )
    lat = u"%d° %0.4f′ %s" % \
        float_to_degrees_minutes( latitude, signs = ( "N", "S" ) )

    return u"%s, %s" % ( lat, long )


def format_lat_long_degrees( longitude, latitude ):
    long = u"%0.6f° %s" % \
        float_to_degrees( longitude, signs = ( "E", "W" ) )
    lat = u"%0.6f° %s" % \
        float_to_degrees( latitude, signs = ( "N", "S" ) )

    return u"%s, %s" % ( lat, long )


def format_lat_line_label( latitude ):
    ( degrees, minutes, sign ) = \
        float_to_degrees_minutes( latitude, signs = ( "N", "S" ) )

    minutes = round( minutes )
    if minutes == 60:
        minutes = 0
        degrees += 1

    return u" %d° %d' %s " % ( degrees, minutes, sign )


def format_long_line_label( longitude ):
    ( degrees, minutes, sign ) = \
        float_to_degrees_minutes( longitude, signs = ( "E", "W" ) )

    minutes = round( minutes )
    if minutes == 60:
        minutes = 0
        degrees += 1

    return u" %d° %d' %s " % ( degrees, minutes, sign )
