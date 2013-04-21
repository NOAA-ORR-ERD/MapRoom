import re
import os
import csv
import time
import pyproj
import datetime
import numpy as np
from maproomlib.plugin.Point_set_layer import Point_set_layer
from maproomlib.plugin.Line_set_layer import Line_set_layer
from maproomlib.plugin.Line_point_layer import Line_point_layer
from maproomlib.plugin.Verdat_saver import Verdat_saver
from maproomlib.plugin.Label_set_layer import Label_set_layer


TIME_FORMAT = "%m/%d/%Y %H:%M:%S"
MAXIMUM_DELTA = 0.25
WHITESPACE_PATTERN = re.compile( "\s+" )
AVERAGE_SPEED_WINDOW = datetime.timedelta( hours = 25 )
HORIZON_MARINE_DATETIME_FORMAT = "%Y/%m/%d %H:%M"
JOUBEH_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
JOUBEH_CUTOFF_TIME = datetime.datetime.now() - datetime.timedelta( days = 10 )
USF_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
USF_CUTOFF_TIME = datetime.datetime.now() - datetime.timedelta( days = 14 )
USCG_DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"
NAVY_DATE_FORMAT = "%Y%m%d"
BOGUS_SPEED_THRESHOLD = 100


def Buoy_loader( filename, command_stack, plugin_loader, parent ):
    """
    A plugin for loading buoy data from a CSV file.
    """

    in_file = file( filename, "rU" )

    if "," in in_file.read( 512 ):
        in_file.seek( 0 )
        reader = csv.reader( in_file, delimiter = "," )
    else:
        in_file.seek( 0 )
        reader = csv.reader( in_file, delimiter = " ", skipinitialspace = True )

    lower_left = ( None, None )
    upper_right = ( None, None )

    point_index = 0
    point_data = []
    line_data = []
    labels = []
    last_buoy_number = None
    color = Line_point_layer.DEFAULT_COLORS[
        Line_point_layer.next_default_color_index
    ]
    buoy_number = None
    buoy_number_header = None
    previous_timestamp = None
    start_buoy_index = 0
    now_time = datetime.datetime.now().utcnow()
    this_time = None
    reverse_chronological = False
    dark_color_steps = None
    buoy_point_starts = set()
    buoy_line_starts = set()

    drogue_depths = {} # buoy number (int) -> drogue depth (int)
    speeds = {}        # buoy number (int) -> speeds (list of floats)

    for data in reader:
        if not data: continue # Skip blank lines.

        # Skip comments, but gather any drogue depth from them first.
        if data[ 0 ].startswith( "# " ):
            pieces = WHITESPACE_PATTERN.split( data[ 0 ][ 2: ].strip() )
            if len( pieces ) != 2: continue
            try:
                drogue_depth = float( pieces[ 1 ] )
            except ValueError:
                continue

            # Support a buoy number range.
            if "-" in pieces[ 0 ]:
                ( start_buoy_number, end_buoy_number ) = \
                    pieces[ 0 ].split( "-" )

                try:
                    start_buoy_number = int( start_buoy_number )
                    end_buoy_number = int( end_buoy_number )
                except ValueError:
                    continue

                for buoy_number in \
                    range( start_buoy_number, end_buoy_number + 1 ):
                    drogue_depths[ buoy_number ] = drogue_depth
            else:
                try:
                    buoy_number = int( pieces[ 0 ] )
                except ValueError:
                    continue
                drogue_depths[ buoy_number ] = drogue_depth

            continue

        if buoy_number_header is None:
            buoy_number_header = data[ 0 ]
            if buoy_number_header not in \
               ( "BuoyNum", "FHD", "#FHD_ID", "Received Date(GMT)", "UNCLASSIFIED", "Date" ) \
               and not filename.endswith( ".dat" ):
                raise RuntimeError(
                    "The buoy file %s is invalid." % filename,
                )
            continue

        # Horizon Marine format
        if "FHD" in buoy_number_header:
            ( buoy_number, datestamp, timestamp, latitude, longitude ) = data[ 0: 5 ]
            try:
                this_time = datetime.datetime.strptime(
                    "%s %s" % ( datestamp.strip(), timestamp.strip() ),
                    HORIZON_MARINE_DATETIME_FORMAT,
                )
            except ValueError:
                this_time = None

            buoy_number = int( buoy_number )
            drogue_depth = drogue_depths.get( buoy_number )

            dark_color_steps = 36
            reverse_chronological = True

            #if drogue_depth != 50:
            #    continue

            if this_time and now_time - this_time <= AVERAGE_SPEED_WINDOW and \
               len( data ) >= 8 and float( data[ 7 ] ) < BOGUS_SPEED_THRESHOLD:
                speeds.setdefault( buoy_number, [] ).append( float( data[ 7 ] ) )

            if buoy_number:
                timestamp = "%s: %s" % ( buoy_number, datestamp )
            else:
                timestamp = datestamp

            timestamp = timestamp.replace( "2010/", "" )

        # JouBeh/Alaska Clean Seas format
        elif "Received Date(GMT)" in buoy_number_header:
            try:
                this_time = datetime.datetime.strptime(
                    data[ 2 ].strip(),
                    JOUBEH_DATETIME_FORMAT,
                )
            except ValueError:
                this_time = None

            if this_time and this_time < JOUBEH_CUTOFF_TIME:
                continue 

            if not buoy_number:
                pieces = os.path.basename( filename ).split( "_" )
                if pieces[ 0 ] == "joubeh":
                    buoy_number = int( pieces[ 1 ].split( "." )[ 0 ] )
                else:
                    buoy_number = int( pieces[ 0 ] )

            dark_color_steps = 144
            reverse_chronological = True

            latitude = data[ 3 ]
            longitude = data[ 4 ]

            if this_time:
                timestamp = "%s: %s/%s" % ( buoy_number, this_time.month, this_time.day )
            else:
                timestamp = "%s" % buoy_number

        # AOML format
        elif filename.endswith( ".dat" ):
            ( buoy_number, _, month, day, year, latitude, longitude, speed ) = data[ 0: 8 ]

            buoy_number = int( buoy_number )

            dark_color_steps = 21
            reverse_chronological = False

            timestamp = "%s: %02d/%02d" % ( buoy_number, int( month ), int( float( day ) ) )

        # Navy format
        elif "UNCLASSIFIED" in buoy_number_header:
            if data[ 0 ] == "DISTRIBUTION":
                continue

            ( buoy_number, datestamp, timestamp, latitude, longitude ) = data[ 0: 5 ]

            this_date = datetime.datetime.strptime(
                datestamp.strip(),
                NAVY_DATE_FORMAT,
            )

            dark_color_steps = 96
            reverse_chronological = False

            timestamp = "%s: %02d/%02d" % ( buoy_number, int( this_date.month ), int( this_date.day ) )

        # new USF / Coast Guard format
        elif "Date" in buoy_number_header:
            ( datestamp, timestamp, latitude, longitude, speed, speed2 ) = data[ 0: 6 ]
            try:
                this_time = datetime.datetime.strptime(
                    "%s %s" % ( datestamp.strip(), timestamp.strip() ),
                    USF_DATETIME_FORMAT,
                )
            except ValueError:
                this_time = None

            # This format includes all historical data, which is way too much
            # to plot if we want the drifter tracks to remain readable. So
            # skip points older than 14 days.
            if this_time and this_time < USF_CUTOFF_TIME:
                continue 

            buoy_number = int(
                os.path.basename( filename ).split( "." )[ 0 ].split( "_" )[ 0 ],
            )

            speed = float( speed )
            speed2 = float( speed2 )

            # If the speed is too large, then this isn't actually the speed
            # column.
            if speed > 10.0:
                speed = speed2

            if this_time and now_time - this_time <= AVERAGE_SPEED_WINDOW and \
               speed < BOGUS_SPEED_THRESHOLD:
                speeds.setdefault( buoy_number, [] ).append( speed )

            dark_color_steps = 144
            reverse_chronological = False

            if this_time:
                timestamp = "%s: %s/%s" % ( buoy_number, this_time.month, this_time.day )
            else:
                timestamp = "%s" % buoy_number

        # old Coast Guard format (via email)
        else:
            ( buoy_number, timestamp, latitude, longitude ) = data[ 0: 4 ]
            try:
                this_time = datetime.datetime.strptime(
                    timestamp.strip(),
                    USCG_DATETIME_FORMAT,
                )
            except ValueError:
                this_time = None

            timestamp = timestamp.split( " " )[ 0 ]
            buoy_number = int( buoy_number )
            dark_color_steps = 144
            reverse_chronological = True

            if this_time and now_time - this_time <= AVERAGE_SPEED_WINDOW and \
               len( data ) >= 7 and float( data[ 6 ] ) < BOGUS_SPEED_THRESHOLD:
                speeds.setdefault( buoy_number, [] ).append( float( data[ 6 ] ) )

            if this_time:
                timestamp = "%s: %s/%s" % \
                    ( buoy_number, this_time.month, this_time.day )

        latitude = float( latitude )
        longitude = float( longitude )

        if abs( latitude ) == 90 or abs( longitude ) == 180:
            continue

        # If there's a buoy transition, then the last line segment added was
        # not needed. Also, each bouy should be a different color.
        if last_buoy_number is not None and buoy_number != last_buoy_number:
            buoy_point_starts.add( point_index )
            buoy_line_starts.add( len( line_data ) )
            line_data.pop()
            color = Line_point_layer.DEFAULT_COLORS[
                Line_point_layer.next_default_color_index
            ]
            Line_point_layer.next_default_color_index = (
                Line_point_layer.next_default_color_index + 1
            ) % len( Line_point_layer.DEFAULT_COLORS )

            if speeds.get( last_buoy_number ):
                last_speeds = speeds[ last_buoy_number ]
                average_speed = sum( last_speeds, 0 ) / len( last_speeds )
                previous_timestamp += ", %.2f M/S" % average_speed

            last_buoy_number = buoy_number

            labels.pop()
            labels.append( previous_timestamp )

            labels.append( timestamp )
            start_buoy_index = point_index
        elif last_buoy_number is None:
            buoy_point_starts.add( point_index )
            buoy_line_starts.add( len( line_data ) )
            last_buoy_number = buoy_number
            labels.append( timestamp )
        else:
            # If the point has moved too far from the previous point, consider it
            # a data error and skip the point.
            if abs( point_data[ -1 ][ 0 ] - longitude ) > MAXIMUM_DELTA or \
               abs( point_data[ -1 ][ 1 ] - latitude ) > MAXIMUM_DELTA:
                continue

            labels.append( None )

        lower_left = (
            longitude if lower_left[ 0 ] is None \
                      else min( longitude, lower_left[ 0 ] ),
            latitude if lower_left[ 1 ] is None \
                     else min( latitude, lower_left[ 1 ] ),
        )
        upper_right = (
            max( longitude, upper_right[ 0 ] ),
            max( latitude, upper_right[ 1 ] ),
        )

        point_data.append( ( longitude, latitude, np.nan, color ) )
        line_data.append( ( point_index, point_index + 1, 0, color ) )

        previous_timestamp = timestamp
        point_index += 1

    # Make another pass through the point data, lightening the alpha component
    # of the color, thereby causing older points to be more transparent than
    # newer points.
    if dark_color_steps is not None:
        # Fade point colors for each buoy.
        if reverse_chronological:
            point_indices = xrange( 0, len( point_data ) )
        else:
            point_indices = xrange( len( point_data ) - 1, -1, -1 )
        MINIMUM_ALPHA = 48
        alpha = 255
        buoy_steps = 0

        for index in point_indices:
            ( longitude, latitude, depth, point_color ) = point_data[ index ]

            if index in buoy_point_starts:
                alpha = 255
                buoy_steps = 0

            color_array = np.array( [ point_color ], np.uint32 ).view( np.uint8 )
            if buoy_steps > dark_color_steps:
                if alpha > MINIMUM_ALPHA:
                    alpha -= 2
                color_array[ -1 ] = alpha
                point_color = color_array.view( np.uint32 )[ 0 ]

                point_data[ index ] = \
                    ( longitude, latitude, depth, point_color )

            buoy_steps += 1

        # Fade line colors for each buoy.
        if reverse_chronological:
            line_indices = xrange( 0, len( line_data ) )
        else:
            line_indices = xrange( len( line_data ) - 1, -1, -1 )
        alpha = 255
        buoy_steps = 0

        for index in line_indices:
            ( point_index_0, point_index_1, line_type, line_color ) = line_data[ index ]

            if index in buoy_line_starts:
                alpha = 255
                buoy_steps = 0

            color_array = np.array( [ line_color ], np.uint32 ).view( np.uint8 )
            if buoy_steps > dark_color_steps:
                if alpha > MINIMUM_ALPHA:
                    alpha -= 2
                color_array[ -1 ] = alpha
                line_color = color_array.view( np.uint32 )[ 0 ]

                line_data[ index ] = \
                    ( point_index_0, point_index_1, line_type, line_color )

            buoy_steps += 1

    if speeds.get( last_buoy_number ):
        last_speeds = speeds[ last_buoy_number ]
        average_speed = sum( last_speeds, 0 ) / len( last_speeds )
        previous_timestamp += ", %.2f M/S" % average_speed

    if labels:
        labels.pop()
    if previous_timestamp:
        labels.append( previous_timestamp )

    origin = lower_left
    if None in upper_right or None in lower_left:
        size = None
    else:
        size = (
            upper_right[ 0 ] - lower_left[ 0 ],
            upper_right[ 1 ] - lower_left[ 1 ],
        )

    in_file.close()

    point_count = len( point_data )
    points = Point_set_layer.make_points( point_count )
    points[ 0: point_count ] = point_data

    line_count = len( line_data )
    lines = Line_set_layer.make_lines( line_count )
    lines[ 0: line_count ] = line_data

    projection = pyproj.Proj( "+proj=latlong" )

    points_layer = Point_set_layer(
        command_stack,
        "Buoy data points", points, point_count,
        Point_set_layer.DEFAULT_POINT_SIZE, projection,
        origin = origin, size = size,
    )

    lines_layer = Line_set_layer(
        command_stack,
        "Buoy path", points_layer, lines, line_count,
        Line_set_layer.DEFAULT_LINE_WIDTH,
    )

    points_layer.lines_layer = lines_layer

    labels_layer = Label_set_layer(
        "Labels", points_layer, command_stack, labels,
    )

    line_point_layer = Line_point_layer(
        filename, command_stack, plugin_loader, parent,
        points_layer, lines_layer, None, origin, size,
        saver = Verdat_saver,
        labels_layer = labels_layer,
        hide_labels = False,
    )

    if point_count > 1:
        line_point_layer.hidden_children.add( points_layer )

    return line_point_layer


Buoy_loader.PLUGIN_TYPE = "file_loader"
