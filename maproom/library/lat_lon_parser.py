# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experimental code for parsing lat-long coordinates in "various" formats

formats supported:

Decimal degrees (easy):
   23.43
   -45.21

Decimal Degrees with quadrant:
   23.43 N
   45.21 W

Degrees, decimal minutes: (now it starts getting tricky!)
  23° 25.800'
  -45° 12.600'

  or

  23 25.800'
  -45 12.600'

  or

  23° 25.8' N
  45° 12.6' W

Degrees, Minutes, Seconds: (really fun!!!)

   23° 25' 48.0"
  -45° 12' 36.0"

  or

   23d 25' 48.0"
  -45d 12' 36.0"

  or

   23° 25' 48.0" N
  45° 12' 36.0" S

  or -- lots of other combinations!

"""


import numpy as np

from .unit_conversion import LatLongConverter  # from: https://github.com/NOAA-ORR-ERD/PyNUCOS

from sawx.utils.textutil import check_for_matching_lines, parse_for_matching_lines


# new test version -- much simpler
import re

def parse(string):
    """
    Attempts to parse a latitude or longitude string

    Returns the value in floating point degrees

    If parsing fails, it raises a ValueError

    NOTE: This is a naive brute-force approach.
          I imagine someone that can make regular expressions dance could do better..
    """

    orig_string = string

    string = string.strip()
    # change W and S to a negative value
    if string.endswith('W') or string.endswith('w'):
        string = '-' + string[:-1]
    elif string.endswith('S') or string.endswith('s'):
        string = '-' + string[:-1]

    # get rid of everything that is not numbers
    string = re.sub(r"[^0-9.-]", " ", string).strip()

    try:
        parts = [float(part) for part in string.split()]
        if parts:
            return LatLongConverter.ToDecDeg(*parts)
        else:
            raise ValueError()
    except ValueError:
        raise ValueError("%s is not a valid coordinate string" % orig_string)


re_latlon = rb'^\s*([-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?))\s*[/,|\s]+\s*([-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?))'

re_lonlat = rb'^\s*([-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?))\s*[/|,\s]+\s*([-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?))'

def parse_coordinate_text(text):
    mime = None
    if type(text) == str:
        text = text.encode('utf-8')
    latlon, num_latlon_unmatched = parse_for_matching_lines(text, re_latlon, [1, 2])
    print("Trying lat/lon order: %d, unmatched=%d" % (len(latlon), num_latlon_unmatched))
    lonlat, num_lonlat_unmatched = parse_for_matching_lines(text, re_lonlat, [1, 2])
    print("Trying lon/lat order: %d, unmatched=%d" % (len(latlon), num_lonlat_unmatched))
    if len(latlon) > num_latlon_unmatched or len(lonlat) > num_lonlat_unmatched:
        latlon = np.asarray(latlon, dtype=np.float64)
        lonlat = np.asarray(lonlat, dtype=np.float64)

        latfirst = num_latlon_unmatched / (len(latlon) + num_latlon_unmatched)
        lonfirst = num_lonlat_unmatched / (len(lonlat) + num_lonlat_unmatched)
        if latfirst == lonfirst:
            # sum columns. If negative, that column is likely to be longitude
            # because of the hemisphere we're operating in.
            cols = np.sum(latlon, axis=0)
            if cols[0] < 0.0:
                mime = "text/lonlat"
            elif cols[1] < 0.0:
                mime = "text/latlon"
            else:
                # check if one of the columns is > 90 degrees. If so, that's
                # the longitude
                cols = np.amax(latlon, axis=0)
                if cols[0] > 90.0:
                    mime = "text/lonlat"
                elif cols[1] > 90.0:
                    mime = "text/latlon"
                else:
                    # guess one...
                    mime = "text/latlon"

        elif latfirst > lonfirst:
            mime = "text/latlon"
        else:
            mime = "text/lonlat"
    print("Guessing mime: %s" % mime)

    if mime == "text/latlon":
        return mime, latlon, num_latlon_unmatched
    elif mime is not None:
        return "text/lonlat", lonlat, num_lonlat_unmatched
    return None, [], -1


if __name__ == "__main__":
    # print out all the forms that work
    print("All these forms work:")
    for string, val in test_values:
        print(string)
    print("And these don't:")
    for string in invalid_values:
        print(string)

