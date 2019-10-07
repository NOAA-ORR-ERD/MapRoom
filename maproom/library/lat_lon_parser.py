import re

import numpy as np

# removed local copy and now use parse function from Chris's pypi package
from lat_lon_parser import parse, to_deg_min, to_deg_min_sec

from sawx.utils.textutil import check_for_matching_lines, parse_for_matching_lines


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

