# coding=utf8
import re
import math

import unittest
import pytest

import mock

from maproom.library.coordinates import *

class CoordinateTests(unittest.TestCase):

    def setUp(self):
        self.coord = (-62.242, 12.775)

    def testToDecimalDegrees(self):
        result = u"12.775000° N, 62.242000° W"
        self.assertEquals(result, format_lat_lon_degrees(self.coord[0], self.coord[1]))

    def testToDegreesMinutesSeconds(self):
        result = u"12°46′30″N, 62°14′31″W"
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
        string2 = format_lat_lon_degrees_minutes_seconds(result[0], result[1])
        result2 = lat_lon_from_format_string(string2)
        self.assertEquals((round(result[0], 3), round(result[1], 3)), (round(result2[0], 3), round(result2[1], 3)))
        equator_string = u"0° 0′ 0″, 62° 14′ 31″ W"
        result = lat_lon_from_format_string(equator_string)
        self.assertEquals((self.coord[0], 0.0), (round(result[0], 3), round(result[1], 3)))

    def testFromFormatStringDegreesMinutes(self):
        lat_lon_string = u"12° 46.5000′ N, 62° 14.5200′ W"
        result = lat_lon_from_format_string(lat_lon_string)
        self.assertEquals(self.coord, (round(result[0], 3), round(result[1], 3)))

    def testInvalidFormatStringToLatLon(self):
        with pytest.raises(ValueError):
            lat_lon_from_format_string("Hello!")


def getTestSuite():
    return unittest.makeSuite(CoordinateTests)
