# coding=utf8
import re
import math

import unittest
import pytest

from mock import *

from maproom.library.coordinates import *

class CoordinateTests(unittest.TestCase):

    def setUp(self):
        self.coord = (-62.242, 12.775)

    def testToDecimalDegrees(self):
        result = "12.775000° N, 62.242000° W"
        self.assertEqual(result, format_lat_lon_degrees(self.coord[0], self.coord[1]))

    def testToDegreesMinutesSeconds(self):
        result = "12°46′30″N, 62°14′31″W"
        self.assertEqual(result, format_lat_lon_degrees_minutes_seconds(self.coord[0], self.coord[1]))

    def testToDegreesDecimalMinutes(self):
        result = "12° 46.5000′ N, 62° 14.5200′ W"
        self.assertEqual(result, format_lat_lon_degrees_minutes(self.coord[0], self.coord[1]))

    def testFromFormatStringDecimalDegrees(self):
        lat_lon_string = "12.775000° N, 62.242000° W"
        self.assertEqual(self.coord, lat_lon_from_format_string(lat_lon_string))

    def testFromFormatStringDegreesMinutesSeconds(self):
        lat_lon_string = "12° 46′ 30″ N, 62° 14′ 31″ W"
        result = lat_lon_from_format_string(lat_lon_string)
        self.assertEqual(self.coord, (round(result[0], 3), round(result[1], 3)))
        string2 = format_lat_lon_degrees_minutes_seconds(result[0], result[1])
        result2 = lat_lon_from_format_string(string2)
        self.assertEqual((round(result[0], 3), round(result[1], 3)), (round(result2[0], 3), round(result2[1], 3)))
        equator_string = "0° 0′ 0″, 62° 14′ 31″ W"
        result = lat_lon_from_format_string(equator_string)
        self.assertEqual((self.coord[0], 0.0), (round(result[0], 3), round(result[1], 3)))

    def testFromFormatStringDegreesMinutes(self):
        lat_lon_string = "12° 46.5000′ N, 62° 14.5200′ W"
        result = lat_lon_from_format_string(lat_lon_string)
        self.assertEqual(self.coord, (round(result[0], 3), round(result[1], 3)))

    def testInvalidFormatStringToLatLon(self):
        with pytest.raises(ValueError):
            lat_lon_from_format_string("Hello!")

class TestUnits(object):
    def test_feet(self):
        cases = [
            (.5, "0 ft", "2640.0 ft", "500 m", "500.0 m"),
            (100, "100 ft", "100.0 mi", "100 km", "100.0 km"),
            (4000.25, "4000 ft", "4000.2 mi", "4000 km", "4000.2 km"),
            (5000, "1 mi", "5000.0 mi", "5000 km", "5000.0 km"),
            (5280, "1 mi", "5280.0 mi", "5280 km", "5280.0 km"),
            (6000, "1 mi", "6000.0 mi", "6000 km", "6000.0 km"),
            (16000, "3 mi", "16000.0 mi", "16000 km", "16000.0 km"),
            ]
        for dist, imperial, imperial_rounded, metric, metric_rounded in cases:
            assert ft_to_string(dist) == imperial
            assert mi_to_rounded_string(dist) == imperial_rounded
            assert km_to_string(dist) == metric
            assert km_to_rounded_string(dist) == metric_rounded
        assert mi_to_rounded_string(100, area=True) == "100.0 mi\u00b2"
        assert mi_to_rounded_string(.1, area=True) == "528.0 ft\u00b2"
        assert km_to_rounded_string(.2, area=True) == "200.0 m\u00b2"
        assert km_to_rounded_string(2, area=True) == "2.0 km\u00b2"

class TestHaversine(object):
    def test_haversine(self):
        assert_almost_equal(haversine(20, 30, 25, 35), 727.0688, decimal=4)
        assert_almost_equal(haversine((20, 30), (25, 35)), 727.0688, decimal=4)
        assert_almost_equal(distance_bearing(20, 30, 75, 20), (20.2007, 30.0464), decimal=4)
        assert_almost_equal(haversine_at_const_lat(5, 30), 481.4499, decimal=4)
        assert_almost_equal(haversine(20, 30, 25, 30), 481.4499, decimal=4)
        
    