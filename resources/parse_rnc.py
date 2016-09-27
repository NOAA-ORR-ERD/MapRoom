#!/usr/bin/env python

import os
import sys
import re
from xml.etree import ElementTree as ET

import numpy as np


import logging
log = logging.getLogger(__name__)


class RNCChart(object):
    def __init__(self, description, polygon):
        self.title = ""
        self.filename = ""
        self.points = []
        try:
            self.parse_desc(description)
            self.parse_polygon(polygon)
        except Exception, e:
            print "ERROR in ", description[0].text, polygon
            raise

    def parse_desc(self, desc):
        text = desc[0].text
        log.debug("DESCRIPTION:", text)
        args = text.split(";")
        print("ARGS", args)
        last = None
        for arg in args:
            if ":" in arg:
                k, v = arg.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "title":
                    self.title = v
                    last = "title"
                elif k == "filename":
                    self.filename = v
                    last = "filename"
                else:
                    last = None
            elif last:
                # sometimes there is an extra ";" in the string, so append it
                # to the last item if we find an arg that doesn't have a ":"
                # keyword separator
                if last == "title":
                    self.title +=  "; " + arg
                elif last == "filename":
                    self.filename += "; " + arg


    def parse_polygon(self, p):
        log.debug("POLYGON:", p)
        if p.tag == "{http://www.opengis.net/gml/3.2}LinearRing":
            for child in p:
                lat, lon = [float(x) for x in child.text.split()]
                lon = (lon % 360) - 360
                log.debug("point:", lon, lat)
                self.points.append((lon, lat))


class RNCParser(object):
    def __init__(self, filename):
        self.filename = filename
        self.fileroot, _ = os.path.splitext(os.path.basename(self.filename))
        self.maps = []
        self.parse()
    
    def parse(self):
        self.maps = []
        tree = ET.parse(self.filename)
        root = tree.getroot()
        root = ET.parse(self.filename)
        value = root.findall("{http://www.isotc211.org/2005/gmd}composedOf/{http://www.isotc211.org/2005/gmd}DS_DataSet/{http://www.isotc211.org/2005/gmd}has/{http://www.isotc211.org/2005/gmd}MD_Metadata")
        for v in value:
            e = v.findall(".//{http://www.isotc211.org/2005/gmd}EX_Extent")
            m = self.parse_extent(e)
            self.maps.extend(m)

    def parse_extent(self, extent):
        maps = []
        for e in extent:
            desc = e.findall(".//{http://www.isotc211.org/2005/gmd}description/{http://www.isotc211.org/2005/gco}CharacterString")
            log.debug("DESCRIPTION:", desc[0].text)
            polygons = e.findall(".//{http://www.isotc211.org/2005/gmd}EX_BoundingPolygon/{http://www.isotc211.org/2005/gmd}polygon/{http://www.opengis.net/gml/3.2}Polygon/{http://www.opengis.net/gml/3.2}exterior/")
            p = polygons[0]
            log.debug("POLYGON:", p)
            maps.append(RNCChart(desc, p))
        return maps

    def create_bna(self, filename):
        with open(filename, "w") as fh:
            for m in self.maps:
                fh.write('"%s","1",%d\n' % (m.title, len(m.points)))
                fh.write("%s\n" % "\n".join(["%f,%f" % pt for pt in m.points]))


if __name__ == "__main__":
    mm = RNCParser("small.xml")
    # print mm.maps
    mm = RNCParser("RNCProdCat_19115.xml")
    print mm.maps
    print mm.maps[2].points
    mm.create_bna("RNCProdCat_19115.bna")
