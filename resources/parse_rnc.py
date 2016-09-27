#!/usr/bin/env python

import os
import sys
import re
from xml.etree import ElementTree as ET

import numpy as np

class RNCParser(object):
    def __init__(self, filename):
        self.filename = filename
        self.fileroot, _ = os.path.splitext(os.path.basename(self.filename))
        self.parse()
    
    def parse(self):
        maps = []
        print self.filename
        tree = ET.parse(self.filename)
        root = tree.getroot()
        print root
        root = ET.parse(self.filename)
        value = root.findall("{http://www.isotc211.org/2005/gmd}composedOf/{http://www.isotc211.org/2005/gmd}DS_DataSet/{http://www.isotc211.org/2005/gmd}has/{http://www.isotc211.org/2005/gmd}MD_Metadata")
        print value
        print len(value)
        for v in value:
            print v
            e = v.findall(".//{http://www.isotc211.org/2005/gmd}EX_Extent")
            print e

        value = root.findall(".")
        print value

        value = root.findall("{http://www.isotc211.org/2005/gmd}composedOf")
        print value

        value = root.findall(".//{http://www.isotc211.org/2005/gmd}EX_BoundingPolygon")
        print value

if __name__ == "__main__":
#    mm = RNCParser("RNCProdCat_19115.xml")
    mm = RNCParser("small.xml")
