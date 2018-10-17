from datetime import datetime
from dateutil import parser as date_parser

import xml.etree.ElementTree as ET


class Waypoint:
    def __init__(self):
        self.lat = 0.
        self.lon = 0.
        self.ele = 0.
        self.name = ""
        self.desc = ""
        self.sym = ""
        self.type = ""
        self.time = datetime.now()

    def __str__(self):
        return f"waypoint lat:{self.lat} lon:{self.lon} ele:{self.ele} name:{self.name} time:{self.time} desc:{self.desc}"

    @classmethod
    def fromxml(cls, r):
        w = cls()
        w.lat = float(r.attrib['lat'])
        w.lon = float(r.attrib['lon'])
        for t in r.getchildren():
            if t.tag.endswith("name"):
                w.name = t.text
            elif t.tag.endswith("ele"):
                w.ele = float(t.text)
            elif t.tag.endswith("desc"):
                w.desc = t.text
            elif t.tag.endswith("cmt"):
                w.cmt = t.text
            elif t.tag.endswith("sym"):
                w.sym = t.text
            elif t.tag.endswith("type"):
                w.type = t.text
            elif t.tag.endswith("time"):
                w.time = date_parser.parse(t.text)
        return w


class Trackpoint:
    def __init__(self):
        self.lat = 0.
        self.lon = 0.
        self.ele = 0.
        self.time = datetime.now()

    def __str__(self):
        return f"trackpoint lat:{self.lat} lon:{self.lon} ele:{self.ele} time:{self.time}"

    @classmethod
    def fromxml(cls, r):
        w = cls()
        w.lat = float(r.attrib['lat'])
        w.lon = float(r.attrib['lon'])
        for t in r.getchildren():
            if t.tag.endswith("ele"):
                w.ele = float(t.text)
            elif t.tag.endswith("time"):
                w.time = date_parser.parse(t.text)
        return w


class GPSDataset:
    """Superclass to parse and hold GPS trackpoint and waypoint data
    """
    def __init__(self, input_data):
        self.metadata = None
        self.waypoints = []
        self.track = []
        self.parse(input_data)

    def parse(self, input_data):
        pass


class GarminGPSDataset(GPSDataset):
    """Class to parse GPSDataset from Garmin XML data
    """
    XMLNS = "http://www.topografix.com/GPX/1/1"

    def parse(self, xmltext):
        root_element = ET.fromstring(xmltext)
        for r in root_element.getchildren():
            if r.tag.endswith("metadata"):
                self.metadata = r
            elif r.tag.endswith("wpt"):
                w = Waypoint.fromxml(r)
                self.waypoints.append(w)
                print(w)
            elif r.tag.endswith("trk"):
                #self.name = r.findtext(f"{self.XMLNS}name")
                for t in r.getchildren():
                    if t.tag.endswith("name"):
                        self.name = t.text
                    elif t.tag.endswith("trkseg"):
                        for trk in t.getchildren():
                            w = Trackpoint.fromxml(trk)
                            self.track.append(w)
                            print(w)

                # print(r)
                # print(r.getchildren())
                # print(r.tag)
                # print(r.tail)
                # print()

        # for node in root_element.findall('trk'):
        #     print("node", node)
        #     url = node.attrib.get('xmlUrl')
        #     if url:
        #         print(url)
        print(f"{len(self.waypoints)} waypoints")
        print(f"{len(self.track)} trackpoints")


if __name__ == "__main__":
    t = open("../../TestData/GPS/sand_point1.gpx").read()
    g = GarminGPSDataset(t)
