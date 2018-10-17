import xml.etree.ElementTree as ET


class Waypoint:
    def __init__(self):
        pass

    def __str__(self):
        return f"waypoint lat:{self.lat} lon:{self.lon}"

    @classmethod
    def fromxml(cls, r):
        w = cls()
        w.lat = r.attrib['lat']
        w.lon = r.attrib['lon']
        print(r)
        print(r.items())
        print(r.attrib)
        print(r.getchildren())
        print(r.keys())
        return w


class GPSDataset:
    """Subclass that renders SVG images without clearing the screen
    """
    XMLNS = "http://www.topografix.com/GPX/1/1"

    def __init__(self, xmltext):
        root_element = ET.fromstring(xmltext)
        self.metadata = None
        self.waypoints = []
        self.track = []
        for r in root_element.getchildren():
            if r.tag.endswith("metadata"):
                self.metadata = r
            elif r.tag.endswith("wpt"):
                w = Waypoint.fromxml(r)
                self.waypoints.append(w)
                print(w)
            elif r.tag.endswith("trk") and False:
                #self.name = r.findtext(f"{self.XMLNS}name")
                for t in r.getchildren():
                    if t.tag.endswith("name"):
                        self.name = t
                    elif t.tag.endswith("trkseg"):
                        for trk in t.getchildren():
                            print(trk)
                            print(dir(trk))
                            print(trk.items())
                            print(trk.attrib)
                            print(trk.getchildren())
                            print(trk.keys())

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


if __name__ == "__main__":
    t = open("../../TestData/GPS/amy.gpx").read()
    g = GPSDataset(t)
