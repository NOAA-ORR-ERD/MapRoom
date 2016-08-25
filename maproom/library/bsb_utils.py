#!/usr/bin/env python

import os
import sys
import re

from fs.opener import fsopen

class KAPImage(object):
    tag_map = {
        "FN": "filename",
        "TY": "type",
        "NU": "pane",
        "NA": "name",
    }

    def __init__(self, dirname, tags):
        self.tags = {}
        self.parse_tags(dirname, tags)

    def __str__(self):
        return "KAP: %s: pane=%s name=%s" % (self.filename, self.pane, self.name)

    def parse_tags(self, dirname, tags):
        for item in tags.split(","):
            tag, value = item.split("=", 1)
            self.tags[tag] = value
            if tag in self.tag_map:
                if tag == "FN":
                    value = os.path.join(dirname, value)
                setattr(self, self.tag_map[tag], value)

class BSBInfo(object):
    def __init__(self, dirname, items):
        self.items = items
        self.images = []
        self.parse_images(dirname)
    
    def __str__(self):
        items = []
        for k in sorted(self.items.keys()):
            items.append("%s /// %s" % (k, self.items[k]))
        for image in self.images:
            items.append(str(image))
        return "\n".join(items)

    def parse_images(self, dirname):
        index = 1
        images = []
        while True:
            k = "K%02d" % index
            if k in self.items:
                image = KAPImage(dirname, self.items[k])
                images.append(image)
            else:
                break
            index += 1

        images.sort(key=lambda a:a.pane)
        self.images = images

class BSBParser(object):
    delimiter_separators = {
        "CRR":" ",
    }

    def __init__(self, filename):
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.fileroot, _ = os.path.splitext(os.path.basename(self.filename))
        self.info = None
        self.parse()
    
    def parse(self):
        items = {}
        tag = None
        current = ""
        with fsopen(self.filename) as fh:
            for text in fh:
                text = text.rstrip()
                if text.startswith(" "):
                    sep = self.delimiter_separators.get(tag, ",")
                    current += sep + text.lstrip()
                    continue
                if tag:
                    items[tag] = current
                if not text.startswith("!"):
                    tag, current = text.split("/", 1)
                else:
                    tag = None
                    current = ""
        if current:
            items[tag] = current

        self.get_info(items)

    def get_info(self, items):
        # this is where we could check the version, but only have 3.0 samples
        self.info = BSBInfo(self.dirname, items)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = BSBParser(arg)
        print p.info
