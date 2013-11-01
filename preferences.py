import json
import os
import sys


class MaproomPreferences(object):

    """
    This class behaves like a standard dict object, except it has built-in json load and
    save capabilities, and will return default values if they exist when the user has 
    not set a preference.
    """

    def __init__(self, filename):
        self.filename = filename

        self.prefs = {}

        self.load(filename)

        self.pref_defaults = {
            "Coordinate Display Format": "degrees decimal minutes",
            "Scroll Zoom Speed": "Slow",
            
            # FIXME: preferences doesn't handle non-string items when saving/loading
            "Number of Recent Files": 20,
        }

    def load(self, filename):
        if os.path.exists(filename):
            f = open(filename, 'rb')
            self.prefs = json.load(f)
            f.close()

    def save(self, filename=None):
        if filename is None:
            filename = self.filename

        f = open(filename, "wb")
        json.dump(self.prefs, f)
        f.close()

    def __len__(self):
        return len(self.prefs)

    def __contains__(self, item):
        return item in self.prefs

    def __getitem__(self, key):
        if not key in self.prefs:
            if key in self.pref_defaults:
                return self.pref_defaults[key]
            else:
                return None

        return self.prefs[key]

    def __setitem__(self, key, value):
        self.prefs[key] = value
        self.save()  # immediately save any preference changes

    def __delitem__(self, key):
        del self.prefs[key]
