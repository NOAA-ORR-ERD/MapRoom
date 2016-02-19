import os
import json

import numpy as np

from mock import *

from maproom.library.jsonutil import collapse_json


class TestBasic(object):
    def setup(self):
        self.items = [
            ([50, 90, 80, 182], 4, """[
    50,
    90,
    80,
    182
]"""),
            (["a", {"1":2, "3":4}], 4, """[
    "a",
    {"1": 2, "3": 4}
]"""),
            (["first", {"second": 2, "third": 3, "fourth": 4, "items": [[1,2,3,4], [5,6,7,8]]}], 4, """[
    "first",
    {"items": [[1, 2, 3, 4], [5, 6, 7, 8]], "second": 2, "fourth": 4, "third": 3}
]"""),
            ([0, 1, [2, [1,2,3,4], [5,6,7,8]], 9, 10, [11, [12, [13, [14, 15]]]]], 4, """[
    0,
    1,
    [2, [1, 2, 3, 4], [5, 6, 7, 8]],
    9,
    10,
    [11, [12, [13, [14, 15]]]]
]"""),
            ({"zero": ["first", {"second": 2, "third": 3, "fourth": 4, "items": [[1,2,3,4], [5,6,7,8], 9, 10, [11, [12, [13, [14, 15]]]]]}]}, 12, """{
    "zero": [
        "first", 
        {
            "items": [[1, 2, 3, 4], [5, 6, 7, 8], 9, 10, [11, [12, [13, [14, 15]]]]],
            "second": 2,
            "fourth": 4,
            "third": 3
        }
    ]
}"""),
            ]

    def test_simple(self):
        for before, level, after in self.items:
            text = json.dumps(before, indent=4)
            processed = collapse_json(text, indent=level)
            assert processed == after
            rebuilt = json.loads(processed)
            assert rebuilt == before

if __name__ == "__main__":
    t = TestBasic()
    t.setup()
    t.test_simple()
