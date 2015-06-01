import os

import unittest
from nose.tools import *

import numpy as np

from mock import *

from maproom.command import *
from maproom.mouse_commands import *
from maproom.menu_commands import *
from test_command_log import TestLogBase

class TestBasic(TestLogBase):
    logfile = "../TestData/CommandLog/verdat1.mrc"

    def test_save(self):
        lm = self.manager
        lm.save_all("test.mrp")


if __name__ == "__main__":
    #unittest.main()
    import time
    
    t = TestBasic()
    t.setup()
    t.test_save()
