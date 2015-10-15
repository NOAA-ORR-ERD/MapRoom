#!/usr/bin/env python
"""

"""

import os
from hookutils import collect_submodules, collect_data_files

subpkgs = [
    "traitsui.editors",
    "traitsui.extras",
    "traitsui.wx",
 
    "pyface",
    "pyface.*",
    "pyface.wx",
 
    "pyface.ui.wx",
    "pyface.ui.wx.init",
    "pyface.ui.wx.*",
    "pyface.ui.wx.grid.*",
    "pyface.ui.wx.action.*",
    "pyface.ui.wx.timer.*",
    "pyface.ui.wx.tasks.*",
    "pyface.ui.wx.workbench.*",
]
subpkgs = [
    "traits",
    "traitsui",
    "pyface",
    "omnimon",
    "maproom",
]

hiddenimports = []
for s in subpkgs:
    hiddenimports.extend(collect_submodules(s))
#print hiddenimports

subpkgs = [
    "traitsui",
    "pyface",
    "omnimon",
    "maproom",
]
datas = []
for s in subpkgs:
    datas.extend(collect_data_files(s))
#print datas
