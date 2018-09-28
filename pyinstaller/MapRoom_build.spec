# -*- mode: python -*-

block_cipher = None
DEBUG = False

# to pass -v to the python interpreter, uncomment this:
VERBOSE_INTERPRETER = False

if VERBOSE_INTERPRETER:
    options = [ ('v', None, 'OPTION'),]
else:
    options = []

appname = "MapRoom_build"

with open("../maproom.py", "r") as fh:
    script = fh.read()
with open("%s.py" % appname, "w") as fh:
    fh.write(script)

import sys
sys.modules['FixTk'] = None

import os

pathex = [os.path.abspath("..")]

a = Analysis(["%s.py" % appname],
             pathex=pathex,
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'Cython', 'sphinx', 'nose', 'pygments', 'pytest'],
             cipher=block_cipher)

for pymod, path, tag in sorted(a.pure):
    if ".qt" in pymod or ".test" in pymod:
        print("why is this still here?", pymod)

# pytz zip bundle from https://github.com/pyinstaller/pyinstaller/wiki/Recipe-pytz-zip-file
# DOESN'T WORK ON MAC!

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

if sys.platform == "darwin":
    icon = '../resources/maproom.icns'
    exe = EXE(pyz,
        a.scripts,
        options,
        exclude_binaries=True,
        name=appname,
        debug=DEBUG,
        strip=not DEBUG,
        upx=not DEBUG,
        console=False,
        icon=icon)
    coll = COLLECT(exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=None,
        upx=not DEBUG,
        name=appname)
    app = BUNDLE(coll,
       name="%s.app" % appname,
       bundle_identifier="gov.noaa.maproom",
       icon=icon)

elif sys.platform == "win32":
    if not DEBUG:
        exe = EXE(pyz,
            a.scripts,
            a.binaries,
            a.zipfiles,
            a.datas,
            name=appname,
            debug=False,
            strip=False,
            upx=True,
            console=False,
            icon="../maproom/icons/maproom.ico")
    else:
        exe = EXE(pyz,
            a.scripts,
            options,
            exclude_binaries=True,
            name=appname,
            debug=True,
            strip=False,
            upx=True,
            console=True,
            icon="../maproom/icons/maproom.ico")
        coll = COLLECT(exe,
            a.binaries,
            a.zipfiles,
            a.datas,
            strip=False,
            upx=True,
            name=appname)

