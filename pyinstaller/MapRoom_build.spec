# -*- mode: python -*-

block_cipher = None

appname = "MapRoom_build"
bundle = True

with open("../maproom.py", "r") as fh:
    script = fh.read()
with open("%s.py" % appname, "w") as fh:
    fh.write(script)

import sys
sys.modules['FixTk'] = None

import os
maproom_path = os.path.abspath("..")


a = Analysis(["%s.py" % appname],
             pathex=[maproom_path],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=[os.path.join(maproom_path, 'pyinstaller')],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'Cython', 'sphinx', 'nose', 'pygments'],
             cipher=block_cipher)

for pymod, path, tag in sorted(a.pure):
  if ".qt" in pymod or ".test" in pymod:
    print "why is this still here?", pymod

# pytz zip bundle from https://github.com/pyinstaller/pyinstaller/wiki/Recipe-pytz-zip-file
# DOESN'T WORK ON MAC!

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

if sys.platform == "darwin":
    icon = '../resources/maproom.icns'
    exe = EXE(pyz,
        a.scripts,
        exclude_binaries=True,
        name=appname,
        debug=False,
        strip=True,
        upx=True,
        console=False,
        icon=icon)
    coll = COLLECT(exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=None,
        upx=True,
        name=appname)
    app = BUNDLE(coll,
       name="%s.app" % appname,
       bundle_identifier="gov.noaa.maproom",
       icon=icon)

elif sys.platform == "win32":
    if bundle:
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
        options = [ ('v', None, 'OPTION'),]
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

