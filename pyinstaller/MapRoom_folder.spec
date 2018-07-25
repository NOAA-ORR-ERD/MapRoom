# -*- mode: python -*-

block_cipher = None

appname = "MapRoom_folder"
bundle = False

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
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[os.path.join(maproom_path, 'pyinstaller')],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter', 'Cython', 'sphinx', 'nose', 'pygments'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

for pymod, path, tag in sorted(a.pure):
    if ".qt" in pymod or ".test" in pymod:
        print("why is this still here?", pymod)
    else:
        print(pymod, path, tag)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=appname,
          debug=True,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name=appname)
