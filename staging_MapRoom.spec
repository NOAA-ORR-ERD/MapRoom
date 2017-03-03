# -*- mode: python -*-

block_cipher = None

import sys
sys.modules['FixTk'] = None

a = Analysis(['staging_MapRoom.py'],
             pathex=['/Users/rob.mcmullen/src/maproom'],
             binaries=None,
             datas=None,
             hiddenimports=[],
             hookspath=['pyinstaller'],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='staging_MapRoom',
          debug=True,
          strip=False,
          upx=True,
          console=False , icon='maproom/icons/maproom.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='staging_MapRoom')
app = BUNDLE(coll,
             name='staging_MapRoom.app',
             icon='maproom/icons/maproom.icns',
             bundle_identifier='gov.noaa.maproom')
