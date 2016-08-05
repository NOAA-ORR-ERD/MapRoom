#!/bin/bash -x
set -v

echo $CONDA_DEFAULT_ENV
echo $PATH

TARGET=MapRoom27
cp maproom.py $TARGET.py
rm -rf build/$TARGET dist/$TARGET dist/$TARGET.app
pyinstaller -y -i maproom/icons/maproom.icns --osx-bundle-identifier gov.noaa.maproom --debug --additional-hooks-dir=pyinstaller --windowed $TARGET.py

# Manual fixes
cp pyinstaller/Info.plist dist/$TARGET.app/Contents

cd dist/$TARGET.app/Contents/MacOS/

# fixup the duplicate wxPython libs
rm libwx_osx_cocoau-3.0.dylib 
ln -s libwx_osx_cocoau-3.0.0.2.0.dylib libwx_osx_cocoau-3.0.dylib

# fixup the missing geos library
ln -s libgeos_c.1.dylib libgeos_c.dylib

# CLEANUP
cd ../../../..
#rm $TARGET.py
