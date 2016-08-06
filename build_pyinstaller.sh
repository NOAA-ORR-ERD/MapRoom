#!/bin/bash -x
set -v

echo $CONDA_DEFAULT_ENV
echo $PATH

# can't use MapRoom because maproom is a directory name and the filesystem is
# case-insensitive
BUILD_TARGET=MapRoom_app

# Use the old py2app name in case users have symlinks or something and they are
# updating in place
FINAL_TARGET=Maproom

cp maproom.py $BUILD_TARGET.py
rm -rf build/$BUILD_TARGET dist/$BUILD_TARGET dist/$BUILD_TARGET.app dist/$FINAL_TARGET.app
pyinstaller -y -i maproom/icons/maproom.icns --osx-bundle-identifier gov.noaa.maproom --debug --additional-hooks-dir=pyinstaller --windowed $BUILD_TARGET.py

#### Configuration fixes

# use more detailed Info.plist than the one created by pyinstaller
cp pyinstaller/Info.plist dist/$BUILD_TARGET.app/Contents

#### Library fixes

cd dist/$BUILD_TARGET.app/Contents/MacOS/

# fixup the duplicate wxPython libs
rm libwx_osx_cocoau-3.0.dylib 
ln -s libwx_osx_cocoau-3.0.0.2.0.dylib libwx_osx_cocoau-3.0.dylib

# fixup the missing geos library
ln -s libgeos_c.1.dylib libgeos_c.dylib

#### CLEANUP
cd ../../../..
rm $BUILD_TARGET.py

# Make it the same application name as the old py2app name
mv dist/$BUILD_TARGET.app dist/$FINAL_TARGET.app
