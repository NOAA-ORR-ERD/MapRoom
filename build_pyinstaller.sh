#!/bin/bash -x
set -v

# can't use MapRoom because maproom is a directory name and the filesystem is
# case-insensitive
BUILD_TARGET=staging_MapRoom

# Use the old py2app name in case users have symlinks or something and they are
# updating in place
FINAL_TARGET=Maproom

cp maproom.py $BUILD_TARGET.py
rm -rf build/$BUILD_TARGET dist/$BUILD_TARGET dist/$BUILD_TARGET.app dist/$FINAL_TARGET.app
#pyinstaller -y -i maproom/icons/maproom.icns --osx-bundle-identifier gov.noaa.maproom --debug --additional-hooks-dir=pyinstaller --windowed $BUILD_TARGET.py
pyinstaller -y --debug --windowed $BUILD_TARGET.spec

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

# remove unnecessary stuff
cd ../Resources
rm maproom/library/*.c
rm maproom/library/*.pyx
rm maproom/renderer/gl/*.c
rm maproom/renderer/gl/*.pyx
rm maproom/renderer/gl/*.h

cd ../../../..
rm $BUILD_TARGET.py

#### BUNDLE

VERSION=`python -c "import maproom.Version; print maproom.Version.VERSION"`
mkdir -p dist-$VERSION
rm -rf dist-$VERSION/$FINAL_TARGET.app dist-$VERSION/$FINAL_TARGET-$VERSION-darwin.zip

# Make it the same application name as the old py2app name, and do it in one
# step by removing any arch except 64 bit
/usr/bin/ditto -arch x86_64 dist/$BUILD_TARGET.app dist-$VERSION/$FINAL_TARGET.app

# create zip file
cd dist-$VERSION
tar cfj $FINAL_TARGET-$VERSION-darwin.tbz $FINAL_TARGET.app
cd ..
