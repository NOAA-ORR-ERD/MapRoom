#!/bin/bash

# Update the Version.py file and Changelog for a new version
# Also commits the version to git and creates a tag

VERSION=`python make-changelog.py --next-version`
echo $VERSION

cat Version.py|sed -e s/VERSION.*/VERSION\ =\ \"$VERSION\"/ > Version.py.new
mv Version.py.new Version.py

python make-changelog.py 

python setup.py py2app

git commit -m "updated ChangeLog & Version.py for $VERSION" ChangeLog Version.py

git tag -a $VERSION -m "Released $VERSION"
