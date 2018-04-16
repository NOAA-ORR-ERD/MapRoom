#!/bin/bash

# Update the Version.py file and Changelog for a new version
# Also commits the version to git and creates a tag

function commit_new_version {
    VERSION=`python make-changelog.py --next-version`
    echo $VERSION

    cat maproom/Version.py|sed -e s/VERSION.*/VERSION\ =\ \"$VERSION\"/ > Version.py.new
    mv Version.py.new maproom/Version.py

    python make-changelog.py 

    git commit -m "updated ChangeLog & Version.py for $VERSION" ChangeLog maproom/Version.py

    git tag -a $VERSION -m "Released $VERSION"

    (cd pyinstaller; python build_pyinstaller.py)
}

(cd tests; bash run.sh)
if test $? == 0
then
    echo All tests passed... Creating new version
    commit_new_version
else
    echo Unit test failure. Not creating new version
fi
