#!/bin/bash

case $PATH in
	*conda*)
        ./py.testw --no-cov
        ;;
    *)
	    py.test --no-cov
	    ;;
esac
RESULT=$?
exit $RESULT
