#!/bin/bash

case $PATH in
	*conda*)
        ./py.testw
        ;;
    *)
	    py.test
	    ;;
esac
RESULT=$?
exit $RESULT
