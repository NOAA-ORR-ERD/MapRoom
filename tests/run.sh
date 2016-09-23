#!/bin/bash

ARGS=--cov=maproom
case $PATH in
	*conda*)
        ./py.testw $ARGS
        ;;
    *)
	    py.test $ARGS
	    ;;
esac
RESULT=$?
exit $RESULT
