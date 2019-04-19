#!/bin/bash

case $PATH in
	*conda*)
        ./py.testw --cov=maproom --cov-report html --cov-report term
        ;;
    *)
	    py.test --cov=maproom --cov-report html --cov-report term
	    ;;
esac
RESULT=$?
exit $RESULT
