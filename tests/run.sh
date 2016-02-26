#!/bin/bash

py.test --cov=maproom
RESULT=$?
exit $RESULT
