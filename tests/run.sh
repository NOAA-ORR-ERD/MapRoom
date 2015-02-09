#!/bin/bash

nosetests -v --all-modules . ../maproom/library/ -a '!slow'
