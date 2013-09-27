Sample data for MapRoom
========================

This directory contains assorted sample data for MapRoom. Each directoy has a different type of data in it


ChartsAndImages
-------------------------
Assorted raster images -- NOAA nautical charts are key (called BSB, but the *.kap files have all the data). Other formats as well, we shuld supporting reading and displying all GDAL-supported raster tyes that makes sense (i.e. palleted, RGB, RGBA, and greyscale)


Verdat
-------------------------
Old format for CATS: bathymetry points and boundaries (defined on those points) -- still need to support read/write/edit.


BNA
-------------------------
Simple text format for polygons -- used for GNOME shoreline. Need to read/write and edit. (and display...)


ENCs
-------------------------
NOAA vector electronic nautical charts -- there is a lot of data in these
(layers) -- we need bathymetry points and shoreline, maybe nothing else.


NGA-DNCs
-------------------------
Verdat nautical charts from NGA -- different format, but much of the same info as the NOAA ENCs -- again, shoreline and bathymetry points.


Binary LE
-------------------------
old binary format from GNOME -- we can probably ignore this one.


MOSS
-------------------------
Older text format for GNOME output -- maybe be abel to ignore.


Triangle
-------------------------
Formats native to the "triangle" program -- we're using the code to triangulate, but probably dont need to read or write this.
