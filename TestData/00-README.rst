Sample data for MapRoom
========================

This directory contains assorted sample data for MapRoom. Each directoy has a different type of data in it. 


ChartsAndImages
-------------------------
Assorted raster images -- NOAA nautical charts are key (called BSB, but the *.kap files have all the data). Other formats as well, we shuld supporting reading and displying all GDAL-supported raster types that makes sense (i.e. palleted, RGB, RGBA, and greyscale)

Specifics:
............
11361_1.KAP, 11361_2.KAP, 11361_3.KAP, 11361_4.KAP
    Charts of Mississipi River delta -- main and insets. 

11370_1.KAP, 11370_2.KAP, 11370_3.KAP
    Charts of the Missippi: not north up, but yes mercator.


13003_1.KAP
    Chart of Northeastern US -- large scale. 

13260_1.KAP
    Mercator, Bay of Fundy to Cape Cod

14771_1.KAP
   POLYCONIC and not north-up projection -- good test case for non-mercator.

16004_1.KAP
    Alaska North Slope -- very far north, but still mercator.

16592_1.KAP
   Alaska: Kodiak Island -- far north and west, mecator.

18649_1.KAP
    San Francisco Bay Entrance.
SanDiego_1.KAP SanDiego_2.KAP
    San Diego Bay -- these two should fit together well. Matches one of the verdats.

Admiralty-0463-2.tif Admiralty-0465-2.tif
    Admiralty charts of Haiti -- converted to geotiff -- grayscale, but seem to be palleted RGB

CoastwatchSST-geotiff.tif
    random image as a geotiff

MobileBay.KAP
    Mobile Bay chart -- matches MobileBay.verdat

NOAA18649.png, NOAA18649.png.aux.xml
    SFBAy as a png -- with xml geo-referencing file.

NOAA18649_partial.png NOAA18649_small.png
    parts of NOAA chart -- jsut PNG, no geo-referencing -- shoudl be abel to geo-reference in MapRoom, ultimately.

SST.png, SST.pgw
    png with accompanying "world file" should load and display, referenced to east coast US.

gs_09apr16_0227_mult_geo.png, gs_09apr16_0227_mult_geo.png.aux.xml
    png with aux.xml file for geo-referencing -- alighned with east coast.

o39075g2-geo.tif
    geotiff of USGS topo map.

o39075g2.tif o39075g2.tfw 
    regulat tif with "world file" -- same as above.


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


NC_particles
----------------
Netcdf "particles" file -- output from GNOME. You should be able to read these with the nc_particles module found in the gnome_tools project:

<https://github.com/NOAA-ORR-ERD/GnomeTools/tree/master/post_gnome/post_gnome>


Binary LE
-------------------------
old binary format from GNOME -- we can probably ignore this one.


MOSS
-------------------------
Older text format for GNOME output -- maybe be abel to ignore.


Triangle
-------------------------
Formats native to the "triangle" program -- we're using the code to triangulate, but probably dont need to read or write this.
