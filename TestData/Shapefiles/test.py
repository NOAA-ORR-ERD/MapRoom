from osgeo import ogr
ogr.UseExceptions()

def summarize(filename):
    ds = ogr.Open(filename, 0)
    print dir(ds)
    layer = ds.GetLayer()
    capabilities = [
        ogr.OLCRandomRead,
        ogr.OLCSequentialWrite,
        ogr.OLCRandomWrite,
        ogr.OLCFastSpatialFilter,
        ogr.OLCFastFeatureCount,
        ogr.OLCFastGetExtent,
        ogr.OLCCreateField,
        ogr.OLCDeleteField,
        ogr.OLCReorderFields,
        ogr.OLCAlterFieldDefn,
        ogr.OLCTransactions,
        ogr.OLCDeleteFeature,
        ogr.OLCFastSetNextByIndex,
        ogr.OLCStringsAsUTF8,
        ogr.OLCIgnoreFields
    ]

    print("Layer Capabilities:")
    for cap in capabilities:
        print("  %s = %s" % (cap, layer.TestCapability(cap)))

    print("Features:")
    for feature in layer:
        #print " ", feature.GetField("STATE_NAME")
        geom = feature.GetGeometryRef()

        geo_type = geom.GetGeometryName()

        if geo_type == 'MULTIPOLYGON':
            for i in range(geom.GetGeometryCount()):
                poly = geom.GetGeometryRef(i)
                ring = poly.GetGeometryRef(i)

                print geo_type, i, ring.GetPoints()
        elif geo_type == 'POLYGON':
            poly = geom.GetGeometryRef(0)

            print geo_type, poly.GetPoints()
        elif geo_type == 'LINESTRING':
            line_strings.append(geom.GetPoints())
            print geo_type, geom.GetPoints()
        else:
            print 'unknown type: ', geo_type


if __name__ == "__main__":
    summarize("Overflight1050_May14.shp")
    summarize("20160516_0013d.shp")