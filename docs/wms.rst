====================
Web Mapping Services
====================

* `ERMA web application <https://erma.noaa.gov/atlantic/erma.html#/x=-76.85270&y=37.93959&z=13&layers=27+11355>`_


WMS
===

* Lots of WMS servers at the national map site: http://nationalmap.gov/small_scale/infodocs/wms_intro.html




WMS Examples
============

NOAA_RNC
--------

* WMS 1.1.1 URL: http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WMSServer?

`Capabilities <http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0>`_::

    <WMS_Capabilities xmlns="http://www.opengis.net/wms" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esri_wms="http://www.esri.com/wms" version="1.3.0" xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.esri.com/wms http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?version=1.3.0%26service=WMS%26request=GetSchemaExtension">
    <Service>
    <Name>
    <![CDATA[ WMS ]]>
    </Name>
    <Title>
    <![CDATA[ RNC_NOAA_RNC ]]>
    </Title>
    <Abstract>WMS</Abstract>
    <KeywordList>
    <Keyword>
    <![CDATA[ ]]>
    </Keyword>
    </KeywordList>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?"/>
    <ContactInformation>
    <ContactPersonPrimary>
    <ContactPerson>
    <![CDATA[ ]]>
    </ContactPerson>
    <ContactOrganization>
    <![CDATA[ ]]>
    </ContactOrganization>
    </ContactPersonPrimary>
    <ContactPosition>
    <![CDATA[ ]]>
    </ContactPosition>
    <ContactAddress>
    <AddressType>
    <![CDATA[ ]]>
    </AddressType>
    <Address>
    <![CDATA[ ]]>
    </Address>
    <City>
    <![CDATA[ ]]>
    </City>
    <StateOrProvince>
    <![CDATA[ ]]>
    </StateOrProvince>
    <PostCode>
    <![CDATA[ ]]>
    </PostCode>
    <Country>
    <![CDATA[ ]]>
    </Country>
    </ContactAddress>
    <ContactVoiceTelephone>
    <![CDATA[ ]]>
    </ContactVoiceTelephone>
    <ContactFacsimileTelephone>
    <![CDATA[ ]]>
    </ContactFacsimileTelephone>
    <ContactElectronicMailAddress>
    <![CDATA[ ]]>
    </ContactElectronicMailAddress>
    </ContactInformation>
    <Fees>
    <![CDATA[ ]]>
    </Fees>
    <AccessConstraints>
    <![CDATA[ ]]>
    </AccessConstraints>
    <MaxWidth>4096</MaxWidth>
    <MaxHeight>4096</MaxHeight>
    </Service>
    <Capability>
    <Request>
    <GetCapabilities>
    <Format>application/vnd.ogc.wms_xml</Format>
    <Format>text/xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetCapabilities>
    <GetMap>
    <Format>image/bmp</Format>
    <Format>image/jpeg</Format>
    <Format>image/tiff</Format>
    <Format>image/png</Format>
    <Format>image/png8</Format>
    <Format>image/png24</Format>
    <Format>image/png32</Format>
    <Format>image/gif</Format>
    <Format>image/svg+xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetMap>
    <GetFeatureInfo>
    <Format>application/vnd.esri.wms_raw_xml</Format>
    <Format>application/vnd.esri.wms_featureinfo_xml</Format>
    <Format>application/vnd.ogc.wms_xml</Format>
    <Format>application/geojson</Format>
    <Format>text/xml</Format>
    <Format>text/html</Format>
    <Format>text/plain</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetFeatureInfo>
    <esri_wms:GetStyles>
    <Format>application/vnd.ogc.sld+xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?"/>
    </Get>
    </HTTP>
    </DCPType>
    </esri_wms:GetStyles>
    </Request>
    <Exception>
    <Format>application/vnd.ogc.se_xml</Format>
    <Format>application/vnd.ogc.se_inimage</Format>
    <Format>application/vnd.ogc.se_blank</Format>
    <Format>text/xml</Format>
    <Format>XML</Format>
    </Exception>
    <Layer>
    <Title>
    <![CDATA[ NOAA RNCs ]]>
    </Title>
    <CRS>CRS:84</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:3857</CRS>
    <!--  alias 3857  -->
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-179.999996</westBoundLongitude>
    <eastBoundLongitude>179.999996</eastBoundLongitude>
    <southBoundLatitude>-89.000000</southBoundLatitude>
    <northBoundLatitude>89.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-179.999996" miny="-89.000000" maxx="179.999996" maxy="89.000000"/>
    <BoundingBox CRS="EPSG:4326" minx="-89.000000" miny="-179.999996" maxx="89.000000" maxy="179.999996"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037507.842788" miny="-30240971.458386" maxx="20037507.842788" maxy="30240971.458386"/>
    <Layer queryable="1">
    <Title>
    <![CDATA[ NOAA_RNC ]]>
    </Title>
    <Abstract>
    <![CDATA[
    The NOAA_RNC MapService provides a seamless collarless mosaic of the NOAA Raster Nautical Charts. Source charts are updated before the 10th of every month. This map service is not to be used for navigation.
    ]]>
    </Abstract>
    <CRS>CRS:84</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:3857</CRS>
    <!--  alias 3857  -->
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-179.999989</westBoundLongitude>
    <eastBoundLongitude>179.999989</eastBoundLongitude>
    <southBoundLatitude>-14.647070</southBoundLatitude>
    <northBoundLatitude>74.915788</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-179.999989" miny="-14.647070" maxx="179.999989" maxy="74.915788"/>
    <BoundingBox CRS="EPSG:4326" minx="-14.647070" miny="-179.999989" maxx="74.915788" maxy="179.999989"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037507.067200" miny="-1648559.538400" maxx="20037507.067200" maxy="12896121.959700"/>
    <Layer queryable="1">
    <Name>1</Name>
    <Title>
    <![CDATA[ NOAA Raster Charts ]]>
    </Title>
    <Abstract>
    <![CDATA[
    The NOAA_RNC MapService provides a seamless collarless mosaic of the NOAA Raster Nautical Charts. Source charts are updated once per month. This map service is not to be used for navigation.
    ]]>
    </Abstract>
    <CRS>CRS:84</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:3857</CRS>
    <!--  alias 3857  -->
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-179.999996</westBoundLongitude>
    <eastBoundLongitude>179.999996</eastBoundLongitude>
    <southBoundLatitude>-89.000000</southBoundLatitude>
    <northBoundLatitude>89.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-179.999996" miny="-89.000000" maxx="179.999996" maxy="89.000000"/>
    <BoundingBox CRS="EPSG:4326" minx="-89.000000" miny="-179.999996" maxx="89.000000" maxy="179.999996"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037507.842788" miny="-30240971.458386" maxx="20037507.842788" maxy="30240971.458386"/>
    <Style>
    <Name>default</Name>
    <Title>1</Title>
    <LegendURL width="100" height="48">
    <Format>image/png</Format>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?request=GetLegendGraphic%26version=1.3.0%26format=image/png%26layer=1" xlink:type="simple"/>
    </LegendURL>
    </Style>
    </Layer>
    <Layer queryable="1">
    <Name>2</Name>
    <Title>
    <![CDATA[ NOAA Raster Chart Footprints ]]>
    </Title>
    <Abstract>
    <![CDATA[
    The NOAA_RNC MapService provides a seamless collarless mosaic of the NOAA Raster Nautical Charts. Source charts are updated once per month. This map service is not to be used for navigation.
    ]]>
    </Abstract>
    <CRS>CRS:84</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:3857</CRS>
    <!--  alias 3857  -->
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-179.999989</westBoundLongitude>
    <eastBoundLongitude>179.999989</eastBoundLongitude>
    <southBoundLatitude>-14.647070</southBoundLatitude>
    <northBoundLatitude>74.915788</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-179.999989" miny="-14.647070" maxx="179.999989" maxy="74.915788"/>
    <BoundingBox CRS="EPSG:4326" minx="-14.647070" miny="-179.999989" maxx="74.915788" maxy="179.999989"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037507.067200" miny="-1648559.538400" maxx="20037507.067200" maxy="12896121.959700"/>
    <Style>
    <Name>default</Name>
    <Title>2</Title>
    <LegendURL width="64" height="80">
    <Format>image/png</Format>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?request=GetLegendGraphic%26version=1.3.0%26format=image/png%26layer=2" xlink:type="simple"/>
    </LegendURL>
    </Style>
    </Layer>
    <Layer queryable="1">
    <Name>3</Name>
    <Title>
    <![CDATA[ NOAA RNC Boundary ]]>
    </Title>
    <Abstract>
    <![CDATA[
    The NOAA_RNC MapService provides a seamless collarless mosaic of the NOAA Raster Nautical Charts. Source charts are updated once per month. This map service is not to be used for navigation.
    ]]>
    </Abstract>
    <CRS>CRS:84</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:3857</CRS>
    <!--  alias 3857  -->
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-179.999989</westBoundLongitude>
    <eastBoundLongitude>179.999989</eastBoundLongitude>
    <southBoundLatitude>-14.647070</southBoundLatitude>
    <northBoundLatitude>74.915788</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-179.999989" miny="-14.647070" maxx="179.999989" maxy="74.915788"/>
    <BoundingBox CRS="EPSG:4326" minx="-14.647070" miny="-179.999989" maxx="74.915788" maxy="179.999989"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037507.067200" miny="-1648559.538400" maxx="20037507.067200" maxy="12896121.959700"/>
    <Style>
    <Name>default</Name>
    <Title>3</Title>
    <LegendURL width="16" height="16">
    <Format>image/png</Format>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="http://seamlessrnc.nauticalcharts.noaa.gov/arcgis/services/RNC/NOAA_RNC/MapServer/WmsServer?request=GetLegendGraphic%26version=1.3.0%26format=image/png%26layer=3" xlink:type="simple"/>
    </LegendURL>
    </Style>
    </Layer>
    </Layer>
    </Layer>
    </Capability>
    </WMS_Capabilities>

Maritime Chart Server
---------------------

`Sample chart <http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?BBOX=-8556942.2885109,4566851.4970803,-8551142.6289909,4570907.4368929&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0>`_

NOTE: Requires upper case URL params

* WMS 1.3.0 URL: http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?

`Capabilities <http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0>`_::

    <WMS_Capabilities xmlns="http://www.opengis.net/wms" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esri_wms="http://www.esri.com/wms" version="1.3.0" xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.esri.com/wms">
    <Service>
    <Name>
    <![CDATA[ WMS ]]>
    </Name>
    <Title>
    <![CDATA[ S57 ]]>
    </Title>
    <Abstract>WMS</Abstract>
    <KeywordList>
    <Keyword>
    <![CDATA[ S57 ]]>
    </Keyword>
    </KeywordList>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer"/>
    <ContactInformation>
    <ContactPersonPrimary>
    <ContactPerson>
    <![CDATA[ ]]>
    </ContactPerson>
    <ContactOrganization>
    <![CDATA[ ]]>
    </ContactOrganization>
    </ContactPersonPrimary>
    <ContactPosition>
    <![CDATA[ ]]>
    </ContactPosition>
    <ContactAddress>
    <AddressType>
    <![CDATA[ ]]>
    </AddressType>
    <Address>
    <![CDATA[ ]]>
    </Address>
    <City>
    <![CDATA[ ]]>
    </City>
    <StateOrProvince>
    <![CDATA[ ]]>
    </StateOrProvince>
    <PostCode>
    <![CDATA[ ]]>
    </PostCode>
    <Country>
    <![CDATA[ ]]>
    </Country>
    </ContactAddress>
    <ContactVoiceTelephone>
    <![CDATA[ ]]>
    </ContactVoiceTelephone>
    <ContactFacsimileTelephone>
    <![CDATA[ ]]>
    </ContactFacsimileTelephone>
    <ContactElectronicMailAddress>
    <![CDATA[ ]]>
    </ContactElectronicMailAddress>
    </ContactInformation>
    <Fees>
    <![CDATA[ ]]>
    </Fees>
    <AccessConstraints>
    <![CDATA[ ]]>
    </AccessConstraints>
    <MaxWidth>2048</MaxWidth>
    <MaxHeight>2048</MaxHeight>
    </Service>
    <Capability>
    <Request>
    <GetCapabilities>
    <Format>application/vnd.ogc.wms_xml</Format>
    <Format>text/xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetCapabilities>
    <GetMap>
    <Format>image/png</Format>
    <Format>image/png8</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetMap>
    <GetFeatureInfo>
    <Format>text/html</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetFeatureInfo>
    <esri_wms:GetStyles>
    <Format>application/vnd.ogc.sld+xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://gis.charttools.noaa.gov/arcgis/rest/services/MCS/ENCOnline/MapServer/exts/Maritime%20Chart%20Server/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </esri_wms:GetStyles>
    </Request>
    <Exception>
    <Format>application/vnd.ogc.se_xml</Format>
    <Format>application/vnd.ogc.se_inimage</Format>
    <Format>application/vnd.ogc.se_blank</Format>
    <Format>text/xml</Format>
    <Format>XML</Format>
    </Exception>
    <Layer>
    <Title>
    <![CDATA[ Layers ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    <Layer queryable="1">
    <Name>7</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Services and small craft facilities ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>6</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Buoys, beacons, lights, fog signals, radar ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>5</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Special areas ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>4</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Traffic routes ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>3</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Seabed, obstructions, pipelines ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>2</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Depths, currents, etc ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>1</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Natural and man-made features, port features ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    <Layer queryable="1">
    <Name>0</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Information about the chart display ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180.000000</westBoundLongitude>
    <eastBoundLongitude>180.000000</eastBoundLongitude>
    <southBoundLatitude>-64.850000</southBoundLatitude>
    <northBoundLatitude>74.000000</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-20037508.342789" miny="-9568971.310870" maxx="20037508.342789" maxy="12515545.212468"/>
    </Layer>
    </Layer>
    </Capability>
    </WMS_Capabilities>


Inland Charts
-------------

`Sample chart <http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?BBOX=-9151321.3960644,4688981.1460582,-9128122.757984,4705204.9053088&BUFFER=0&FORMAT=image%2Fpng&HEIGHT=849&LAYERS=0%2C1%2C2%2C3%2C4%2C5%2C6%2C7&REQUEST=GetMap&SERVICE=WMS&SRS=EPSG%3A102113&STYLES=&TRANSPARENT=true&VERSION=1.1.1&WIDTH=1214&etag=0>`_

NOTE: Requires upper case URL params

* WMS 1.3.0 URL: http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?

`Capabilities <http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0>`_::

    <WMS_Capabilities xmlns="http://www.opengis.net/wms" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:esri_wms="http://www.esri.com/wms" version="1.3.0" xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd http://www.esri.com/wms">
    <Service>
    <Name>
    <![CDATA[ WMS ]]>
    </Name>
    <Title>
    <![CDATA[ S57 ]]>
    </Title>
    <Abstract>WMS</Abstract>
    <KeywordList>
    <Keyword>
    <![CDATA[ S57 ]]>
    </Keyword>
    </KeywordList>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer"/>
    <ContactInformation>
    <ContactPersonPrimary>
    <ContactPerson>
    <![CDATA[ ]]>
    </ContactPerson>
    <ContactOrganization>
    <![CDATA[ ]]>
    </ContactOrganization>
    </ContactPersonPrimary>
    <ContactPosition>
    <![CDATA[ ]]>
    </ContactPosition>
    <ContactAddress>
    <AddressType>
    <![CDATA[ ]]>
    </AddressType>
    <Address>
    <![CDATA[ ]]>
    </Address>
    <City>
    <![CDATA[ ]]>
    </City>
    <StateOrProvince>
    <![CDATA[ ]]>
    </StateOrProvince>
    <PostCode>
    <![CDATA[ ]]>
    </PostCode>
    <Country>
    <![CDATA[ ]]>
    </Country>
    </ContactAddress>
    <ContactVoiceTelephone>
    <![CDATA[ ]]>
    </ContactVoiceTelephone>
    <ContactFacsimileTelephone>
    <![CDATA[ ]]>
    </ContactFacsimileTelephone>
    <ContactElectronicMailAddress>
    <![CDATA[ ]]>
    </ContactElectronicMailAddress>
    </ContactInformation>
    <Fees>
    <![CDATA[ ]]>
    </Fees>
    <AccessConstraints>
    <![CDATA[ ]]>
    </AccessConstraints>
    <MaxWidth>2048</MaxWidth>
    <MaxHeight>2048</MaxHeight>
    </Service>
    <Capability>
    <Request>
    <GetCapabilities>
    <Format>application/vnd.ogc.wms_xml</Format>
    <Format>text/xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetCapabilities>
    <GetMap>
    <Format>image/png</Format>
    <Format>image/png8</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetMap>
    <GetFeatureInfo>
    <Format>text/html</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetFeatureInfo>
    <esri_wms:GetStyles>
    <Format>application/vnd.ogc.sld+xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:type="simple" xlink:href="http://maps8.arcgisonline.com/arcgis/rest/services/USACE_InlandENC/MapServer/exts/Maritime%20Chart%20Service/WMSServer"/>
    </Get>
    </HTTP>
    </DCPType>
    </esri_wms:GetStyles>
    </Request>
    <Exception>
    <Format>application/vnd.ogc.se_xml</Format>
    <Format>application/vnd.ogc.se_inimage</Format>
    <Format>application/vnd.ogc.se_blank</Format>
    <Format>text/xml</Format>
    <Format>XML</Format>
    </Exception>
    <Layer>
    <Title>
    <![CDATA[ Layers ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    <Layer queryable="1">
    <Name>7</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Services and small craft facilities ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>6</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Buoys, beacons, lights, fog signals, radar ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>5</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Special areas ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>4</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Traffic routes ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>3</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Seabed, obstructions, pipelines ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>2</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Depths, currents, etc ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>1</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Natural and man-made features, port features ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    <Layer queryable="1">
    <Name>0</Name>
    <Abstract>
    <![CDATA[ S57 Data ]]>
    </Abstract>
    <Title>
    <![CDATA[ Information about the chart display ]]>
    </Title>
    <CRS>EPSG:102100</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-96.699746</westBoundLongitude>
    <eastBoundLongitude>-79.502292</eastBoundLongitude>
    <southBoundLatitude>28.877388</southBoundLatitude>
    <northBoundLatitude>45.142195</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:102100" minx="-10764566.540219" miny="3360049.485969" maxx="-8850154.684602" maxy="5643935.025904"/>
    </Layer>
    </Layer>
    </Capability>
    </WMS_Capabilities>


ows.terrestris.de
-----------------

* WMS 1.3.0 URL: http://ows.terrestris.de/osm/service?

`Capabilities <http://ows.terrestris.de/osm/service?SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0>`_::

    <WMS_Capabilities xmlns="http://www.opengis.net/wms" xmlns:sld="http://www.opengis.net/sld" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.3.0" xsi:schemaLocation="http://www.opengis.net/wms http://schemas.opengis.net/wms/1.3.0/capabilities_1_3_0.xsd">
    <Service>
    <Name>WMS</Name>
    <Title>OpenStreetMap WMS Deutschland</Title>
    <Abstract>
    OpenStreetMap WMS für Deutschland, bereitgestellt durch terrestris GmbH und Co. KG. Beschleunigt mit MapProxy (http://mapproxy.org/)
    </Abstract>
    <OnlineResource xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="http://www.terrestris.de"/>
    <ContactInformation>
    <ContactPersonPrimary>
    <ContactPerson>Johannes Weskamm</ContactPerson>
    <ContactOrganization>terrestris GmbH und Co. KG</ContactOrganization>
    </ContactPersonPrimary>
    <ContactPosition>Technical Director</ContactPosition>
    <ContactAddress>
    <AddressType>postal</AddressType>
    <Address>Pützchens Chaussee 56</Address>
    <City>Bonn</City>
    <StateOrProvince/>
    <PostCode>53227</PostCode>
    <Country>Germany</Country>
    </ContactAddress>
    <ContactVoiceTelephone>+49(0)228 962 899 51</ContactVoiceTelephone>
    <ContactFacsimileTelephone>+49(0)228 962 899 57</ContactFacsimileTelephone>
    <ContactElectronicMailAddress>info@terrestris.de</ContactElectronicMailAddress>
    </ContactInformation>
    <Fees>None</Fees>
    <AccessConstraints>
    (c) OpenStreetMap contributors (http://www.openstreetmap.org/copyright) (c) OpenStreetMap Data (http://openstreetmapdata.com) (c) Natural Earth Data (http://www.naturalearthdata.com)
    </AccessConstraints>
    </Service>
    <Capability>
    <Request>
    <GetCapabilities>
    <Format>text/xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xlink:href="http://ows.terrestris.de/osm/service?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetCapabilities>
    <GetMap>
    <Format>image/gif</Format>
    <Format>image/png</Format>
    <Format>image/jpeg</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xlink:href="http://ows.terrestris.de/osm/service?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetMap>
    <GetFeatureInfo>
    <Format>text/plain</Format>
    <Format>text/html</Format>
    <Format>text/xml</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xlink:href="http://ows.terrestris.de/osm/service?"/>
    </Get>
    </HTTP>
    </DCPType>
    </GetFeatureInfo>
    <sld:GetLegendGraphic>
    <Format>image/gif</Format>
    <Format>image/png</Format>
    <Format>image/jpeg</Format>
    <DCPType>
    <HTTP>
    <Get>
    <OnlineResource xlink:href="http://ows.terrestris.de/osm/service?"/>
    </Get>
    </HTTP>
    </DCPType>
    </sld:GetLegendGraphic>
    </Request>
    <Exception>
    <Format>XML</Format>
    <Format>INIMAGE</Format>
    <Format>BLANK</Format>
    </Exception>
    <Layer queryable="1">
    <Title>OpenStreetMap WMS Deutschland</Title>
    <CRS>EPSG:900913</CRS>
    <CRS>EPSG:3857</CRS>
    <CRS>EPSG:25832</CRS>
    <CRS>EPSG:25833</CRS>
    <CRS>EPSG:29192</CRS>
    <CRS>EPSG:29193</CRS>
    <CRS>EPSG:31466</CRS>
    <CRS>EPSG:31467</CRS>
    <CRS>EPSG:31468</CRS>
    <CRS>EPSG:32648</CRS>
    <CRS>EPSG:4326</CRS>
    <CRS>EPSG:4674</CRS>
    <CRS>EPSG:3068</CRS>
    <CRS>EPSG:2100</CRS>
    <CRS>EPSG:3034</CRS>
    <CRS>EPSG:3035</CRS>
    <CRS>EPSG:31463</CRS>
    <CRS>EPSG:4258</CRS>
    <CRS>EPSG:4839</CRS>
    <CRS>EPSG:2180</CRS>
    <CRS>EPSG:21781</CRS>
    <CRS>EPSG:2056</CRS>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180</westBoundLongitude>
    <eastBoundLongitude>180</eastBoundLongitude>
    <southBoundLatitude>-89.999999</southBoundLatitude>
    <northBoundLatitude>89.999999</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-180" miny="-89.999999" maxx="180" maxy="89.999999"/>
    <BoundingBox CRS="EPSG:900913" minx="-20037508.3428" miny="-147730762.67" maxx="20037508.3428" maxy="147730758.195"/>
    <BoundingBox CRS="EPSG:4326" minx="-90" miny="-180" maxx="90" maxy="180"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037508.3428" miny="-147730762.67" maxx="20037508.3428" maxy="147730758.195"/>
    <Layer queryable="1">
    <Name>OSM-WMS</Name>
    <Title>OpenStreetMap WMS - by terrestris</Title>
    <EX_GeographicBoundingBox>
    <westBoundLongitude>-180</westBoundLongitude>
    <eastBoundLongitude>180</eastBoundLongitude>
    <southBoundLatitude>-89.999999</southBoundLatitude>
    <northBoundLatitude>89.999999</northBoundLatitude>
    </EX_GeographicBoundingBox>
    <BoundingBox CRS="CRS:84" minx="-180" miny="-89.999999" maxx="180" maxy="89.999999"/>
    <BoundingBox CRS="EPSG:900913" minx="-20037508.3428" miny="-147730762.67" maxx="20037508.3428" maxy="147730758.195"/>
    <BoundingBox CRS="EPSG:4326" minx="-90" miny="-180" maxx="90" maxy="180"/>
    <BoundingBox CRS="EPSG:3857" minx="-20037508.3428" miny="-147730762.67" maxx="20037508.3428" maxy="147730758.195"/>
    <Style>
    <Name>default</Name>
    <Title>default</Title>
    <LegendURL>
    <Format>image/png</Format>
    <OnlineResource xlink:type="simple" xlink:href="http://ows.terrestris.de/osm/service?styles=&layer=OSM-WMS&service=WMS&format=image%2Fpng&sld_version=1.1.0&request=GetLegendGraphic&version=1.1.1"/>
    </LegendURL>
    </Style>
    </Layer>
    </Layer>
    </Capability>
    </WMS_Capabilities>




WMTS
====

* http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&\LAYER=VIIRS_CityLights_2012&STYLE=default&TILEMATRIXSET=EPSG4326_500m&\TILEMATRIX=6&TILEROW=4&TILECOL=4&FORMAT=image%2Fjpeg

