let serviceDict = {
    'awifs_fcc':{datasetId:'T0S1P1',
        r_index:3,
        g_index:2,
        b_index:1,
        r_max:0.3,
        g_max:0.3,
        b_max:0.3,
        r_min:0.001,
        g_min:0.001,
        b_min:0.001,
        serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
        getLayer:buildRGBMapLayer
    },
    'sentinel2_fcc':{datasetId:'T0S1P0',
        r_index:8,
        g_index:4,
        b_index:3,
        r_max:6000,
        g_max:4000,
        b_max:4000,
        r_min:0,
        g_min:0,
        b_min:0,
        serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
        getLayer:buildRGBMapLayer
    },
    'sentinel2_NDVI_RGB':{datasetId:'T3S1P1',
      r_index:1,
      g_index:1,
      b_index:1,
      r_max:251,
      g_max:251,
      b_max:251,
      r_min:1,
      g_min:1,
      b_min:1,
        serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
        getLayer:NDVI_RGB
    },
    'Sentinel-2 NDVI Difference (A-B)':{datasetId:'T3S1P1',
        STYLES:"[-250:ff0000FF:-100:ff0000FF:-80:ff6026FF:-60:febf4cFF:-40:fec946FF:-20:ffebb4FF:0:ffffffff:20:e6f0e6FF:40:b4d3b3FF:60:82b681FF:80:4d974cFF:100:187817FF:250:187817FF]",
        serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
        getLayer:s2_ndvi_diff_layer
    },
    'sentinel2_NDVI':{
      param:'NDVI',
      datasetId:'T3S1P1',

      STYLES:"[0:FFFFFF00:1:f0ebecFF:25:d8c4b6FF:50:ab8a75FF:75:917732FF:100:70ab06FF:125:459200FF:150:267b01FF:175:0a6701FF:200:004800FF:255:001901FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
      getLayer:buildIndexMapLayer
    },

    'sentinel2_NDWI':{
      param:"NDWI",
      datasetId:'T0S1P0',
      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:FDFE25FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_NDMI':{
      param:'NDMI',
      datasetId:'T3S6P1',
      STYLES:  "[0:FFFFFF00:101:EE0000FF:126:FDFE25FF:133:CBD921FF:138:516B12FF:157:314606FF:169:152106FF:255:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_NDMI_CLOUD_MASK':{
      param:"NDMI_CMASK",
      datasetId:'T3S6P1',
      STYLES:"[-2:FFFFFF00:0:FFFFFF00:101:EE0000FF:126:FDFE25FF:133:CBD921FF:138:516B12FF:157:314606FF:169:152106FF:255:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server3/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_NBR':{
      param:"NBR",
      datasetId:'T0S1P0',
      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:FDFE25FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
    
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_NDSI':{
      param:"NDSI",
      datasetId:'T0S1P0',
      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:314606FF:0:152106FF:0.2:CBD921FF:0.4:0000FFFF:0.6:0000FF:0.8:516B12FF:1:516B12FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_NDSI_CLOUD_MASK':{
      param:"NDSI_CMASK",
      datasetId:'T0S1P0',

      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:314606FF:0:152106FF:0.2:CBD921FF:0.4:0000FFFF:0.6:0000FF:0.8:516B12FF:1:516B12FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },





    
    'sentinel2_WRF':{
      param:"WRF",
      datasetId:'T2S1P24',
      STYLES:
      "[0:30123bff:114:466be3ff:225:28bcebff:399:32f298ff:450:a4fc3cff:564:eecf3aff:675:fb7e21ff:789:d02f05ff:900:7a0403ff];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server/wms/',
      getLayer:buildWRFMapLayer
    },
    'sentinel2_TEMPERATURE_FORECAST':{
      param:"TEMPERATURE_FORECAST",
      datasetId:'T5S2P4',
      STYLES:"[-31:00000000:-20:0E0EFF00:-10:0E0EFFFF:0:4040FFFF:10:40A8FFFF:20:40FFFFFF:30:A8FFA8FF:40:FFFF40FF:50:FFA800FF:60:FF4000FF;nodata:FFFFFFFF]",
      serverUrl:'https://vedas.sac.gov.in/ridam_server3/wms/',
      getLayer:buildTemperatureForecast
    },




    'sentinel2_MNDWI':{
      param:"MNDWI",
      datasetId:'T0S1P0',
      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:FDFE25FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },    'sentinel2_MNDWI_CLOUD_MASK':{
      param:"MNDWI_CMASK",
      datasetId:'T0S1P0',
      STYLES:"[-1:FFFFFF00:-0.5:EE0000FF:-0.2:FDFE25FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/ridam_server2b/wms/',
      getLayer:buildIndexMapLayer
    },

    'sentinel2_SAVI':{
      param:"SAVI",
      datasetId:'T0S1P0',
      STYLES:  "[-1:FFFFFF00:-0.5:EE0000FF:-0.2:FDFE25FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_ARVI':{
      param:"ARVI",
      datasetId:'T0S1P0',
      STYLES:  "[-1:FFFFFF00:-0.5:EE0000FF:-0.2:0000FF:0.2:CBD921FF:0.4:516B12FF:0.6:314606FF:0.8:152106FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
 
    'sentinel2_BVI':{
      param:"BVI",
      datasetId:'T0S1P0',
      STYLES:  "[-1:FFFFFF00:-0.2:CBD921FF:-0.1:EE0000FF:0:FDFE25FF:0.1:CBD921FF:0.6:516B12FF:0.8:314606FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
    'sentinel2_EVI':{
      param:"EVI",
      datasetId:'T0S1P0',
      STYLES:  "[0:FFFFFF00:0.1:FFFFFF00:0.2:EE0000FF:0.4:FDFE25FF:0.6:CBD921FF:0.8:516B12FF:0.9:314606FF:1:152106FF];nodata:FFFFFF00",
      serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
      getLayer:buildIndexMapLayer
    },
   'sentinel2_BAND_PROFILE':{
    param:"BAND_PROFILE",
    datasetId:'T0S1P0',
    r_index:8,
    g_index:4,
    b_index:3,
    r_max:6000,
    g_max:4000,
    b_max:4000,
    r_min:0,
    g_min:0,
    b_min:0,
    serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
    getLayer:buildRGBMapLayer
   },
   'sentinel2_CLOUD_MASK':{
    param:"FUSED_CLOUD_MASK",
    datasetId:'T0S5P1',
    serverUrl:'https://vedas.sac.gov.in/vapi/ridam_server2/wms/',
    STYLES:"[0:FFFFFF00:1:FFFFFF00:2:FFFFFF00:3:643200FF:4:FFFFFF00:5:FFFFFF00:6:FFFFFF00:7:FFFFFF00:8:64c8ffFF:9:64c8ffFF:10:64c8ffFF:11:FFFFFF00];nodata:FFFFFF00",
    getLayer:buildCLOUD
  }
}

let dateURL =  "https://vedas.sac.gov.in/ridam_server3/meta/dataset_timestamp?prefix=";


//Sort array in descending order
async function sortDateArray(arr) {
  return arr.sort((a, b) => {
    return parseInt(b.val) - parseInt(a.val);
  });
}


async function formatDates(data, splitOn, dateAtIndex) {
  let processedData = data.map((dtTime) => {
    //Spliting date to get only yyyymmdd format date
    splittedDt = dtTime.split(splitOn);
    //at 0 index date in yyyymmdd and at 1 index time

    let requiredData = splittedDt[parseInt(dateAtIndex)];

    //get Year, month, date from required data
    let year = requiredData.substring(0, 4);
    let month = requiredData.substring(4, 6);
    let dt = requiredData.substring(6, 8);

    requiredData = year + month + dt;
    let label = year + "-" + month + "-" + dt;

    return { lbl: label, val: requiredData };
  });

  let sortedData = await this.sortDateArray(processedData);
  return sortedData;
}


async function getAsyncData(url) {
  console.log("URL is", url);
  let response = await fetch(url);
  let res = await response.json();

  return res;
}
async function getAddress(lon, lat) {
  try {
    let response = await fetch(
      "https://apis.mapmyindia.com/advancedmaps/v1/nwsgvbqbbw5ejwj112vvisgoggiq4ov3/rev_geocode?lat=" +
        lat +
        "&lng=" +
        lon
    );
    let result = await response.json();
    result = result["results"];
    result = result[0];
    let locInfo = result["formatted_address"];
    if (locInfo.startsWith("Unnamed Road")) {
      locInfo = locInfo.substring(14, 200);
    }
    let address =
      "<span style='font-weight:bold;color: #0488d0;font-size:16px'>(Lat, Long):</span><span style='font-size:16px'> (" +
      lat.toFixed(5) + ", " + lon.toFixed(5) +
      ") &nbsp;&nbsp;&nbsp;</span><span style='font-weight:bold;color: #0488d0;font-size:16px'>Location:</span><span style='font-size:16px'> " +
      locInfo + "</span>";
    return address;
  } catch {
    return (
      "<span style='font-weight:bold;color: #0488d0;font-size:16px'>(Lat, Long):</span><span style='font-size:16px'> (" +
      lat.toFixed(5) + ", " + lon.toFixed(5) +
      ") &nbsp;&nbsp;&nbsp;</span><span style='font-weight:bold;color: #0488d0;font-size:16px'>Location:</span><span style='font-size:16px'> " +
      "Reverse Geocoding Failed" + "</span>"
    );
  }
}
async function getTimestamps(serviceName){
  let serviceConfig = serviceDict[serviceName];

  console.log("service_config",serviceConfig)
  let serviceDatasetId = serviceConfig.datasetId;
  let date_url = dateURL + serviceDatasetId;
  console.log("dateurl",date_url)
  let res = await this.getAsyncData(date_url);
  //Get date array from response dictionary
  res = res["result"][serviceDatasetId];
  console.log("Response is",res);
  // if (serviceConfig.param === "WRF") {
  //   return res;
  // }
  let processedData = await this.formatDates(res, " ", 0);
  console.log("Processed data i", processedData);

  return processedData;
}

function buildVAPILayer(obj){
  let serviceConfig = serviceDict[obj.serviceName];

  console.log("NAME",serviceConfig)

  return serviceConfig.getLayer(obj);
}

function buildRGBMapLayer(obj){
    let serviceConfig = serviceDict[obj.serviceName];
    return new ol.layer.Tile({
        source: new ol.source.TileWMS({
          projection: "EPSG:4326",
          url: serviceConfig.serverUrl, 
          params: {
            name: "RDSGrdient",
            layers: "T0S0M1",
            PROJECTION: "EPSG:4326",
            ARGS:
              "r_dataset_id:" +
             serviceConfig.datasetId +
              ";g_dataset_id:" +
             serviceConfig.datasetId +
              ";b_dataset_id:" +
             serviceConfig.datasetId+
              ";r_from_time:" +
              (obj.from_time || obj.r_from_time)  +
              ";r_to_time:" +
              (obj.to_time || obj.r_to_time)  +
              ";g_from_time:" +
              (obj.from_time || obj.g_from_time) +
              ";g_to_time:" +
              (obj.to_time || obj.g_to_time) +
              ";b_from_time:" +
              (obj.from_time || obj.b_from_time)+
              ";b_to_time:" +
              (obj.to_time || obj.b_to_time) +
              ";r_index:"+serviceConfig.r_index+";g_index:"
              +serviceConfig.g_index+";b_index:"+serviceConfig.b_index
              +";r_max:"+serviceConfig.r_max+";g_max:"+serviceConfig.g_max
              +";b_max:"+serviceConfig.b_max
              +";r_min:"+serviceConfig.r_min+";g_min:"+serviceConfig.g_min
              +";b_min:"+serviceConfig.b_min,
             
            LEGEND_OPTIONS: "columnHeight:400;height:100",
            "X-API-KEY": obj.key,
          },
        }),
        opacity: 1,
        zIndex: 1,
      });
}


function NDVI_RGB(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  return new ol.layer.Tile({
      source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
          name: "RIDAM_RGB",
          layers: "T0S0M1",
          PROJECTION: "EPSG:4326",
          ARGS:"r_merge_method:max;g_merge_method:max;b_merge_method:max;"+
            "r_dataset_id:" +
           serviceConfig.datasetId +
            ";g_dataset_id:" +
           serviceConfig.datasetId +
            ";b_dataset_id:" +
           serviceConfig.datasetId+
            ";r_from_time:" +
            (obj.from_time || obj.r_from_time)  +
            ";r_to_time:" +
            (obj.to_time || obj.r_to_time)  +
            ";g_from_time:" +
            (obj.from_time || obj.g_from_time) +
            ";g_to_time:" +
            (obj.to_time || obj.g_to_time) +
            ";b_from_time:" +
            (obj.from_time || obj.b_from_time)+
            ";b_to_time:" +
            (obj.to_time || obj.b_to_time) +
            ";r_max:"+serviceConfig.r_max+";g_max:"+serviceConfig.g_max
            +";b_max:"+serviceConfig.b_max+
            ";r_index:"+serviceConfig.r_index+";g_index:"
            +serviceConfig.g_index+";b_index:"+serviceConfig.b_index
            +";r_min:"+serviceConfig.r_min+";g_min:"+serviceConfig.g_min
            +";b_min:"+serviceConfig.b_min
        
           ,

          LEGEND_OPTIONS: "columnHeight:400;height:100",
          //LAYERS: "RIDAM_RGB",
          STYLES: "",
          "X-API-KEY": obj.key,
        },
      }),
      opacity: 1,
      zIndex: 1,
    });
}

function buildCLOUD(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  return new ol.layer.Tile({
      source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
          name: "RDSGrdient",
          layers: "T6S0M0",
          PROJECTION: "EPSG:4326",
          ARGS:"param:"+serviceConfig.param+";from_time:"+obj.from_time+";datasetId:"+serviceConfig.datasetId+
          ";to_time:" +
          obj.to_time ,
      STYLES:serviceConfig.STYLES,      
      LEGEND_OPTIONS: "columnHeight:400;height:100",
      "X-API-KEY": obj.key,
        },
      }),
      opacity: 1,
      zIndex: 1,
    });

}


function buildIndexMapLayer(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  console.log("obejct_key",serviceConfig.param)
  return new ol.layer.Tile({
    source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
        name: "RDSGrdient",
        layers: "T5S1M1",
        PROJECTION: "EPSG:4326",
        ARGS:
            "param:"+serviceConfig.param+";from_time:" +
            obj.from_time +";datasetId:"+serviceConfig.datasetId+
            ";to_time:" +
            obj.to_time,
        STYLES:serviceConfig.STYLES,      
        LEGEND_OPTIONS: "columnHeight:400;height:100",
        "X-API-KEY": obj.key,
        },
    }),
    opacity: 1,
    zIndex: 1,
    });
}

function buildIndexMapLayer_ndsi(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  console.log("obejct_key",serviceConfig.param)
  return new ol.layer.Tile({
    source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
        name: "RDSGrdient",
        layers: "T5S1M1",
        PROJECTION: "EPSG:4326",
        ARGS:
            "param:"+serviceConfig.param+";from_time:" +
            obj.from_time +";datasetId:"+serviceConfig.datasetId+";merge_method:max"+
            ";to_time:" +
            obj.to_time,
        STYLES:serviceConfig.STYLES,      
        LEGEND_OPTIONS: "columnHeight:400;height:100",
        "X-API-KEY": obj.key,
        },
    }),
    opacity: 1,
    zIndex: 1,
    });
}

function buildWRFMapLayer(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  console.log("obejct_key",serviceConfig.param)
  return new ol.layer.Tile({
    source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
        name: "RDSGrdient",
        layers: "T0S0M0",
        PROJECTION: "EPSG:4326",
        ARGS:
             "param:"+serviceConfig.param+";from_time:" +
            obj.from_time +";dataset_id:"+serviceConfig.datasetId+
            ";to_time:" +
            obj.to_time,
          
        STYLES:serviceConfig.STYLES,      
        LEGEND_OPTIONS: "columnHeight:400;height:100",
         "X-API-KEY": obj.key,
        },
    }),
    opacity: 1,
    zIndex: 1,
    });
}


function buildTemperatureForecast(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  console.log("obejct_key",serviceConfig.param)
  return new ol.layer.Tile({
    source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
        name: "RDSGrdient",
        layers: "T0S0M0",
        PROJECTION: "EPSG:4326",
        ARGS:
             "param:"+serviceConfig.param+";from_time:" +
            obj.from_time +";dataset_id:"+"T5S2P4"+
            ";to_time:" +
            obj.to_time,
          
        STYLES:serviceConfig.STYLES,      
        LEGEND_OPTIONS: "columnHeight:400;height:100",
         "X-API-KEY": obj.key,
        },
    }),
    opacity: 1,
    zIndex: 1,
    });
}


function buildMapLayer(obj){
  let serviceConfig = serviceDict[obj.serviceName];
  return new ol.layer.Tile({
    source: new ol.source.TileWMS({
        projection: "EPSG:4326",
        url: serviceConfig.serverUrl, 
        params: {
        name: "RDSGrdient",
        layers: "T0S0M0",
        PROJECTION: "EPSG:4326",
        ARGS:
            "merge_method:max;dataset_id:"+serviceConfig.datasetId+";from_time:" +
            obj.from_time +
            ";to_time:" +
            obj.to_time +
            ";indexes:1",
        styles:serviceConfig.style,      
        LEGEND_OPTIONS: "columnHeight:400;height:100",
        "X-API-KEY": obj.key,
        },
    }),
    opacity: 1,
    zIndex: 1,
    });
}


function s2_ndvi_diff_layer(obj) {
  console.log("s2_ndvi_diff_layer() received:", obj);

  let serviceConfig = serviceDict[obj.serviceName];
  return new ol.layer.Tile({
      source: new ol.source.TileWMS({
          projection: "EPSG:4326",
          url: serviceConfig.serverUrl,
          params: {
              name: "RDSGrdient",
              layers: "T0S0M2",
              PROJECTION: "EPSG:4326",
              ARGS: "merge_method1:min;merge_method2:max;dataset_id1:" +
                  serviceConfig.datasetId +
                  ";from_time1:" + obj.from_time1 +
                  ";to_time1:" + obj.to_time1 +
                  ";dataset_id2:" + serviceConfig.datasetId +
                  ";from_time2:" + obj.from_time2 +
                  ";to_time2:" + obj.to_time2,
              STYLES: serviceConfig.STYLES,      
              LEGEND_OPTIONS: "columnHeight:400;height:100",
              //LAYERS: "RIDAM_RGB",
              //STYLES: "RIDAM_RGB",
              "X-API-KEY": obj.key,
          },
      }),
      opacity: 1,
      zIndex: 1,
  });
}






async function getVAPIChartData(lon, lat, key, param, fromTime, toTime, band) {
  let selectedService = Object.values(serviceDict).find(service => service.param === param);

  if (!selectedService) {
    console.error("Invalid parameter: No matching dataset found for", param);
    return null;
  }

  const isWRF = param === "WRF";
  const isTEMP=param==="TEMPERATURE_FORECAST"
  const isBand_profile=param==="BAND_PROFILE";
  let datasetId = isTEMP ? "T5S2P4" : selectedService.datasetId;
  // Build main data args
  let bodyArgs1 = {
    dataset_id: datasetId,
    from_time: fromTime,
    to_time: toTime,
    param: param,
    lon: lon,
    lat: lat
  };

  if (!isWRF && !isTEMP && !isBand_profile) {
    bodyArgs1.filter_nodata = 'no';
    bodyArgs1.composite = false;
  }

  if (param === "BAND_PROFILE") {
    bodyArgs1.indexes = [parseInt(band)];
  }

  console.log("Request BodyArgs1:", bodyArgs1);

  // Define URLs and request payload for the main service
  const layer1 = isWRF || isTEMP ? "T0S0I0" : "T5S1I1";
  const server1 = isWRF ? "ridam_server" :
                  isTEMP ? "ridam_server" :
                  (param === "NDVI" || param === "NDMI") ? "ridam_server3" : "ridam_server2";
  
  const url1 = `https://vedas.sac.gov.in/vapi/${server1}/info/?X-API-KEY=${encodeURIComponent(key)}`;
  const req1Payload = {
    layer: layer1,
    args: bodyArgs1
  };

  const headers = {
    accept: "application/json",
    "content-type": "application/json"
  };

  // Always send request 1
  const res1 = await fetch(url1, {
    method: "POST",
    headers,
    body: JSON.stringify(req1Payload)
  });

  const result1 = await res1.json();
  let data1 = result1["result"] || [];

  if (param === "BAND_PROFILE") {
    data1 = data1.map(entry => [entry[0], entry[1][0]]);
  }

  let data2 = [];

  // Send second request only if not WRF
  if (!isWRF && !isBand_profile && !isTEMP) {
    const bodyArgs2 = {
      lon: lon,
      lat: lat,
      from_time: fromTime,
      to_time: toTime
    };
    const req2Payload = {
      layer: "T5S1I3",
      args: bodyArgs2
    };
    const url2 = `https://vedas.sac.gov.in/vapi/ridam_server3/info/?X-API-KEY=${encodeURIComponent(key)}`;

    const res2 = await fetch(url2, {
      method: "POST",
      headers,
      body: JSON.stringify(req2Payload)
    });

    const result2 = await res2.json();
    data2 = result2["result"] || [];
  }

  console.log(`Result for ${layer1}:`, data1);
  console.log("Result for T5S1I3:", data2);

  return { data1, data2 };
}



// function processChartData(data, isNDVI = true) {
//   return data.map(([date, value]) => {
//     let timeStamp = new Date(date).getTime();
//     let val = Array.isArray(value) ? value[0] : value; 

 
//     let processedValue = val !== null ? (isNDVI ? val / 250 : val) : null;

//     return [timeStamp, processedValue];
//   });
// }

function processChartData(data) {
  let today= new Date();
  today.setFullYear(today.getFullYear() - 1); 
  let startDate = today.getTime(); 
  return data
    .map(([date, value]) => {
      let timeStamp = new Date(date).getTime();
      let val = Array.isArray(value) ? value[0] : value; 

      let processedValue = val !== null ? val : null;

      return [timeStamp, processedValue];
    })
    .filter(([timeStamp]) => timeStamp >= startDate); 
}


function processPolygonChartData(data) {
  let processedData = data.map((x) => {
    let timeStamp = yyyymmddToUnixTimestamp(x[0]);
    
    let val = x[1];
    return [timeStamp, val];
  });

  return processedData;
}

function yyyymmddToUnixTimestamp(yyyymmdd) {
  const year = parseInt(yyyymmdd.substring(0, 4), 10);
  const month = parseInt(yyyymmdd.substring(4, 6), 10) - 1; // Months are 0-indexed in JavaScript
  const day = parseInt(yyyymmdd.substring(6), 10);

  const date = new Date(year, month, day);
  const unixTimestamp = Math.floor(date.getTime());
  console.log('Unix timstamp is',unixTimestamp);

  return unixTimestamp;
}


async function getVAPIPolygonChartData(coord,key,dataset,fromTime,toTime) {

 

  let datasetMap = {
    "NDVI": "T3S1P1",
    "NDMI": "T3S6P1"
  };
  console.log('Coords are',coord[0]);
  let datasetId = datasetMap[dataset];
  console.log("selected_dataset_id",datasetId)

  let fetchURL = "https://vedas.sac.gov.in/vapi/ridam_server3/info/?X-API-KEY="+key

  let res = await fetch(fetchURL, {
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    referrerPolicy: "strict-origin-when-cross-origin",
    body: JSON.stringify({
      layer: "T5S1I2",
      args: {
        dataset_id: datasetId,
        filter_nodata: "no",
        polygon: coord,
        indexes: [1],
        from_time: fromTime,
        to_time: toTime,
        interval: 10,
        merge_method: 'max',
      },
    }),
    method: "POST",
  });

  let result = await res.json();
  let data = result["result"];

  return data;
}
