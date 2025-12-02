// -----------------------------------------------------------------------------
// 0. BASIC SETTINGS
// -----------------------------------------------------------------------------
var aoi = ee.FeatureCollection("projects/terraquauav/assets/czo");

// Safe geometry getter
var geom = (aoi.geometry ? aoi.geometry() : aoi).simplify(100);

Map.centerObject(geom, 10);
Map.addLayer(geom, {color:'red'}, 'kanpur_shapefile');

var start_date = '2025-01-01';
var end_date   = '2025-06-30';
var RESOLUTION = 30;

// Default temperatures for CWSI if reduction fails (in Kelvin)
var DEFAULT_T_COLD = 300;
var DEFAULT_T_HOT  = 320;

// 🟢 ADD: Specific location point
var specificLocation = ee.Geometry.Point([80.15383585, 26.56286922]);
Map.addLayer(specificLocation, {color: 'yellow', pointSize: 10}, 'Specific Location');

// ------------------------------------------------------------------
// 1. MERGE L8 & L9 → KEEP ONLY CLOUD_COVER < 20 FROM THE START
// ------------------------------------------------------------------
var merged = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
               .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
               .filterBounds(aoi)
               .filterDate(start_date, end_date)
               .filter(ee.Filter.lt('CLOUD_COVER', 50)); // Start with less aggressive filter

var withDate = merged.map(function(img) {
  return img.set('date_str', img.date().format('YYYY-MM-dd'));
});

var uniqueDates = withDate.aggregate_array('date_str').distinct();

var onePerDay = ee.ImageCollection(
  uniqueDates.map(function(d) {
    var dayCol = withDate.filterDate(d, ee.Date(d).advance(1, 'day'));
    var best = dayCol.sort('CLOUD_COVER').first();
    return best.set('selected_date', d);
  })
).sort('system:time_start')
.filter(ee.Filter.lt('CLOUD_COVER', 20)); // 🟢 FIX 1: Apply strict cloud filter after selecting best image

print('Number of unique daily images (<20% cloud):', onePerDay.size());
print('Selected Dates (YYYY-MM-DD):', onePerDay.aggregate_array('selected_date'));

// -----------------------------------------------------------------------------
// 2. EXTERNAL DATA (ETo & Precipitation)
// -----------------------------------------------------------------------------
var eto = ee.ImageCollection("MODIS/061/MOD16A2")
  .filterBounds(aoi)
  .filterDate(start_date, end_date)
  .select('PET') // PET in 0.1 mm / 8 days
  .map(function(img) {
    // Convert to mm/day (0.1 mm / 8 days → mm/day)
    var etoDaily = img.multiply(0.1).divide(8).rename('eto');
    // Mosaic overlapping MODIS tiles to avoid edge gaps
    var date = ee.Date(img.get('system:time_start'));
    var mosaicked = ee.ImageCollection("MODIS/061/MOD16A2")
      .filterDate(date, date.advance(1, 'day'))
      .select('PET')
      .map(function(i) {
        return i.multiply(0.1).divide(8).rename('eto');
      })
      .mosaic();
    // Reproject and clip for consistent coverage
    mosaicked = mosaicked
      .resample('bilinear')
      .reproject({ crs: 'EPSG:4326', scale: 1000 })
      .clip(aoi);
    return mosaicked.copyProperties(img, img.propertyNames());
  });


var precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .select('precipitation');

// -----------------------------------------------------------------------------
// 3. HELPERS
// -----------------------------------------------------------------------------
function applyScaleFactors(img) {
  var opt = img.select('SR_B.*').multiply(0.0000275).add(-0.2);
  var thm = img.select('ST_B.*').multiply(0.00341802).add(149.0);
  return img.addBands(opt, null, true).addBands(thm, null, true);
}

function cloudMask(img) {
  var qa = img.select('QA_PIXEL').bitwiseAnd(parseInt('11111',2)).eq(0);
  var sat = img.select('QA_RADSAT').eq(0);
  // 🟢 FIX 2: Relaxed Cloud Masking. Only mask saturation and the main QA bits.
  // For most analyses, the full QA_PIXEL mask is too aggressive.
  // We rely more on the CLOUD_COVER metadata filter in step 1.
  return img.updateMask(qa).updateMask(sat);
}

// -----------------------------------------------------------------------------
// 4. SEASON-WIDE NDVI MIN / MAX (20th & 90th percentiles)
// ... (NDVI calculations are fine) ...
var ndviColl = onePerDay
  .map(applyScaleFactors)
  .map(cloudMask)
  .map(function(i){
    return i.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
  });

var ndviPerc = ndviColl.reduce(ee.Reducer.percentile([20,90]))
                      .rename(['NDVI_p20','NDVI_p90']);

var ndviStats = ndviPerc.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: geom,
  scale: RESOLUTION,
  maxPixels: 1e9
});

var ndvi_min = ee.Number(ndviStats.get('NDVI_p20')).clamp(0.05,0.2);
var ndvi_max = ee.Number(ndviStats.get('NDVI_p90')).clamp(0.6,0.9);

////print('NDVI_MIN (20th %):', ndvi_min);
////print('NDVI_MAX (90th %):', ndvi_max);

var ndviMinImg = ee.Image.constant(ndvi_min);
var ndviMaxImg = ee.Image.constant(ndvi_max);

// -----------------------------------------------------------------------------
// 5. PER-IMAGE INDICES (SAVI, CWSI, Kc, CWR, Irrigation Need …)
// -----------------------------------------------------------------------------
function addAdvancedIndices(img) {
  // Ensure required bands exist (Your existing check is good, but shortened here for brevity)
  var bands = img.bandNames();
  var required = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ST_B10'];
  var allOK = required.every(function(b) { return bands.contains(b); });

  // If image lacks bands, return an empty image (or a minimal placeholder)
  if (!allOK) return ee.Image.constant(-999).rename(['NDVI']).copyProperties(img, ['system:time_start']); 


    var NIR  = img.select('SR_B5');
    var RED  = img.select('SR_B4');
    var BLUE = img.select('SR_B2');
    var LST  = img.select('ST_B10').rename('LST');
    var ndvi = img.normalizedDifference(['SR_B5','SR_B4']).rename('NDVI');
    var date = img.date();

    // === CWSI ===
    var hotmask  = ndvi.lt(ndvi_min);
    var coldmask = ndvi.gt(ndvi_max);

    var T_cold_raw = LST.updateMask(coldmask).reduceRegion({
      reducer: ee.Reducer.percentile([5]),
      geometry: geom,
      scale: RESOLUTION, // 🟢 Set to RESOLUTION to match NDVI/LST scale
      maxPixels: 1e9,
      bestEffort: true
    }).get('LST');

    var T_hot_raw = LST.updateMask(hotmask).reduceRegion({
      reducer: ee.Reducer.percentile([90]),
      geometry: geom,
      scale: RESOLUTION, // 🟢 Set to RESOLUTION to match NDVI/LST scale
      maxPixels: 1e9,
      bestEffort: true
    }).get('LST');

    // 🟢 FIX 3: Use ee.Number for safer null substitution (JavaScript best practice)
    var T_cold_num = ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(T_cold_raw, null), 
      ee.Number(DEFAULT_T_COLD), 
      T_cold_raw
    ));
    var T_hot_num  = ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(T_hot_raw, null), 
      ee.Number(DEFAULT_T_HOT), 
      T_hot_raw
    ));

    // 🟢 FIX 4: Convert T_cold/T_hot numbers to images here, not earlier, for clarity
    var T_cold = ee.Image.constant(T_cold_num);
    var T_hot  = ee.Image.constant(T_hot_num);

    // CWSI now uses guaranteed non-null images for T_cold and T_hot
    var cwsi = LST.subtract(T_cold).divide(T_hot.subtract(T_cold)).clamp(0,1).rename('CWSI');

    // === SAVI & others ===
    var S = ndviMaxImg.subtract(ndvi).divide(ndviMaxImg.subtract(ndviMinImg)).clamp(0,1);
    var L = S.add(1).multiply(0.5);
    var savi = NIR.subtract(RED).divide(NIR.add(RED).add(L)).multiply(L.add(1)).rename('SAVI');

    var fv = ndvi.subtract(ndviMinImg).divide(ndviMaxImg.subtract(ndviMinImg)).clamp(0,1).pow(2).rename('FV');
    var em = fv.multiply(0.004).add(0.986).rename('Emissivity');
    var evi = img.expression(
      '2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)',
      {NIR:NIR, RED:RED, BLUE:BLUE}
    ).rename('EVI');
    var lai = evi.expression('3.618 * EVI - 0.118', {EVI:evi}).rename('LAI');


    // === Kc, CWR, Irrigation ===
    var kc = (savi.multiply(1.796634562654909).add(-0.2869936095897908)).rename('Kc');

    // Get the closest MODIS ETo image to the Landsat date
    var etoSorted = eto.sort('system:time_start'); // ensure chronological order

    var etoClosest = etoSorted.sort('system:time_start').map(function(img) {
      var diff = img.date().difference(date, 'day').abs();
      return img.set('diff', diff);
      }).sort('diff').first();

    // Handle case where no MODIS data exist in range
    var etoImg = ee.Image(
          ee.Algorithms.If(
           ee.Algorithms.IsEqual(etoClosest, null),
           ee.Image.constant(0).rename('eto'),
           ee.Image(etoClosest).select('eto')
       )
      );


    var pFirst = precip.filterDate(date, date.advance(1, 'day')).first();
    var pImg = ee.Image(
      ee.Algorithms.If(
        ee.Algorithms.IsEqual(pFirst, null),
        ee.Image.constant(0.1).rename('precipitation'),
        ee.Image(pFirst).select('precipitation')
      )
    );
/// Irrigation efficiency is assumed to be 40%
    var cwr    = etoImg.multiply(kc).rename('CWR');
    var irrNeed = cwr.subtract(pImg).divide(0.6).rename('Irrigation_Need');

    // 🟢 ADD ETo band to the output
    return img.addBands([ndvi, LST, savi, fv, em, evi, lai, cwsi, kc, pImg.rename('precipitation'), etoImg.rename('eto'), cwr, irrNeed])
              .clip(geom)
              .copyProperties(img, ['system:time_start']);
}


// -----------------------------------------------------------------------------
// 6. FINAL TIME-SERIES (only <20% cloud images)
// -----------------------------------------------------------------------------
var finalTS = onePerDay
  .map(applyScaleFactors)
  .map(cloudMask)
  .map(addAdvancedIndices);

print('Final indexed images:', finalTS.size());

// =============================================================================
// EXPORT ALL PARAMETERS FOR A SINGLE DATE - WITH FOLDER STRUCTURE
// =============================================================================

// 🗓️ SET YOUR TARGET DATE HERE (must match one of the dates in your finalTS)
var exportDate = '2025-06-09';  // Change this to any date from your time series

// Get the image for the selected date
var exportImg = finalTS.filter(ee.Filter.eq('selected_date', exportDate)).first();
exportImg = exportImg.toFloat();

// Check if image exists
print('Exporting data for date:', exportDate);
print('Image exists:', ee.Algorithms.IsEqual(exportImg, null) ? 'NO' : 'YES');

// Define all bands you want to export
var exportBands = [
  'NDVI',
  'SAVI', 
  'CWSI',
  'Kc',
  'eto',
  'precipitation',
  'CWR',
  'Irrigation_Need',
  'LST',
  'LAI',
  'EVI',
  'Emissivity'
];

// =============================================================================
// FOLDER STRUCTURE SETUP
// =============================================================================
// Main folder: Kanpur_CZO_Exports/2025_04_14/
// Subfolders: /AllBands, /Individual, /Grouped, /Summary

var mainFolder = 'Kanpur_CZO_Exports';
var dateFolder = mainFolder + '/' + exportDate.replace(/-/g, '_');

// =============================================================================
// METHOD 1: SINGLE MULTIBAND FILE (RECOMMENDED - 1 TASK ONLY!)
// =============================================================================

Export.image.toDrive({
  image: exportImg.select(exportBands).clip(geom),
  description: 'AllBands_' + exportDate.replace(/-/g, '_'),
  folder: dateFolder + '/AllBands',  // Creates subfolder
  fileNamePrefix: 'AllParams_' + exportDate,
  region: geom,
  crs: 'EPSG:32644',
  scale: 30,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('✅ Task 1: Multiband file (all 13 parameters in 1 file)');



// //  7. --- Generate time series features for multiple indices ---
// var indices = ['NDVI', 'SAVI', 'CWSI', 'Kc' ,'CWR', 'Irrigation_Need', 'eto']; // 🟢 ADD 'eto' to indices list

// indices.forEach(function(indexName) {
//   var ts = finalTS.map(function(img) {
//     var meanDict = img.reduceRegion({
//       reducer: ee.Reducer.percentile([75]),
//       geometry: geom,
//       scale: 250,       // reduce memory load
//       crs:'EPSG:32644',
//       maxPixels: 1e7,
//       bestEffort: true
//     });
//     return ee.Feature(null, meanDict.set('date', img.date().format('YYYY-MM-dd')));
//   });

//   var chart = ui.Chart.feature.byFeature(ts, 'date', [indexName])
//     .setChartType('LineChart')
//     .setOptions({
//       title: indexName + ' Time Series (Kanpur AOI)',
//       hAxis: {title: 'Date'},
//       vAxis: {title: indexName},
//       lineWidth: 2,
//       pointSize: 3
//     });
//   print(chart);
// });


// -----------------------------------------------------------------------------
// 8. VISUALISATION – 100 % safe
// -----------------------------------------------------------------------------
var showDate = '2025-01-24';   // 🗓️ Change this to your desired date (from printed list)
var vizImg = ee.Image(finalTS.filter(ee.Filter.eq('selected_date', showDate)).first())
             .resample('bilinear')
              .reproject({crs: 'EPSG:32644', scale: 120});


if (vizImg) {
  var imgDate = ee.Date(vizImg.get('system:time_start')).format('YYYY-MM-dd');
  print('Showing image from:', imgDate);

  Map.addLayer(vizImg.select(['SR_B5','SR_B4','SR_B3']),
               {min:0, max:0.3, gamma:1.4},
               'True-Color (NIR-Red-Green)');

  var ndviMean = vizImg.select('NDVI')
                       .reduceRegion({
                         reducer: ee.Reducer.mean(),
                         geometry: geom,
                         scale: 500,
                         maxPixels: 1e5,
                         bestEffort: true
                       }).get('NDVI');

  if (ndviMean !== null) {
    Map.addLayer(vizImg.select('Kc'),   {min:0, max:1.5,   palette:['blue','cyan','white','yellow','red']}, 'Kc');
    
    Map.addLayer(vizImg.select('SAVI'),   {min:0, max:1,   palette:['red','orange','yellow','green']}, 'SAVI');
    Map.addLayer(vizImg.select('CWSI'),   {min:0, max:1,   palette:['blue','cyan','white','yellow','red']}, 'CWSI');
    Map.addLayer(vizImg.select('eto'),   {min:0, max:10,   palette:['blue','cyan','white','yellow','red']}, 'ETo'); // 🟢 ADD ETo visualization
    Map.addLayer(vizImg.select('precipitation'),   {min:0, max:1.5,   palette:['blue','cyan','white','yellow','red']}, 'precipitation');
    Map.addLayer(vizImg.select('CWR').clamp(0,10),
                 {min:0, max:10, palette:['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']},
                 'CWR (mm d⁻¹)');
    Map.addLayer(vizImg.select('Irrigation_Need').clamp(0,30),
                 {min:0, max:15, palette:['#f0f9e8','#bae4bc','#7bccc4','#43a2ca','#0868ac']},
                 'Irrigation Need (mm d⁻¹)');
    Map.addLayer(vizImg.select('LST'), {min:280, max:330, palette:['blue','white','red']}, 'LST (K)');
  } else {
    print('Image fully masked – only true-color shown.');
  }

  Map.centerObject(aoi, 12);
} else {
  print('No valid images found with CLOUD_COVER < 20.');
  print('Try:');
  print('  • Increasing cloud limit to 30 or 50');
  print('  • Extending date range');
}

// // 9. EXPORT SELECTED LAYERS (for visualization date) - INCLUDING ETo
// if (vizImg) {
//   var exportBands = [
//     'NDVI','NDMI','SAVI','CWSI','eto','precipitation','CWR','Irrigation_Need','LST','Kc'
//   ];

//   // Use the original image without reprojection for export
//   var exportImg = ee.Image(finalTS.filter(ee.Filter.eq('selected_date', showDate)).first());
  
//   exportBands.forEach(function(bandName) {
//     Export.image.toDrive({
//       image: exportImg.select(bandName),
//       description: 'Export_' + bandName + '_' + showDate,
//       folder: '06-apr',
//       fileNamePrefix: bandName + '_' + showDate,
//       region: geom,
//       crs: 'EPSG:32644',  // Your UTM zone
//       scale: 30,  // ✅ Force 30m resolution
//       maxPixels: 1e13,
//       fileFormat: 'GeoTIFF'
//     });
//   });
// }

// // -----------------------------------------------------------------------------
// // 10. MEAN CWR AND ETo PER DATE → CSV EXPORT
// // -----------------------------------------------------------------------------
// var meanParamsPerDate = finalTS.map(function(img) {
//   var meanDict = img.select(['CWR', 'eto']).reduceRegion({ // 🟢 ADD 'eto' to reduction
//     reducer: ee.Reducer.mean(),
//     geometry: geom,
//     scale: RESOLUTION,
//     maxPixels: 1e9
//   });
//   var date = img.date().format('YYYY-MM-dd');
//   return ee.Feature(null, {
//     'date': date,
//     'CWR_mean_mm_per_day': meanDict.get('CWR'),
//     'ETo_mean_mm_per_day': meanDict.get('eto') // 🟢 ADD ETo to CSV export
//   });
// });

// print('Mean CWR and ETo per Date (AOI):', meanParamsPerDate);

// Export.table.toDrive({
//   collection: meanParamsPerDate,
//   description: 'Mean_CWR_ETo_per_Date_Kanpur_AOI',
//   folder: 'GEE_Exports',
//   fileFormat: 'CSV'
// });

// -----------------------------------------------------------------------------
// 10. CWR AND ETo VALUES AT SPECIFIC LOCATION PER DATE → CSV EXPORT
// -----------------------------------------------------------------------------
// var locationParamsPerDate = finalTS.map(function(img) {
//   // 🟢 MODIFIED: Extract values at specific location instead of mean over AOI
//   var pointDict = img.select(['CWR', 'eto', 'NDVI', 'SAVI', 'Kc', 'precipitation', 'Irrigation_Need']).reduceRegion({
//     reducer: ee.Reducer.first(),
//     geometry: specificLocation,
//     scale: 30,  // Use Landsat resolution for precise location extraction
//     maxPixels: 1e9
//   });
//   var date = img.date().format('YYYY-MM-dd');
//   return ee.Feature(null, {
//     'date': date,
//     'CWR_mm_per_day': pointDict.get('CWR'),
//     'ETo_mm_per_day': pointDict.get('eto'),
//     'NDVI': pointDict.get('NDVI'),
//     'SAVI': pointDict.get('SAVI'),
//     'Kc': pointDict.get('Kc'),
//     'precipitation_mm': pointDict.get('precipitation'),
//     'Irrigation_Need_mm_per_day': pointDict.get('Irrigation_Need')
//   });
// });

// print('CWR and other parameters at specific location per Date:', locationParamsPerDate);

// Export.table.toDrive({
//   collection: locationParamsPerDate,
//   description: 'Location_Parameters_26_562_80_153',
//   folder: 'GEE_Exports',
//   fileFormat: 'CSV'
// });

// // -----------------------------------------------------------------------------
// // 11. ADDITIONAL: PRINT VALUES FOR SPECIFIC LOCATION
// // -----------------------------------------------------------------------------
// print('Specific Location Coordinates:', specificLocation);
// print('Sample values for the first 5 dates:');

// // Get first 5 images to show sample values
// var sampleImages = finalTS.limit(5);
// sampleImages.getInfo(function(images) {
//   images.features.forEach(function(image, index) {
//     var date = image.properties.selected_date || image.properties.date_str;
//     print('Date:', date, 
//           '- CWR:', image.properties.CWR, 
//           '- ETo:', image.properties.eto,
//           '- NDVI:', image.properties.NDVI);
//   });
// });

// =============================================================================
// EXPORT TEMPORALLY ALIGNED DATA AS ASSETS - USING ORIGINAL DATES
// =============================================================================

// Get the actual dates from your final time series collection
var alignedDates = finalTS.aggregate_array('system:time_start').distinct();
var dateList = alignedDates.getInfo(); // Get the actual dates that have data

print('Temporally aligned dates for export:', dateList.length);
print('Date range:', start_date, 'to', end_date);
print('Actual dates with data:', alignedDates);

// Create temporally aligned collections USING ONLY DATES THAT EXIST
var alignedCWSI = ee.ImageCollection.fromImages(
  dateList.map(function(dateMillis) {
    var date = ee.Date(dateMillis);
    var dateStr = date.format('YYYY-MM-dd');
    
    // Get CWSI for this exact date (no date range filtering)
    var cwsiImg = finalTS.select('CWSI')
      .filter(ee.Filter.eq('system:time_start', dateMillis))
      .first()
      .rename('CWSI')
      .set('system:time_start', dateMillis)
      .set('date', dateStr)
      .set('has_data', true);
    
    return cwsiImg;
  })
);

var alignedIrrigation = ee.ImageCollection.fromImages(
  dateList.map(function(dateMillis) {
    var date = ee.Date(dateMillis);
    var dateStr = date.format('YYYY-MM-dd');
    
    // Get Irrigation Need for this exact date
    var irrImg = finalTS.select('Irrigation_Need')
      .filter(ee.Filter.eq('system:time_start', dateMillis))
      .first()
      .rename('Irrigation_Need')
      .set('system:time_start', dateMillis)
      .set('date', dateStr)
      .set('has_data', true);
    
    return irrImg;
  })
);

var alignedPrecipitation = ee.ImageCollection.fromImages(
  dateList.map(function(dateMillis) {
    var date = ee.Date(dateMillis);
    var dateStr = date.format('YYYY-MM-dd');
    
    // Get Precipitation for this exact date
    var rainImg = precip
      .filterDate(date, date.advance(1, 'day'))
      .first();
    
    // Handle case where no precipitation data exists for this date
    var finalRainImg = ee.Image(ee.Algorithms.If(
      rainImg,
      rainImg.select('precipitation'),
      ee.Image.constant(0).rename('precipitation')
    )).rename('precipitation')
      .set('system:time_start', dateMillis)
      .set('date', dateStr)
      .set('has_data', ee.Algorithms.If(rainImg, true, false));
    
    return finalRainImg;
  })
);

// Print verification
print('Aligned CWSI collection:', alignedCWSI.size());
print('Aligned Irrigation collection:', alignedIrrigation.size());
print('Aligned Precipitation collection:', alignedPrecipitation.size());

// Check data availability
var cwsiDataCount = alignedCWSI.filter(ee.Filter.eq('has_data', true)).size();
var irrDataCount = alignedIrrigation.filter(ee.Filter.eq('has_data', true)).size();
var rainDataCount = alignedPrecipitation.filter(ee.Filter.eq('has_data', true)).size();

print('Data availability:');
print('- CWSI dates with data:', cwsiDataCount, '/', alignedCWSI.size());
print('- Irrigation dates with data:', irrDataCount, '/', alignedIrrigation.size());
print('- Precipitation dates with data:', rainDataCount, '/', alignedPrecipitation.size());



// Sort the collections by date FIRST to ensure chronological order
var cwsiSorted = alignedCWSI.sort('system:time_start');
var irrSorted = alignedIrrigation.sort('system:time_start');
var precipSorted = alignedPrecipitation.sort('system:time_start');

// Get sorted dates for band naming
var sortedDates = cwsiSorted.aggregate_array('date');

print('Dates in chronological order:', sortedDates);
print('First date:', sortedDates.get(0));
print('Last date:', sortedDates.get(sortedDates.size().subtract(1)));

// Create multiband stacks with dates in ascending order
var cwsiStack = cwsiSorted.toBands().rename(
  sortedDates.map(function(d) {
    return ee.String('CWSI_').cat(ee.String(d).replace('-', '_', 'g'));
  })
);

var irrStack = irrSorted.toBands().rename(
  sortedDates.map(function(d) {
    return ee.String('Irrigation_').cat(ee.String(d).replace('-', '_', 'g'));
  })
);

var precipStack = precipSorted.toBands().rename(
  sortedDates.map(function(d) {
    return ee.String('Precipitation_').cat(ee.String(d).replace('-', '_', 'g'));
  })
);

// Verify band order
print('CWSI bands (first 5):', cwsiStack.bandNames().slice(0, 5));
print('CWSI bands (last 5):', cwsiStack.bandNames().slice(-5));
print('Total bands:', cwsiStack.bandNames().size());

// Export CWSI Stack
Export.image.toAsset({
  image: cwsiStack.clip(geom),
  description: 'CWSI_MultiDate_Stack_Sorted',
  assetId: 'projects/terraquauav/assets/CWSI_MultiDate_Stack_Sorted',
  region: geom,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:32644'
});

// Export Irrigation Stack
Export.image.toAsset({
  image: irrStack.clip(geom),
  description: 'Irrigation_MultiDate_Stack_Sorted',
  assetId: 'projects/terraquauav/assets/Irrigation_MultiDate_Stack_Sorted',
  region: geom,
  scale: 30,
  maxPixels: 1e13,
  crs: 'EPSG:32644'
});

// Export Precipitation Stack
Export.image.toAsset({
  image: precipStack.clip(geom),
  description: 'Precipitation_MultiDate_Stack_Sorted',
  assetId: 'projects/terraquauav/assets/Precipitation_MultiDate_Stack_Sorted',
  region: geom,
  scale: 5000,  // CHIRPS native resolution
  maxPixels: 1e13,
  crs: 'EPSG:32644'
});

print('✅ MULTIBAND STACKS CREATED WITH DATES IN ASCENDING ORDER');
print('📁 Assets will be saved as:');
print('   - projects/terraquauav/assets/CWSI_MultiDate_Stack_Sorted');
print('   - projects/terraquauav/assets/Irrigation_MultiDate_Stack_Sorted');
print('   - projects/terraquauav/assets/Precipitation_MultiDate_Stack_Sorted');
print('');
print('🎯 Band naming format: PARAMETER_YYYY_MM_DD');
print('   Example: CWSI_2025_01_15, CWSI_2025_01_20, etc.');
print('');
print('📊 To use in another script:');
print('   var cwsi = ee.Image("projects/terraquauav/assets/CWSI_MultiDate_Stack_Sorted");');
print('   var jan15 = cwsi.select("CWSI_2025_01_15");');
print('   var allDates = cwsi.bandNames(); // Get list of all dates');