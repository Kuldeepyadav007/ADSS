"""
SMART IRRIGATION SYSTEM - COMPLETE FLASK BACKEND
Landsat-based daily data + Sentinel-1 soil moisture + Smart irrigation calendar
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import ee
from datetime import timedelta, datetime
import logging
import sys
import traceback
import numpy as np
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user

from report_generator import generate_report_response
from models import db, User, Field

# =============================================================================
# EARTH ENGINE INITIALIZATION
# =============================================================================

service_account = 'ansh-347@karshizest.iam.gserviceaccount.com'
key_file = 'karshizest-a7b3c3acdeee.json'

credentials = ee.ServiceAccountCredentials(service_account, key_file)
ee.Initialize(credentials)

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.permanent_session_lifetime = timedelta(days=7)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///krishizest.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Login Manager Configuration
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables
with app.app_context():
    db.create_all()

# Configure logging
root_logger = logging.getLogger()
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
root_logger.setLevel(logging.DEBUG)

app.config['DEBUG'] = True
app.config['ENV'] = 'development'

logger = logging.getLogger('smart_irrigation_logger')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# =============================================================================
# LANDSAT PROCESSING FUNCTIONS
# =============================================================================

def apply_scale_factors(img):
    """Apply scale factors to Landsat surface reflectance and thermal bands"""
    opt = img.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thm = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opt, None, True).addBands(thm, None, True)

def cloud_mask_landsat(img):
    """Cloud masking for Landsat 8/9"""
    qa = img.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    sat = img.select('QA_RADSAT').eq(0)
    return img.updateMask(qa).updateMask(sat)

def get_landsat_collection(aoi, start_date, end_date, cloud_filter=20):
    """Get merged Landsat 8/9 collection with one image per day"""
    merged = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
               .merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')) \
               .filterBounds(aoi) \
               .filterDate(start_date, end_date) \
               .filter(ee.Filter.lt('CLOUD_COVER', 50))
    
    # Add date string for grouping
    with_date = merged.map(lambda img: img.set('date_str', img.date().format('YYYY-MM-dd')))
    
    # Get unique dates and select best image per day
    unique_dates = with_date.aggregate_array('date_str').distinct()
    
    one_per_day = ee.ImageCollection(
        unique_dates.map(lambda d: 
            with_date.filterDate(d, ee.Date(d).advance(1, 'day'))
                     .sort('CLOUD_COVER')
                     .first()
                     .set('selected_date', d)
        )
    ).sort('system:time_start').filter(ee.Filter.lt('CLOUD_COVER', cloud_filter))
    
    return one_per_day

def calculate_advanced_indices(img, aoi, ndvi_min_img, ndvi_max_img):
    """Calculate vegetation indices, CWSI, Kc, and other parameters"""
    required_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ST_B10']
    bands = img.bandNames()
    
    nir = img.select('SR_B5')
    red = img.select('SR_B4')
    blue = img.select('SR_B2')
    swir = img.select('SR_B6')
    lst = img.select('ST_B10').rename('LST')
    
    # NDVI
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    
    # CWSI Calculation
    hot_mask = ndvi.lt(ndvi_min_img)
    cold_mask = ndvi.gt(ndvi_max_img)
    
    default_t_cold = 300
    default_t_hot = 320
    
    t_cold_raw = lst.updateMask(cold_mask).reduceRegion(
        reducer=ee.Reducer.percentile([10]),
        geometry=aoi,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    ).get('LST')
    
    t_hot_raw = lst.updateMask(hot_mask).reduceRegion(
        reducer=ee.Reducer.percentile([85]),
        geometry=aoi,
        scale=30,
        maxPixels=1e9,
        bestEffort=True
    ).get('LST')
    
    t_cold = ee.Image.constant(ee.Algorithms.If(
        ee.Algorithms.IsEqual(t_cold_raw, None),
        default_t_cold,
        t_cold_raw
    ))
    
    t_hot = ee.Image.constant(ee.Algorithms.If(
        ee.Algorithms.IsEqual(t_hot_raw, None),
        default_t_hot,
        t_hot_raw
    ))
    
    cwsi = lst.subtract(t_cold).divide(t_hot.subtract(t_cold)).clamp(0, 1).rename('CWSI')
    
    # SAVI calculation
    s = ndvi_max_img.subtract(ndvi).divide(ndvi_max_img.subtract(ndvi_min_img)).clamp(0, 1)
    l_factor = s.add(1).multiply(0.5)
    savi = nir.subtract(red).divide(nir.add(red).add(l_factor)).multiply(l_factor.add(1)).rename('SAVI')
    
    # Fractional vegetation
    fv = ndvi.subtract(ndvi_min_img).divide(ndvi_max_img.subtract(ndvi_min_img)).clamp(0, 1).pow(2).rename('FV')
    
    # Emissivity
    em = fv.multiply(0.004).add(0.986).rename('Emissivity')
    
    # EVI (Enhanced Vegetation Index)
    evi = img.expression(
        '2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)',
        {'NIR': nir, 'RED': red, 'BLUE': blue}
    ).rename('EVI')
    
    # LAI (Leaf Area Index)
    lai = evi.expression('3.618 * EVI - 0.118', {'EVI': evi}).rename('LAI')
    

    
    # Crop coefficient (Kc)
    kc = savi.multiply(1.796634562654909).add(-0.2869936095897908).rename('Kc')
    
    return img.addBands([ndvi, lst, savi, fv, em, evi, lai, cwsi, kc])



# =============================================================================
# SENTINEL-1 SOIL MOISTURE (OPTIMIZED)
# =============================================================================

def get_sentinel1_soil_moisture_simple(aoi, start_date, end_date):
    """Simplified Sentinel-1 processing for daily data"""
    try:
        sen1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
                 .filterDate(start_date, end_date) \
                 .filterBounds(aoi) \
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                 .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                 .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                 .select('VV')
        
        def process_simple(img):
            sigma = ee.Image(10).pow(img.divide(10))
            filtered = sigma.focalMean(30, 'square', 'meters')
            return filtered.copyProperties(img, img.propertyNames())
        
        processed = sen1.map(process_simple)
        
        col_min = processed.min()
        col_max = processed.max()
        
        normalized = processed.map(lambda img: 
            img.subtract(col_min)
               .divide(col_max.subtract(col_min))
               .multiply(100)
               .rename('soil_moisture')
               .copyProperties(img, img.propertyNames())
        )
        
        return normalized
        
    except Exception as e:
        logger.warning(f"Sentinel-1 simple processing failed: {e}")
        return None

def get_sentinel1_soil_moisture_advanced(aoi, start_date, end_date):
    """Advanced Sentinel-1 processing for visualization"""
    try:
        dem = ee.Image('USGS/SRTMGL1_003').clip(aoi)
        terrain = ee.Terrain.products(dem)
        slope = terrain.select('slope')
        
        sen1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
                 .filterDate(start_date, end_date) \
                 .filterBounds(aoi) \
                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                 .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                 .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
                 .select(['VV', 'angle'])
        
        def process_advanced(img):
            incidence_angle = img.select('angle')
            vv = img.select('VV')
            
            sigma = ee.Image(10).pow(vv.divide(10))
            
            ref_angle = 39
            angle_correction = ee.Image.constant(ref_angle).divide(incidence_angle).cos().pow(1.2)
            
            slope_rad = slope.multiply(np.pi / 180)
            incidence_rad = incidence_angle.multiply(np.pi / 180)
            terrain_correction = incidence_rad.cos().divide(slope_rad.cos())
            
            sigma_corrected = sigma.multiply(angle_correction) \
                                  .multiply(terrain_correction) \
                                  .clamp(0.0001, 1)
            
            filtered = sigma_corrected.focalMean(30, 'square', 'meters')
            
            return filtered.copyProperties(img, img.propertyNames())
        
        processed = sen1.map(process_advanced)
        
        water_mask = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").select('label') \
                       .filterDate(start_date, end_date) \
                       .filterBounds(aoi).mode().eq(0).Not()
        
        urban_mask = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").select('label') \
                       .filterDate(start_date, end_date) \
                       .filterBounds(aoi).mode().eq(6).Not()
        
        col_min = processed.min()
        col_max = processed.max()
        
        normalized = processed.map(lambda img: 
            img.subtract(col_min)
               .divide(col_max.subtract(col_min))
               .multiply(water_mask)
               .multiply(urban_mask)
               .multiply(100)
               .rename('soil_moisture')
               .copyProperties(img, img.propertyNames())
        )
        
        return normalized
        
    except Exception as e:
        logger.error(f"Sentinel-1 advanced processing failed: {e}")
        return None

# =============================================================================
# FLASK ROUTES
# =============================================================================

@app.route('/')
@login_required
def index():
    """Dashboard landing page"""
    return render_template('index.html', user=current_user)

@app.route('/app2')
@login_required
def app2():
    """Main map application"""
    return render_template('app2.html', user=current_user)

@app.route('/create_field')
@login_required
def create_field():
    """Field creation page with drawing tools"""
    return render_template('create_field.html', user=current_user)

# =============================================================================
# AUTH ROUTES
# =============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user:
            flash('Email already exists')
            return redirect(url_for('register'))
            
        new_user = User(
            email=email,
            name=name,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        login_user(new_user)
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# =============================================================================
# FIELD API ROUTES
# =============================================================================

@app.route('/api/save_field', methods=['POST'])
@login_required
def save_field():
    try:
        data = request.get_json()
        
        new_field = Field(
            user_id=current_user.id,
            name=data['name'],
            field_type=data['type'],
            area_hectares=data['areaHectares'],
            area_acres=data['areaAcres'],
            geometry_json=json.dumps(data['geometry'])
        )
        
        db.session.add(new_field)
        db.session.commit()
        
        return jsonify({'success': True, 'field_id': new_field.id})
        
    except Exception as e:
        logger.error(f"Error saving field: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/fields')
@login_required
def fields_page():
    """Fields management page"""
    return render_template('fields.html', user=current_user)

@app.route('/api/get_fields', methods=['GET'])
@login_required
def get_fields():
    try:
        fields = Field.query.filter_by(user_id=current_user.id).order_by(Field.created_at.desc()).all()
        fields_data = []
        for f in fields:
            try:
                fields_data.append(f.to_dict())
            except Exception as e:
                logger.error(f"Error serializing field {f.id}: {e}")
                continue
        return jsonify(fields_data)
    except Exception as e:
        logger.error(f"Error fetching fields: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_field/<int:field_id>', methods=['GET'])
@login_required
def get_field(field_id):
    try:
        field = Field.query.get_or_404(field_id)
        if field.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        return jsonify(field.to_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def _chrome_probe():
    return ('', 204)

@app.route('/get_daily_data', methods=['POST'])
def get_daily_data():
    """Enhanced daily data endpoint using Landsat with Sentinel-1 soil moisture"""
    try:
        data = request.get_json()
        user_aoi = data['aoi']
        coords = user_aoi['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)

        start_date = data['start_date']
        end_date = data['end_date']
        
        # Validate date range
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        diff_days = end.difference(start, 'day').getInfo()
        
        if diff_days > 365:
            return jsonify({'error': 'Date range too large. Please select ≤ 12 months (≤ 365 days).'}), 400

        # Get Landsat collection (one image per day)
        landsat_col = get_landsat_collection(aoi, start_date, end_date, cloud_filter=20)
        
        landsat_count = landsat_col.size().getInfo()
        if landsat_count == 0:
            return jsonify({'error': 'No cloud-free Landsat imagery found for the selected period.'}), 404

        # Get Sentinel-1 soil moisture (simple processing)
        logger.info("Processing Sentinel-1 soil moisture (optimized)...")
        sentinel1_col = get_sentinel1_soil_moisture_simple(aoi, start_date, end_date)
        
        # Calculate seasonal NDVI min/max for CWSI
        ndvi_coll = landsat_col.map(apply_scale_factors).map(cloud_mask_landsat).map(
            lambda img: img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        )
        
        ndvi_perc = ndvi_coll.reduce(ee.Reducer.percentile([20, 90])).rename(['NDVI_p20', 'NDVI_p90'])
        ndvi_stats = ndvi_perc.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        
        ndvi_min = ee.Number(ndvi_stats.get('NDVI_p20')).clamp(0.05, 0.2)
        ndvi_max = ee.Number(ndvi_stats.get('NDVI_p90')).clamp(0.4, 0.9)
        
        ndvi_min_img = ee.Image.constant(ndvi_min)
        ndvi_max_img = ee.Image.constant(ndvi_max)

        logger.info(f"NDVI range: {ndvi_min.getInfo():.3f} - {ndvi_max.getInfo():.3f}")

        # Get all unique dates from Landsat
        dates_list = landsat_col.aggregate_array('selected_date').distinct().sort().getInfo()
        
        logger.info(f"Processing {len(dates_list)} unique dates with Landsat data")

        # Process each date
        def compute_daily(date_str):
            date = ee.Date(date_str)
            
            # Get Landsat image for this date
            landsat_img = landsat_col.filter(ee.Filter.eq('selected_date', date_str)).first()
            
            # Process Landsat image
            landsat_img = apply_scale_factors(landsat_img)
            landsat_img = cloud_mask_landsat(landsat_img)
            processed_img = calculate_advanced_indices(landsat_img, aoi, ndvi_min_img, ndvi_max_img)
            
            # Get ETo (MODIS)
            eto_col = ee.ImageCollection("MODIS/061/MOD16A2") \
                        .filterDate(date, date.advance(4, 'day')) \
                        .select('PET') \
                        .map(lambda img: img.multiply(0.1).divide(8).rename('eto'))
            eto_count = eto_col.size()
            
            eto_img = ee.Algorithms.If(
                eto_count.gt(0),
                eto_col.mean().rename('ETo'),
                ee.Image.constant(0).mask(ee.Image.constant(0)).rename('ETo')
            )
            eto_img = ee.Image(eto_img)

            # Get Precipitation (CHIRPS)
            chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                          .filterDate(date, date.advance(1, 'day'))
            chirps_count = chirps_col.size()
            
            rain_img = ee.Algorithms.If(
                chirps_count.gt(0),
                chirps_col.sum().rename('Rain'),
                ee.Image.constant(0).rename('Rain')
            )
            rain_img = ee.Image(rain_img)
            eff_rain = rain_img.multiply(0.8).rename('EffRain')

            # Get Sentinel-1 soil moisture (7-day window)
            if sentinel1_col:
                sm_window = sentinel1_col.filterDate(
                    date.advance(-4, 'day'),
                    date.advance(4, 'day')
                )
                sm_count = sm_window.size()
                sm_available = sm_count.gt(0)
            else:
                sm_count = ee.Number(0)
                sm_available = ee.Number(0)

            # Fallback to SMAP
            smap_col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006") \
                         .filterDate(date, date.advance(1, 'day')) \
                         .select('soil_moisture_am')
            smap_count = smap_col.size()

            # Prioritize Sentinel-1, fallback to SMAP
            if sentinel1_col:
                sm_img = ee.Algorithms.If(
                    sm_available,
                    sm_window.mean().rename('SoilMoisture'),
                    ee.Algorithms.If(
                        smap_count.gt(0),
                        smap_col.mean().multiply(200).rename('SoilMoisture'),
                        ee.Image.constant(35).rename('SoilMoisture')
                    )
                )
            else:
                sm_img = ee.Algorithms.If(
                    smap_count.gt(0),
                    smap_col.mean().multiply(200).rename('SoilMoisture'),
                    ee.Image.constant(35).rename('SoilMoisture')
                )
            
            sm_img = ee.Image(sm_img)

            # Previous day soil moisture for deltaS
            if sentinel1_col:
                sm_prev_window = sentinel1_col.filterDate(
                    date.advance(-5, 'day'),
                    date.advance(-1, 'day')
                )
                sm_prev_available = sm_prev_window.size().gt(0)
                sm_prev_img = ee.Algorithms.If(
                    sm_prev_available,
                    sm_prev_window.mean().rename('SoilMoisturePrev'),
                    sm_img.rename('SoilMoisturePrev')
                )
            else:
                sm_prev_col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006") \
                                .filterDate(date.advance(-1, 'day'), date) \
                                .select('soil_moisture_am')
                sm_prev_img = ee.Algorithms.If(
                    sm_prev_col.size().gt(0),
                    sm_prev_col.mean().multiply(200).rename('SoilMoisturePrev'),
                    sm_img.rename('SoilMoisturePrev')
                )
            
            sm_prev_img = ee.Image(sm_prev_img)

            # Calculate deltaS (change in soil moisture)
            deltaS_mm = sm_img.subtract(sm_prev_img).rename('DeltaS_mm')

            # ETc and CWR
            kc = processed_img.select('Kc')
            etc_img = kc.multiply(eto_img).rename('ETc')
            
            cwr_candidate = etc_img.subtract(eff_rain).subtract(deltaS_mm)
            cwr_safe = cwr_candidate.where(cwr_candidate.lt(0), 0).rename('CWR')

            # Irrigation need (70% efficiency)
            irrigation_need = cwr_safe.divide(0.7).rename('Irrigation_Need')

            # Combine all bands
            combined = ee.Image.cat([
                processed_img.select(['NDVI', 'SAVI', 'CWSI', 'Kc', 'LST', 'EVI', 'LAI']),
                eto_img, rain_img, eff_rain, deltaS_mm, sm_img,
                etc_img, cwr_safe, irrigation_need
            ])

            stats = combined.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            )

            # Determine soil moisture source
            sm_source = ee.Algorithms.If(
                sentinel1_col,
                ee.Algorithms.If(sm_available, 'Sentinel-1', 
                                ee.Algorithms.If(smap_count.gt(0), 'SMAP', 'Default')),
                ee.Algorithms.If(smap_count.gt(0), 'SMAP', 'Default')
            )

            feature = ee.Feature(None, stats) \
                .set('date', date_str) \
                .set('landsat_count', 1) \
                .set('sentinel1_count', sm_count if sentinel1_col else 0) \
                .set('smap_count', smap_count) \
                .set('chirps_count', chirps_count) \
                .set('eto_count', eto_count) \
                .set('has_landsat', True) \
                .set('has_sentinel1', sm_available if sentinel1_col else False) \
                .set('has_smap', smap_count.gt(0)) \
                .set('has_eto', eto_count.gt(0)) \
                .set('has_chirps', chirps_count.gt(0)) \
                .set('soil_moisture_source', sm_source)

            return feature

        # Process all dates
        daily_data = ee.FeatureCollection([compute_daily(d) for d in dates_list])

        logger.debug("About to call getInfo() on daily_data for date range %s to %s", start_date, end_date)

        try:
            raw = daily_data.getInfo().get('features', [])
            logger.debug("getInfo() returned %d features", len(raw))

        except Exception as e:
            logger.error("Exception when calling getInfo(): %s", str(e))
            logger.error("Traceback:\n%s", traceback.format_exc())
            return jsonify({'error': 'Earth Engine getInfo() failed', 'details': str(e)}), 500

        result = []
        for f in raw:
            if not f:
                continue

            props = f.get('properties', {})
            date_str = props.get('date', 'UNKNOWN_DATE')
            normalized = {k.lower(): v for k, v in props.items()}

            # Fill missing keys
            for key in ['ndvi', 'savi', 'cwsi', 'kc', 'lst', 'eto', 'rain', 'effrain', 
                       'soilmoisture', 'deltas_mm', 'etc', 'cwr', 'irrigation_need', 'evi', 'lai']:
                normalized.setdefault(key, None)
            if 'soilmoisture' in normalized:
                normalized['soilmoisture_mm'] = normalized['soilmoisture']
    
# Ensure deltas is available
            if 'deltas_mm' in normalized:
                normalized['deltas'] = normalized['deltas_mm']    

            landsat_c = normalized.get('landsat_count', 1)
            s1_c = normalized.get('sentinel1_count', 0)
            smap_c = normalized.get('smap_count', 0)
            chirps_c = normalized.get('chirps_count', 0)
            eto_c = normalized.get('eto_count', 0)
            sm_source = normalized.get('soil_moisture_source', 'Unknown')

            logger.info("DATE: %s | landsat: %s | sentinel1: %s | smap: %s | chirps: %s | eto: %s | SM source: %s",
                        date_str, landsat_c, s1_c, smap_c, chirps_c, eto_c, sm_source)

            result.append(normalized)

        logger.debug("Returning JSON result with %d date entries", len(result))
        return jsonify(result)

    except Exception as e:
        logger.error("Unhandled exception in /get_daily_data: %s", str(e))
        logger.error("Traceback:\n%s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/get_irrigation_calendar', methods=['POST'])
def get_irrigation_calendar():
    """Smart irrigation calendar using Landsat + Sentinel-1 - FIXED VERSION"""
    try:
        data = request.get_json()
        user_aoi = data['aoi']
        coords = user_aoi['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)

        start_date = data['start_date']
        end_date = data['end_date']
        
        # Thresholds
        PRECIPITATION_THRESHOLD = data.get('rain_threshold', 2.0)
        CWSI_THRESHOLD = data.get('cwsi_threshold', 0.35)
        SOIL_MOISTURE_THRESHOLD = data.get('sm_threshold', 25.0)
        
        logger.info(f"Generating irrigation calendar with thresholds: Rain={PRECIPITATION_THRESHOLD}mm, CWSI={CWSI_THRESHOLD}, SM={SOIL_MOISTURE_THRESHOLD}%")
        
        # Validate date range
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        diff_days = end.difference(start, 'day').getInfo()
        
        if diff_days > 365:
            return jsonify({'error': 'Date range too large. Please select ≤ 12 months (≤ 365 days).'}), 400

        # Get Landsat collection
        landsat_col = get_landsat_collection(aoi, start_date, end_date, cloud_filter=20)
        landsat_count = landsat_col.size().getInfo()
        
        if landsat_count == 0:
            return jsonify({'error': 'No cloud-free Landsat imagery found for the selected period.'}), 404

        # Get Sentinel-1 soil moisture
        sentinel1_col = get_sentinel1_soil_moisture_simple(aoi, start_date, end_date)
        
        # Calculate seasonal NDVI min/max
        ndvi_coll = landsat_col.map(apply_scale_factors).map(cloud_mask_landsat).map(
            lambda img: img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        )
        
        ndvi_perc = ndvi_coll.reduce(ee.Reducer.percentile([20, 90])).rename(['NDVI_p20', 'NDVI_p90'])
        ndvi_stats = ndvi_perc.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        
        ndvi_min = ee.Number(ndvi_stats.get('NDVI_p20')).clamp(0.05, 0.2)
        ndvi_max = ee.Number(ndvi_stats.get('NDVI_p90')).clamp(0.4, 0.9)
        ndvi_min_img = ee.Image.constant(ndvi_min)
        ndvi_max_img = ee.Image.constant(ndvi_max)

        # Get unique dates from Landsat
        dates_list = landsat_col.aggregate_array('selected_date').distinct().sort().getInfo()
        
        logger.info(f"Processing {len(dates_list)} unique dates for irrigation calendar")

        calendar_data = []

        for idx, date_str in enumerate(dates_list):
            date = ee.Date(date_str)
            
            try:
                # Get Landsat image
                landsat_img = landsat_col.filter(ee.Filter.eq('selected_date', date_str)).first()
                landsat_img = apply_scale_factors(landsat_img)
                landsat_img = cloud_mask_landsat(landsat_img)
                processed_img = calculate_advanced_indices(landsat_img, aoi, ndvi_min_img, ndvi_max_img)
                
                # Get ETo and calculate irrigation parameters
                eto_col = ee.ImageCollection("MODIS/061/MOD16A2") \
                            .filterDate(date, date.advance(4, 'day')) \
                            .select('PET') \
                            .map(lambda img: img.multiply(0.1).divide(8).rename('eto'))
                
                eto_img = ee.Algorithms.If(
                    eto_col.size().gt(0),
                    eto_col.mean(),
                    ee.Image.constant(0).rename('eto')
                )
                eto_img = ee.Image(eto_img)
                
                # Get Kc and calculate irrigation parameters
                kc = processed_img.select('Kc')
                etc = kc.multiply(eto_img) ##Total water loss
                
                # Get precipitation
                precip_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                              .filterDate(date, date.advance(1, 'day'))
                precip_img = ee.Algorithms.If(
                    precip_col.size().gt(0),
                    precip_col.sum(),
                    ee.Image.constant(0)
                )
                precip_img = ee.Image(precip_img)
                eff_rain = precip_img.multiply(0.8)
                
                # Calculate irrigation need components
                cwr = etc.subtract(eff_rain).max(0)
                irrigation_need = cwr.divide(0.7)
                
                # Get all parameters in one reduceRegion call
                combined_img = ee.Image.cat([
                    processed_img.select(['CWSI', 'Kc']),
                    etc.rename('ETc'),
                    cwr.rename('CWR'),
                    irrigation_need.rename('Irrigation_Need'),
                    precip_img.rename('Rain'),
                    eff_rain.rename('EffRain')
                ])
                
                # Get all stats at once
                all_stats = combined_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=30,
                    maxPixels=1e9,
                    bestEffort=True
                ).getInfo()
                
                # Extract values with safe defaults
                cwsi_mean = all_stats.get('CWSI', 0.4)
                precip_value = all_stats.get('Rain', 0)
                irr_value = all_stats.get('Irrigation_Need', 0)
                kc_value = all_stats.get('Kc', 0.5)
                etc_value = all_stats.get('ETc', 0)
                cwr_value = all_stats.get('CWR', 0)
                
                # Get Sentinel-1 soil moisture (7-day window)
                soil_moisture = 35.0  # Default
                sm_source = 'Default'
                sm_images_used = 0
                
                if sentinel1_col:
                    sm_coll = sentinel1_col.filterDate(
                        date.advance(-4, 'day'),
                        date.advance(4, 'day')
                    )
                    has_sm = sm_coll.size().getInfo() > 0
                    
                    if has_sm:
                        try:
                            sm_stats = sm_coll.mean().reduceRegion(
                                reducer=ee.Reducer.mean(),
                                geometry=aoi,
                                scale=30,
                                maxPixels=1e9,
                                bestEffort=True
                            ).getInfo()
                            soil_moisture = sm_stats.get('soil_moisture', 35.0)
                            sm_source = 'Sentinel-1'
                            sm_images_used = sm_coll.size().getInfo()
                        except:
                            soil_moisture = 35.0
                    else:
                        # Fallback to SMAP
                        smap_col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006") \
                                     .filterDate(date, date.advance(1, 'day')) \
                                     .select('soil_moisture_am')
                        if smap_col.size().getInfo() > 0:
                            try:
                                smap_stats = smap_col.mean().multiply(200).reduceRegion(
                                    reducer=ee.Reducer.mean(),
                                    geometry=aoi,
                                    scale=30,
                                    maxPixels=1e9,
                                    bestEffort=True
                                ).getInfo()
                                soil_moisture = smap_stats.get('soil_moisture_am', 35.0)
                                sm_source = 'SMAP'
                                sm_images_used = 1
                            except:
                                soil_moisture = 35.0
                
                # SMART IRRIGATION LOGIC: Rainfall-first OR logic
                should_irrigate = False
                advice = ''
                priority = 'LOW'
                
                if precip_value < PRECIPITATION_THRESHOLD:
                    # Low rainfall - check stress indicators
                    if cwsi_mean > CWSI_THRESHOLD or soil_moisture < SOIL_MOISTURE_THRESHOLD:
                        should_irrigate = True
                        if irr_value > 2.0:
                            advice = ' Low rain + Stress - Immediate irrigation needed'
                            priority = 'URGENT'
                        elif irr_value > 0.5:
                            advice = ' Low rain + Stress - Irrigation needed soon'
                            priority = 'HIGH'
                        else:
                            advice = ' Low rain + Stress - Light irrigation needed'
                            priority = 'MEDIUM'
                    else:
                        advice = ' Low rain but plants healthy - No irrigation needed'
                        priority = 'LOW'
                else:
                    advice = ' Adequate rainfall - No irrigation needed'
                    priority = 'LOW'
                
                final_irrigation = irr_value if should_irrigate else 0.0
                
                # Collect data
                calendar_entry = {
                    'date': date_str,
                    'month': date.format('YYYY-MM').getInfo(),
                    'cwsi_mean': round(cwsi_mean, 3),
                    'soil_moisture_percent': round(soil_moisture, 2),
                    'rain_mm': round(precip_value, 2),
                    'original_irrigation_mm': round(irr_value, 2),
                    'final_irrigation_mm': round(final_irrigation, 2),
                    'should_irrigate': should_irrigate,
                    'advice': advice,
                    'priority': priority,
                    'soil_moisture_source': sm_source,
                    'sm_images_used': sm_images_used,
                    'kc_mean': round(kc_value, 3),
                    'etc_mm': round(etc_value, 2),
                    'cwr_mm': round(cwr_value, 2)
                }
                
                calendar_data.append(calendar_entry)
                
            except Exception as e:
                logger.warning(f"Failed to process date {date_str}: {str(e)}")
                # Skip this date and continue
                continue
            
            # Progress logging
            if (idx + 1) % 5 == 0:
                logger.info(f"Processed {idx + 1}/{len(dates_list)} dates")
        
        if not calendar_data:
            return jsonify({'error': 'No valid data could be processed for the selected period.'}), 404
        
        # Calculate summary statistics
        irrigation_events = [d for d in calendar_data if d['should_irrigate']]
        total_water = sum(d['final_irrigation_mm'] for d in irrigation_events)
        original_water = sum(d['original_irrigation_mm'] for d in calendar_data)
        water_saved = original_water - total_water
        
        urgent_count = len([d for d in irrigation_events if d['priority'] == 'URGENT'])
        high_count = len([d for d in irrigation_events if d['priority'] == 'HIGH'])
        medium_count = len([d for d in irrigation_events if d['priority'] == 'MEDIUM'])
        
        summary = {
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'total_days': len(calendar_data),
                'successful_dates': len(calendar_data)
            },
            'irrigation': {
                'total_events': len(irrigation_events),
                'total_water_mm': round(total_water, 2),
                'average_per_event_mm': round(total_water / len(irrigation_events), 2) if len(irrigation_events) > 0 else 0,
                'water_saved_mm': round(water_saved, 2),
                'savings_percent': round((water_saved / original_water * 100), 2) if original_water > 0 else 0
            },
            'urgency': {
                'urgent': urgent_count,
                'high': high_count,
                'medium': medium_count
            },
            'thresholds_used': {
                'precipitation_mm': PRECIPITATION_THRESHOLD,
                'cwsi': CWSI_THRESHOLD,
                'soil_moisture_percent': SOIL_MOISTURE_THRESHOLD
            }
        }
        
        logger.info(f"Calendar generated: {len(irrigation_events)} irrigation events, {round(total_water, 1)}mm total water")
        
        return jsonify({
            'success': True,
            'summary': summary,
            'calendar': calendar_data
        })

    except Exception as e:
        logger.error(f"Error in /get_irrigation_calendar: {e}")
        logger.error("Traceback:\n%s", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    

@app.route('/get_land_cover_analysis', methods=['POST'])
def get_land_cover_analysis():
    """Land cover analysis using Landsat"""
    try:
        data = request.get_json()
        user_aoi = data['aoi']
        coords = user_aoi['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)
        
        start_date = data['start_date']
        end_date = data['end_date']
        time_scale = data.get('time_scale', 'monthly')
        
        # Validate date range
        start = ee.Date(start_date)
        end = ee.Date(end_date)
        diff_days = end.difference(start, 'day').getInfo()

        if diff_days > 365:
            return jsonify({'error': 'Date range too large. Please select ≤ 12 months (≤ 365 days).'}), 400

        # Use Landsat collection
        landsat_col = get_landsat_collection(aoi, start_date, end_date, cloud_filter=30)
        
        # Add NDVI band
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
            return image.addBands(ndvi)

        # Group by time period
        def make_time_key(image):
            date = image.date()
            if time_scale == 'monthly':
                key = date.format('YYYY-MM')
            else:
                key = date.format('YYYY-ww')
            return image.set('time_key', key)

        landsat_with_keys = landsat_col.map(apply_scale_factors).map(cloud_mask_landsat).map(make_time_key)

        # Composite by period
        distinct_periods = landsat_with_keys.distinct('time_key')
        
        def composite_function(img):
            key = img.get('time_key')
            period_images = landsat_with_keys.filter(ee.Filter.eq('time_key', key))
            composite = period_images.median() \
                                   .set('time_key', key) \
                                   .set('date', ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'))
            return composite

        composite_by_period = ee.ImageCollection(distinct_periods.map(composite_function)) \
                                .sort('time_key').limit(12)

        landsat_with_ndvi = composite_by_period.map(add_ndvi)

        # Classification
        def classify_pixels(image):
            ndvi = image.select('NDVI')
            water = ndvi.lt(0)
            bareland = ndvi.gte(0).And(ndvi.lt(0.1))
            builtup = ndvi.gte(0.1).And(ndvi.lt(0.2))
            sparse_veg = ndvi.gte(0.2).And(ndvi.lt(0.4))
            full_veg = ndvi.gte(0.4)

            classified = water.multiply(1) \
                .add(bareland.multiply(2)) \
                .add(builtup.multiply(3)) \
                .add(sparse_veg.multiply(4)) \
                .add(full_veg.multiply(5)) \
                .rename('class')

            return image.addBands(classified)

        classified_collection = landsat_with_ndvi.map(classify_pixels)

        # Pixel counting
        def count_pixels(image):
            class_band = image.select('class')
            pixel_counts = ee.Dictionary(
                class_band.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=aoi,
                    scale=30,
                    maxPixels=1e13
                ).get('class')
            )
            return ee.Feature(None, {
                'period': image.get('time_key'),
                'pixel_counts': pixel_counts
            })

        pixel_count_per_class = classified_collection.map(count_pixels)
        features_list = pixel_count_per_class.toList(12).getInfo()
        
        # Process results
        result_data = []
        class_names = {
            '1': 'Water',
            '2': 'Bare Land', 
            '3': 'Built-up',
            '4': 'Sparse Vegetation',
            '5': 'Dense Vegetation'
        }
        
        for feature in features_list:
            properties = feature['properties']
            period = properties['period']
            pixel_counts = properties.get('pixel_counts', {})
            
            readable_counts = {}
            total_pixels = 0
            
            for class_id, count in pixel_counts.items():
                class_name = class_names.get(class_id, f'Class {class_id}')
                readable_counts[class_name] = count
                total_pixels += count
            
            percentages = {}
            for class_name, count in readable_counts.items():
                if total_pixels > 0:
                    percentages[class_name] = round((count / total_pixels) * 100, 2)
                else:
                    percentages[class_name] = 0
            
            result_data.append({
                'period': period,
                'pixel_counts': readable_counts,
                'percentages': percentages,
                'total_pixels': total_pixels
            })
        
        return jsonify({
            'success': True,
            'time_scale': time_scale,
            'periods_analyzed': len(result_data),
            'data': result_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_gee_tile', methods=['POST'])
def get_gee_tile():
    """Enhanced tile generation with Landsat and Sentinel-1 + STATS"""
    try:
        data = request.get_json()
        coords = data['aoi']['geometry']['coordinates']
        aoi = ee.Geometry.Polygon(coords)
        layer_type = data.get("layer", "ndvi")

        start_date_param = data.get('start_date')
        end_date_param = data.get('end_date')
        
        if start_date_param and end_date_param:
            start_date = ee.Date(start_date_param)
            end_date = ee.Date(end_date_param)
        else:
            end_date = ee.Date(datetime.now())
            start_date = end_date.advance(-30, 'day')

        # Get Landsat collection
        landsat_col = get_landsat_collection(aoi, start_date, end_date, cloud_filter=30)
        landsat_count = landsat_col.size().getInfo()
        
        if landsat_count == 0:
            return jsonify({'error': 'No cloud-free Landsat imagery found for the selected period.'}), 404

        # Process Landsat data
        processed_col = landsat_col.map(apply_scale_factors).map(cloud_mask_landsat)
        
        # Calculate seasonal NDVI for CWSI
        ndvi_coll = processed_col.map(
            lambda img: img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        )
        ndvi_perc = ndvi_coll.reduce(ee.Reducer.percentile([20, 90])).rename(['NDVI_p20', 'NDVI_p90'])
        ndvi_stats = ndvi_perc.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        ndvi_min = ee.Number(ndvi_stats.get('NDVI_p20')).clamp(0.05, 0.2)
        ndvi_max = ee.Number(ndvi_stats.get('NDVI_p90')).clamp(0.6, 0.9)
        ndvi_min_img = ee.Image.constant(ndvi_min)
        ndvi_max_img = ee.Image.constant(ndvi_max)

        # Create median composite
        landsat_median = processed_col.median()
        processed_img = calculate_advanced_indices(landsat_median, aoi, ndvi_min_img, ndvi_max_img)

        # Layer-specific processing
        stats_data = None
        
        if layer_type == 'ndvi':
            selected_img = processed_img.select('NDVI')
            vis_params = {'min': 0, 'max': 1, 'palette': ['#FFFFFF', '#CE7E45', '#DF923D', '#F1B555', '#FCD163', '#99B718', '#74A901', '#66A000', '#529400', '#3E8601', '#207401', '#056201', '#004C00', '#023B01', '#012E01', '#011D01', '#011301']}
            
            # Calculate statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.stdDev(), '', True
                ).combine(
                    ee.Reducer.min(), '', True
                ).combine(
                    ee.Reducer.max(), '', True
                ).combine(
                    ee.Reducer.median(), '', True
                ).combine(
                    ee.Reducer.percentile([10, 25, 75, 90]), '', True
                ),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('NDVI_mean'),
                'stdDev': stats.get('NDVI_stdDev'),
                'min': stats.get('NDVI_min'),
                'max': stats.get('NDVI_max'),
                'median': stats.get('NDVI_median'),
                'p10': stats.get('NDVI_p10'),
                'p25': stats.get('NDVI_p25'),
                'p75': stats.get('NDVI_p75'),
                'p90': stats.get('NDVI_p90')
            }
            
        elif layer_type == 'savi':
            selected_img = processed_img.select('SAVI')
            vis_params = {'min': 0, 'max': 1, 'palette': ['red', 'orange', 'yellow', 'green']}
            
        elif layer_type == 'cwsi':
            selected_img = processed_img.select('CWSI')
            vis_params = {'min': 0, 'max': 1, 'palette': ['blue', 'cyan', 'white', 'yellow', 'red']}
            
        elif layer_type == 'kc':
            selected_img = processed_img.select('Kc')
            vis_params = {'min': 0, 'max': 1.5, 'palette': ['blue', 'cyan', 'white', 'yellow', 'red']}
            
            # Calculate Kc statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('Kc_mean'),
                'min': stats.get('Kc_min'),
                'max': stats.get('Kc_max')
            }
            
        elif layer_type == 'lst':
            selected_img = processed_img.select('LST')
            vis_params = {'min': 280, 'max': 330, 'palette': ['PeachPuff', 'SandyBrown', 'OrangeRed','FireBrick','Tomato']}
            
            # Calculate LST statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('LST_mean'),
                'min': stats.get('LST_min'),
                'max': stats.get('LST_max')
            }
            
        elif layer_type == 'lai':
            selected_img = processed_img.select('LAI')
            vis_params = {'min': 0, 'max': 6, 'palette': ['white', 'lightgreen', 'green', 'darkgreen']}
            
        elif layer_type == 'rgb':
            selected_img = landsat_median.select(['SR_B4', 'SR_B3', 'SR_B2'])
            vis_params = {'min': 0, 'max': 0.3, 'gamma': 1.4}
            
        elif layer_type == 'etc':
            eto_col = ee.ImageCollection("MODIS/061/MOD16A2") \
                        .filterDate(start_date, end_date) \
                        .select('PET') \
                        .map(lambda img: img.multiply(0.1).divide(8).rename('eto'))
            eto_img = eto_col.mean()
            kc = processed_img.select('Kc')
            selected_img = kc.multiply(eto_img).rename('ETc')
            vis_params = {'min': 0, 'max': 10, 'palette': ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']}
            
            # Calculate ETc statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('ETc_mean'),
                'min': stats.get('ETc_min'),
                'max': stats.get('ETc_max')
            }
            
        elif layer_type == 'soilmoisture':
            logger.info("Processing Sentinel-1 soil moisture for visualization...")
            sentinel1_col = get_sentinel1_soil_moisture_advanced(aoi, start_date, end_date)
            
            if sentinel1_col and sentinel1_col.size().getInfo() > 0:
                selected_img = sentinel1_col.median()
                vis_params = {'min': 0, 'max': 100, 'palette': ['brown', 'yellow', 'lightgreen', 'darkgreen', 'blue']}
                
                # Calculate soil moisture statistics
                stats = selected_img.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                    geometry=aoi,
                    scale=30,
                    maxPixels=1e13
                ).getInfo()
                
                stats_data = {
                    'mean': stats.get('soil_moisture_mean'),
                    'min': stats.get('soil_moisture_min'),
                    'max': stats.get('soil_moisture_max')
                }
            else:
                smap_col = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006") \
                            .filterBounds(aoi) \
                            .filterDate(start_date, end_date) \
                            .select('soil_moisture_am')
                selected_img = smap_col.median().multiply(200).rename('SoilMoisture')
                vis_params = {'min': 0, 'max': 100, 'palette': ['brown', 'yellow', 'lightgreen', 'darkgreen', 'blue']}
            
        elif layer_type == 'irrigation_need':
            # Calculate irrigation need based on ETc and rainfall
            eto_col = ee.ImageCollection("MODIS/061/MOD16A2") \
                        .filterDate(start_date, end_date) \
                        .select('PET') \
                        .map(lambda img: img.multiply(0.1).divide(8).rename('eto'))
            eto_img = eto_col.mean()
            
            chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                          .filterDate(start_date, end_date)
            rain_img = chirps_col.sum().rename('Rain')
            eff_rain = rain_img.multiply(0.8)
            
            kc = processed_img.select('Kc')
            etc = kc.multiply(eto_img)
            cwr = etc.subtract(eff_rain).max(0)
            selected_img = cwr.divide(0.7).rename('Irrigation_Need')
            vis_params = {'min': 0, 'max': 10, 'palette': ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']}
            
            # Calculate irrigation need statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('Irrigation_Need_mean'),
                'min': stats.get('Irrigation_Need_min'),
                'max': stats.get('Irrigation_Need_max')
            }
            
        elif layer_type == 'cwr':
            eto_col = ee.ImageCollection("MODIS/061/MOD16A2") \
                        .filterDate(start_date, end_date) \
                        .select('PET') \
                        .map(lambda img: img.multiply(0.1).divide(8).rename('eto'))
            eto_img = eto_col.mean()
            
            chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
                          .filterDate(start_date, end_date)
            rain_img = chirps_col.sum().rename('Rain')
            eff_rain = rain_img.multiply(0.8)
            
            kc = processed_img.select('Kc')
            etc = kc.multiply(eto_img)
            selected_img = etc.subtract(eff_rain).max(0).rename('CWR')
            vis_params = {'min': 0, 'max': 10, 'palette': ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']}
            
            # Calculate CWR statistics
            stats = selected_img.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                geometry=aoi,
                scale=30,
                maxPixels=1e13
            ).getInfo()
            
            stats_data = {
                'mean': stats.get('CWR_mean'),
                'min': stats.get('CWR_min'),
                'max': stats.get('CWR_max')
            }
            
        elif layer_type == 'deltas':
            sentinel1_col = get_sentinel1_soil_moisture_simple(aoi, start_date, end_date)
            if sentinel1_col and sentinel1_col.size().getInfo() > 1:
                images_list = sentinel1_col.toList(sentinel1_col.size())
                last_img = ee.Image(images_list.get(-1))
                first_img = ee.Image(images_list.get(0))
                selected_img = last_img.subtract(first_img).rename('DeltaS')
                vis_params = {'min': -20, 'max': 20, 'palette': ['red', 'orange', 'white', 'lightgreen', 'darkgreen']}
                
                # Calculate delta statistics
                stats = selected_img.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.min(), '', True).combine(ee.Reducer.max(), '', True),
                    geometry=aoi,
                    scale=30,
                    maxPixels=1e13
                ).getInfo()
                
                stats_data = {
                    'mean': stats.get('DeltaS_mean'),
                    'min': stats.get('DeltaS_min'),
                    'max': stats.get('DeltaS_max')
                }
            else:
                return jsonify({'error': 'Not enough soil moisture data for delta calculation'}), 404
        else:
            return jsonify({"error": "Invalid layer type specified"}), 400

        # Visualize and clip
        image_to_visualize = selected_img.visualize(**vis_params)
        clipped_image = image_to_visualize.clip(aoi)
        
        # Get map ID
        map_id_dict = ee.data.getMapId({'image': clipped_image})
        
        response = {
            'mapid': map_id_dict['mapid'],
            'urlFormat': map_id_dict['tile_fetcher'].url_format,
            'stats': stats_data
        }
        
        logger.info(f"Layer {layer_type} generated successfully with stats: {stats_data}")
        
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Tile generation error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Report generation endpoint"""
    try:
        payload = request.get_json()
        return generate_report_response(payload)
    except Exception as e:
        logger.error("generate_report failed: %s", e, exc_info=True)
        return jsonify({'error': str(e)}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5000)