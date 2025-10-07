#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project='mrv-fates')

# # sentinel-2
def add_CIre(image):
    """
    Calculate the Chlorophyll Index Red Edge (CIre) and add it as a new band to the input image.

    Parameters:
    image (ee.Image): Input satellite image containing bands B5 and B7.

    Returns:
    ee.Image: Original image with an additional band named 'CIre' containing the Chlorophyll Index Red Edge values.
    """

    rededge1 = image.select('B5').divide(10000)
    rededge3 = image.select('B7').divide(10000)
    cire = ((rededge3.divide(rededge1)).subtract(1)).rename(['CIre'])   
    return image.addBands(cire)


def add_CIgreen(image):
    """
    Calculate the Chlorophyll Index Green (CIgreen) and add it as a new band to the input image.
    CIgreen = (NIR / Green) - 1

    Parameters:
    image (ee.Image): Input satellite image containing bands B3 (green) and B8 (near-infrared).

    Returns:
    ee.Image: Original image with an additional band named 'CIgreen' containing the Chlorophyll Index Green values.
    """

    green = image.select('B3').divide(10000)
    nir = image.select('B8').divide(10000)
    cigreen = ((nir.divide(green)).subtract(1)).rename(['CIgreen'])   
    return image.addBands(cigreen)


def add_EVI1(image):
    """
    Calculate the Enhanced Vegetation Index (EVI1) and add it as a new band to the input image.

    The formula used here is:
    EVI1 = 2.5 * ( (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1) )

    Parameters:
    image (ee.Image): Input satellite image containing bands B2 (blue), B4 (red), and B8 (near-infrared).

    Returns:
    ee.Image: Original image with an additional band named 'EVI1' containing the Enhanced Vegetation Index values.
    """

    blue = image.select('B2').divide(10000)
    red = image.select('B4').divide(10000)
    nir = image.select('B8').divide(10000)
    evi1 = ((((nir.subtract(red))).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))).multiply(2.5)).rename(['EVI1'])   
    return image.addBands(evi1)


def add_GNDVI(image):
    """
    Calculate the Green Normalized Difference Vegetation Index (GNDVI) and add it as a new band to the input image.

    GNDVI is a vegetation index similar to NDVI but uses the green band instead of the red band,
    which can provide better sensitivity to chlorophyll concentration.

    The formula is:
    GNDVI = (NIR - Green) / (NIR + Green)

    Parameters:
    image (ee.Image): Input satellite image containing bands B3 (green) and B8 (near-infrared).

    Returns:
    ee.Image: Original image with an additional band named 'GNDVI' containing the Green NDVI values.
    """

    green = image.select('B3').divide(10000)
    nir = image.select('B8').divide(10000)
    gndvi = (nir.subtract(green).divide(nir.add(green))).rename(['GNDVI'])   
    return image.addBands(gndvi)


def add_MCARI1(image):
    """
    Calculate the Modified Chlorophyll Absorption in Reflectance Index 1 (MCARI1) and add it as a new band to the input image.

    MCARI1 is an index designed to estimate chlorophyll content by minimizing the effects of soil background and non-photosynthetic vegetation.
    The formula is:
    MCARI1 = [(RedEdge1 - Red) - 0.2 * (RedEdge1 - Green)] * (RedEdge1 / Red)

    Parameters:
    image (ee.Image): Input satellite image containing bands B3 (green), B4 (red), and B5 (red edge 1).

    Returns:
    ee.Image: Original image with an additional band named 'MCARI1' containing the MCARI1 values.
    """    
    rededge1 = image.select('B5').divide(10000)
    red = image.select('B4').divide(10000)
    green = image.select('B3').divide(10000)
    mcari1 = ((rededge1.subtract(red).subtract(rededge1.subtract(green)).multiply(0.2)).multiply(rededge1.divide(red))).rename(['MCARI1'])   
    return image.addBands(mcari1)


def add_MTCI1(image):
    """
    Calculate the MERIS Terrestrial Chlorophyll Index 1 (MTCI1) and add it as a new band to the input image.

    MTCI1 is used to estimate chlorophyll content in vegetation by combining near-infrared and red edge bands.
    The formula is:
    MTCI1 = (NIR - RedEdge1) / (RedEdge1 - Red)

    Parameters:
    image (ee.Image): Input satellite image containing bands B4 (red), B5 (red edge 1), and B8 (near-infrared).

    Returns:
    ee.Image: Original image with an additional band named 'MTCI1' containing the MTCI1 values.
    """   
    rededge1 = image.select('B5').divide(10000)
    nir = image.select('B8').divide(10000)
    red = image.select('B4').divide(10000)
    mtci1 = ((nir.subtract(rededge1)).divide(rededge1.subtract(red))).rename(['MTCI1'])   
    return image.addBands(mtci1)



def add_NDI45(image):
    """
    Calculate the Normalized Difference Index between bands 4 and 5 (NDI45) and add it as a new band to the input image.

    NDI45 is a normalized difference index calculated using red edge 1 (B5) and red (B4) bands.
    The formula is similar to NDVI but uses different bands:
    NDI45 = (RedEdge1 - Red) / (RedEdge1 + Red)

    Parameters:
    image (ee.Image): Input satellite image containing bands B4 (red) and B5 (red edge 1).

    Returns:
    ee.Image: Original image with an additional band named 'NDI45' containing the NDI45 values.
    """

    rededge1 = image.select('B5').divide(10000)
    red = image.select('B4').divide(10000)
    ndi45 = (rededge1.subtract(red).divide(rededge1.add(red))).rename(['NDI45'])   
    return image.addBands(ndi45)


def add_NDWI1(image):
    """
    Calculate the Normalized Difference Water Index 1 (NDWI1) and add it as a new band to the input image.

    It is calculated using the near-infrared (NIR) and shortwave infrared 1 (SWIR1) bands:
    NDWI1 = (NIR - SWIR1) / (NIR + SWIR1)

    Parameters:
    image (ee.Image): Input satellite image containing bands B8 (NIR) and B11 (SWIR1).

    Returns:
    ee.Image: Original image with an additional band named 'NDWI1' containing the NDWI1 values.
    """    
    swir1 = image.select('B11').divide(10000)
    nir = image.select('B8').divide(10000)
    ndwi1 = nir.subtract(swir1).divide(nir.add(swir1)).rename(['NDWI1'])   
    return image.addBands(ndwi1)



def add_NDVI56(image):
    """
    Calculate the Normalized Difference Vegetation Index between bands 5 and 6 (NDVI56) and add it as a new band to the input image.

    The formula is:
    NDVI56 = (RedEdge2 - RedEdge1) / (RedEdge2 + RedEdge1)

    Parameters:
    image (ee.Image): Input satellite image containing bands B5 (red edge 1) and B6 (red edge 2).

    Returns:
    ee.Image: Original image with an additional band named 'NDVI56' containing the NDVI56 values.
    """    
    rededge1 = image.select('B5').divide(10000)
    rededge2 = image.select('B6').divide(10000)
    ndvi56 = (rededge2.subtract(rededge1).divide(rededge2.add(rededge1))).rename(['NDVI56'])   
    return image.addBands(ndvi56)


def add_NLI(image):
    """
    Calculate the Normalized Leaf Index (NLI) and add it as a new band to the input image.

    NLI is an index used to assess leaf characteristics by combining red and red edge bands.
    The formula used here is:
    NLI = (RedEdge^2 - Red) / (RedEdge^2 + Red)

    Parameters:
    image (ee.Image): Input satellite image containing bands B4 (red) and B5 (red edge 1).

    Returns:
    ee.Image: Original image with an additional band named 'NLI' containing the NLI values.
    """
    red = image.select('B4').divide(10000)
    rededge1 = image.select('B5').divide(10000)
    nli = ((rededge1.multiply(rededge1)).subtract(red).divide(rededge1.multiply(rededge1).add(red))).rename(['NLI'])   
    return image.addBands(nli)


def add_PSSRa(image):
    """
    Calculate the (Pigment Specific Simple Ratio for Chlorophyll a (PSSRa) and add it as a new band to the input image.

    PSSRa = RedEdge / Red

    Parameters:
    image (ee.Image): Input satellite image containing bands B7 (red edge 3) and B4 (red).

    Returns:
    ee.Image: Original image with an additional band named 'PSSRa' containing the PSSRa values.
    """
    rededge3 = image.select('B7').divide(10000)
    red = image.select('B4').divide(10000)
    pssra = (rededge3.divide(red)).rename(['PSSRa'])   
    return image.addBands(pssra)


def add_kNDVI(image):
    """
    Calculate the kernel Normalized Difference Vegetation Index (kNDVI) and add it as a new band to the input image.
    The steps are:
    1. Calculate NDVI = (NIR - Red) / (NIR + Red)
    2. Apply kNDVI = tanh(NDVI)

    Parameters:
    image (ee.Image): Input satellite image containing bands B4 (red) and B8 (near-infrared).

    Returns:
    ee.Image: Original image with an additional band named 'kNDVI' containing the kNDVI values.
    """

    red = image.select('B4').divide(10000)
    nir = image.select('B8').divide(10000)
    ndvi = (nir.subtract(red).divide(nir.add(red))).rename(['NDVI'])      
    kndvi = (ndvi.expression('tanh(b)', {'b': ndvi})).rename(['kNDVI'])   
    return image.addBands(kndvi)


def applyMask(image):
    """
    Apply a cloud and cirrus mask to the input image using the QA60 band.

    The QA60 band contains quality assessment information where:
    - Bit 10 indicates clouds
    - Bit 11 indicates cirrus clouds

    This function masks out pixels where either clouds or cirrus are detected.

    Parameters:
    image (ee.Image): Input satellite image containing the QA60 band.

    Returns:
    ee.Image: Masked image with clouds and cirrus pixels removed.
    """    
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = (qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)))
    return image.updateMask(mask)



def applyCLDPRBMask(image):
    """
    Apply a cloud probability mask to the input image using the MSK_CLDPRB band.

    The MSK_CLDPRB band indicates cloud probability, where:
    - 0 means clear pixels
    - Non-zero values indicate varying probabilities of cloud presence

    This function masks out pixels with any cloud probability (i.e., keeps only pixels with cloud probability = 0).

    Parameters:
    image (ee.Image): Input satellite image containing the MSK_CLDPRB band.

    Returns:
    ee.Image: Masked image with probable cloud pixels removed.
    """

    # Select the MSK_CLDPRB band (cloud probability mask)
    cloudProb = image.select('MSK_CLDPRB')  # original 20m band from COPERNICUS/S2_SR_HARMONIZED
    # cloudProb = image.select('CLOUD_PROB')  # 10m band from COPERNICUS/S2_CLOUD_PROBABILITY
    # Mask pixels where MSK_CLDPRB is not equal to 0 (clear pixels)
    mask = cloudProb.eq(0)
    # mask =  cloudProb.lte(1)  
    return image.updateMask(mask)


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Import and filter s2cloudless
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the collections
    joined = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    # Add cloud probability band to each image
    def add_cloud_bands(img):
        # Get the cloud probability image from the property
        cloud_prob = ee.Image(img.get('s2cloudless')).select('probability')
        # Add it as a band to the original image
        return img.addBands(cloud_prob.rename('CLOUD_PROB_10m'))

    return joined.map(add_cloud_bands)


if __name__ == "__main__":

    # gee assets path
    parent = 'projects/mrv-fates/assets'

    # Ask EE for the list of *direct* children of that folder
    resp = ee.data.listAssets({'parent': parent})  # dict with key 'assets'
    assets_meta = resp.get('assets', [])           # list of dictionaries

    # Keep only assets whose type == 'TABLE'   (vectors/shapefiles)
    table_ids = [a['name'] for a in assets_meta if a['type'] == 'TABLE']
    print('Found', len(table_ids), 'vector assets')

    # Print all found asset IDs with index
    for i, aid in enumerate(table_ids):
        print(f'{i:2d}. {aid}')

    #  Read them in a loop
    for asset_id in tqdm(table_ids):
        #print(table_ids)

        # Filter to process only assets
        if not asset_id.endswith('fishnet'):
            continue  # Skip if it does not end with 'fishnet'

        # Extract basename from asset ID for naming outputs
        basename = os.path.basename(asset_id)      # ← fishnet_25000m_2019_25m_190
        print('\n───────────────────────────────────────────────')
        print('Processing:', asset_id)

        # Load the vector asset as a FeatureCollection
        fishnet = ee.FeatureCollection(asset_id)
        # Get number of features (grid cells) in the collection (client-side)
        total_cells = fishnet.size().getInfo()
        print('\nNumber of grid cells :', total_cells)
        # Get schema (property names)
        property_names = fishnet.first().propertyNames().getInfo()
        print('property_names:', property_names)

        # If UID does not exist, create it
        if 'UID' not in property_names:
            print("⚠️ 'UID' column not found — creating one from system:index")
            fishnet = fishnet.map(lambda f: f.set('UID', f.get('system:index')))

        # Check if 'start' and 'end' exist
        if 'start' in property_names and 'end' in property_names:
            # extract the global start/end timestamps from properties
            start_ms = fishnet.aggregate_first('start').getInfo()
            end_ms   = fishnet.aggregate_first('end').getInfo()
            # convert ms → seconds → datetime
            start = datetime.fromtimestamp(start_ms / 1000)
            end   = datetime.fromtimestamp(end_ms   / 1000)
        else:
            # Manually define start and end in 'YYYY-MM-DD' format
            start = '2019-01-01'   # <-- set your desired start date here
            end   = '2019-12-31'   # <-- set your desired end date here
        print(f'Time range: {start} to {end}') 

        # Build one reducer that returns:  min, max, mean, median, stdDev
        stats_reducer = (ee.Reducer.minMax()               # min & max
                         .combine(ee.Reducer.mean(),    '', True)   # mean
                         .combine(ee.Reducer.median(),  '', True)   # median
                         .combine(ee.Reducer.stdDev(),  '', True))  # std

        # Get the list of features
        uids = fishnet.aggregate_array('UID').getInfo() 

        for uid in tqdm(uids):
            # Filter fishnet to get the feature with the current UID
            feature = fishnet.filter(ee.Filter.eq('UID', uid)).first()

            # Get geometry of this feature (polygon) for spatial filtering
            geom = feature.geometry().getInfo()

             # Retrieve the UID value (for logging)
            uid_value = feature.get('UID').getInfo()
            print(f"Processing UID: {uid_value}")

            # Build Sentinel-2 ImageCollection filtered by spatial bounds and date range
            # Then apply several custom masking and index calculation functions
            s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                                .filterBounds(geom)                        
                                .filterDate(start, end)
                                .map(applyMask)
                                .map(applyCLDPRBMask)
                                .map(add_NDI45)
                                .map(add_NLI)
                                .map(add_MTCI1)
                                .map(add_MCARI1)
                                .map(add_GNDVI)
                                .map(add_PSSRa)
                                .map(add_EVI1)
                                .map(add_NDWI1)
                                .map(add_kNDVI)
                                .map(add_CIgreen)
                                .map(add_CIre)
                                .map(add_NDVI56)
                                .select(['B2', 'B3', 'B4', 'B5', 'B11','B12',
                                         'CIre', 'CIgreen', 'EVI1','GNDVI','MCARI1', 'MTCI1',
                                         'NDI45','NDWI1','NDVI56','NLI', 'PSSRa', 'kNDVI' ,
                                         #'QA60', 'MSK_CLDPRB', 'SCL' , 'CLOUD_PROB_10m'
                                        ])
                                .map(lambda image: image.toFloat())
                                # .map(lambda image: image.toDouble())  # Convert to double at image level
                            )

            # Check bands before reduction: Get the number of images in the filtered ImageCollection
            collection_size = s2_collection.size().getInfo()
            print(f"collection_size: {collection_size}")
            if collection_size == 0:
                # No images found for this feature, skip processing
                print(f"No images found for chunk {chunk_idx}. Skipping...")
                continue  # skip to next UID

            else:
                # print bandNames before reduction
                print("\nBands before ee.Reducer:", s2_collection.first().bandNames().getInfo())
                # print(s2_collection.aggregate_array("system:index").getInfo())  

                # Reduce the ImageCollection to single image with stats_reducer, with each band in Float consistently
                s2_stats = (s2_collection.reduce(stats_reducer)
                                         .clip(geom)
                                         .toFloat() 
                                         # .toDouble() 
                                         .select([
                                                'B2_median', 'B2_min', 'B2_mean', 'B3_mean', 'B3_median', 'B4_mean',
                                                'B5_stdDev', 'B5_median', 'B11_median', 'B11_min', 'B11_max', 'B11_stdDev', 'B11_mean',
                                                'B12_mean', 'B12_median', 'B12_stdDev', 'CIre_median', 'CIgreen_median',
                                                'CIgreen_mean', 'EVI1_stdDev', 'GNDVI_mean', 'GNDVI_median', 'MCARI1_stdDev',
                                                'MTCI1_stdDev', 'NDI45_median', 'NDI45_mean', 'NDWI1_median', 'NDVI56_mean',
                                                'NLI_stdDev', 'PSSRa_median', 'kNDVI_mean'
                                         ])
                           )

                # print bands after reduction to confirm selection
                print("Bands after ee.Reducer:", s2_stats.bandNames().getInfo())

                # Create a descriptive task name for export
                task_desc = f'sentinel2_{basename}_uid_{uid}'

                #  Set up and start an export task to Google Drive
                task = ee.batch.Export.image.toDrive(**{
                                                'image':s2_stats, # 
                                                'description': task_desc,   
                                                # 'folder': '_sentinel2_image',   # folder on google drive
                                                # region=geom, 
                                                'scale': 25,        # Resolution 
                                                'crs': 'EPSG:3857', # Coordinate Reference System
                                                'maxPixels': 1e13, # (Large maxPixels to avoid to avoid limit errors) The field "max_pixels" must have a value between 1 and 10000000000000 inclusive. If unspecified, the default value is 100000000.
                                                'fileFormat':'GeoTIFF',
                                                'formatOptions':{'noData':-9999}                    
                                               })
                # kick off
                task.start()

                print(f'\n    → task: {task_desc} submitted ')
                # Optional: sleep to avoid submitting too many tasks too quickly
                # import time
                # time.sleep(3) # sleep interval to aviod creating multi outdir in google drive

