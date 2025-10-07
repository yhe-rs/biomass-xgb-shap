#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project='mrv-fates')


# # GLO30
def add_slope_aspect(img):

    """
    Adds slope and aspect bands derived from a DEM band in the input image.

    Parameters:
        img (ee.Image): An Earth Engine Image containing a band named 'DEM' representing elevation.

    Returns:
        ee.Image: An image with exactly three bands:
                  - 'Ele': elevation (renamed from 'DEM')
                  - 'Slope': slope in degrees calculated from the DEM
                  - 'Aspect': aspect in degrees calculated from the DEM
                  The original 'DEM' band is renamed to 'Ele'.
    """

    # Select the 'DEM' band from the input image and rename it to 'Ele' (Elevation)
    dem = img.select('DEM').rename('Ele')

    # Calculate slope (in degrees) from the elevation band
    slope = ee.Terrain.slope(dem).rename('Slope')

    # Calculate aspect (in degrees) from the elevation band
    aspect = ee.Terrain.aspect(dem).rename('Aspect')

    # Add the slope and aspect bands to the original image (with renamed  Ele band)
    # Return the image with 'Ele', 'Slope', and 'Aspect' bands
    return img.addBands([slope, aspect]).rename(['Ele', 'Slope', 'Aspect'])



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
            continue  # Skip if it does not match this pattern

        # Extract basename from asset ID for naming outputs
        basename = os.path.basename(asset_id)      # ← fishnet
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

        # Get number of features (grid cells) in the collection
        total_cells = fishnet.size().getInfo()
        print('\nNumber of grid cells :', total_cells)

        # Extract all unique 'UID' values
        uids = fishnet.aggregate_array('UID').getInfo()
        print(f"Found {len(uids)} UIDs")

        # Process each polygon feature by its UID
        for uid in tqdm(uids):
            # GFilter the feature collection to get the feature with the current UID
            feature = fishnet.filter(ee.Filter.eq('UID', uid)).first()

            # Extract geometry of the feature (client-side)
            geom = feature.geometry().getInfo()

            # Get the UID value (client-side)
            uid_value = feature.get('UID').getInfo()
            print(f"Processing UID: {uid_value}")

            # Filter the GLO-30 DEM ImageCollection to images intersecting the polygon geometry
            glo30_collection = (ee.ImageCollection('COPERNICUS/DEM/GLO30')
                               .filterBounds(geom)  # geom touched scene
                               .select('DEM')
                               .map(lambda image: image.toFloat())
                               # .map(lambda image: image.toDouble())
                              )

            # Check bands before reduction: get the number of images in the filtered collection (client-side)
            collection_size = glo30_collection.size().getInfo()
            print(f"collection_size: {collection_size}")
            if collection_size == 0:
                 # No images found for this polygon, skip processing
                print(f"No images found for uid {uid}. Skipping...")
                continue  # skip to next uid

            else:
                # Check bands before reduction
                print("\nBands before ee.Reducer:", glo30_collection.first().bandNames().getInfo())
                # print(glo30_collection.aggregate_array("system:index").getInfo())

                # Reduce the ImageCollection by mean to get average values per band
                glo30_mosaic = glo30_collection.mosaic()
                projected_dem = glo30_mosaic.reproject(crs='EPSG:3857', scale=30)
                glo30_stats = add_slope_aspect(projected_dem).select(['Ele', 'Slope', 'Aspect']).clip(geom)


                # Check bands after reduction: print bands after reduction to confirm selection
                print("Bands after ee.Reducer:", glo30_stats.bandNames().getInfo())

                # Create a descriptive task name for export
                task_desc = f'glo30_{basename}_uid_{uid}'

                # Set up and start an export task to Google Drive   
                task = ee.batch.Export.image.toDrive(**{
                                                'image': glo30_stats,
                                                'description': task_desc,   
                                                # 'folder': '_glo30_image',  # folder on google drive
                                                'scale': 25,         # Resolution 
                                                'crs': 'EPSG:3857',  # Coordinate Reference System
                                                'maxPixels': 1e13,    # Large maxPixels to avoid export limits
                                                'fileFormat':'GeoTIFF',
                                                'formatOptions':{'noData':-9999}
                                               })

                task.start()

                print(f'\n    → task: {task_desc} submitted ')

                # Optional: sleep to avoid submitting too many tasks too quickly
                # import time
                # time.sleep(3) # sleep interval to aviod creating multi outdir in google drive

