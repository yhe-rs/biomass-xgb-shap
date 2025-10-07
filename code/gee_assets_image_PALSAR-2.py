#!/usr/bin/env python
# coding: utf-8

import os
from tqdm import tqdm
from datetime import datetime, timedelta
import ee

ee.Authenticate()
ee.Initialize(project='mrv-fates')

# # PALSAR-2
def RefinedLee(img):
    """
    Applies the Refined Lee Speckle Filter to each band of an image individually.
    The image must be in natural units (not in dB).
    The processed bands will be named as 'original_band_name_lee'.

    Important:
    - The input image must be in natural units (linear scale), not in dB.
    - The output image will have the same number of bands as the input, with each band renamed
      by appending '_lee' to the original band name.

    Parameters:
    img (ee.Image): Input Earth Engine Image with one or more bands.

    Returns:
    ee.Image: Filtered image with bands named as 'original_band_name_lee'.
    """

    # Get the list of band names.
    band_names = img.bandNames()

    # Define the function to process each band.
    def per_band(bandName):
        bandName = ee.String(bandName)
        band = img.select(bandName)

        # Set up 5x5 kernels.
        weights5 = ee.List.repeat(ee.List.repeat(1, 5), 5)
        kernel5 = ee.Kernel.fixed(5, 5, weights5, 2, 2, False)

        # Compute mean and variance using the 5x5 kernel.
        mean5 = band.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=kernel5
        )
        variance5 = band.reduceNeighborhood(
            reducer=ee.Reducer.variance(),
            kernel=kernel5
        )

        # Use a sample of the 5x5 windows inside a 9x9 window to determine gradients and directions.
        sample_weights = ee.List([
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        sample_kernel = ee.Kernel.fixed(9, 9, sample_weights, 4, 4, False)

        # Calculate mean and variance for the sampled windows and store as bands.
        sample_mean = mean5.neighborhoodToBands(kernel=sample_kernel)
        sample_var = variance5.neighborhoodToBands(kernel=sample_kernel)

        # Determine the gradients for the sampled windows.
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        gradients = gradients.addBands(
            sample_mean.select(6).subtract(sample_mean.select(2)).abs()
        )
        gradients = gradients.addBands(
            sample_mean.select(3).subtract(sample_mean.select(5)).abs()
        )
        gradients = gradients.addBands(
            sample_mean.select(0).subtract(sample_mean.select(8)).abs()
        )

        # Find the maximum gradient among gradient bands.
        max_gradient = gradients.reduce(ee.Reducer.max())

        # Create a mask for pixels that have the maximum gradient.
        gradmask = gradients.eq(max_gradient)
        # Duplicate gradmask bands: each gradient represents 2 directions.
        gradmask = gradmask.addBands(gradmask)

        # Determine the 8 directions.
        directions = sample_mean.select(1).subtract(sample_mean.select(4)) \
            .gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
        directions = directions.addBands(
            sample_mean.select(6).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2)
        )
        directions = directions.addBands(
            sample_mean.select(3).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3)
        )
        directions = directions.addBands(
            sample_mean.select(0).subtract(sample_mean.select(4))
            .gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4)
        )

        # The next 4 are the Not() of the previous 4.
        directions = directions.addBands(directions.select(0).Not().multiply(5))
        directions = directions.addBands(directions.select(1).Not().multiply(6))
        directions = directions.addBands(directions.select(2).Not().multiply(7))
        directions = directions.addBands(directions.select(3).Not().multiply(8))

        # Mask all values that are not 1-8.
        directions = directions.updateMask(gradmask)

        # "Collapse" the stack into a single band image.
        directions = directions.reduce(ee.Reducer.sum())

        # Calculate local noise variance.
        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))
        sigmaV = sample_stats.toArray() \
            .arraySort() \
            .arraySlice(0, 0, 5) \
            .arrayReduce(ee.Reducer.mean(), [0])

        # Set up the 9x9 kernels for directional statistics.
        rect_weights = ee.List.repeat(ee.List.repeat(0, 9), 4) \
            .cat(ee.List.repeat(ee.List.repeat(1, 9), 5))
        diag_weights = ee.List([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        rect_kernel = ee.Kernel.fixed(9, 9, rect_weights, 4, 4, False)
        diag_kernel = ee.Kernel.fixed(9, 9, diag_weights, 4, 4, False)

        # Create stacks for mean and variance using the directional kernels.
        dir_mean = band.reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=rect_kernel
        ).updateMask(directions.eq(1))
        dir_var = band.reduceNeighborhood(
            reducer=ee.Reducer.variance(),
            kernel=rect_kernel
        ).updateMask(directions.eq(1))

        dir_mean = dir_mean.addBands(
            band.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=diag_kernel
            ).updateMask(directions.eq(2))
        )
        dir_var = dir_var.addBands(
            band.reduceNeighborhood(
                reducer=ee.Reducer.variance(),
                kernel=diag_kernel
            ).updateMask(directions.eq(2))
        )

        # Add bands for rotated kernels.
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(
                band.reduceNeighborhood(
                    reducer=ee.Reducer.mean(),
                    kernel=rect_kernel.rotate(i)
                ).updateMask(directions.eq(2 * i + 1))
            )
            dir_var = dir_var.addBands(
                band.reduceNeighborhood(
                    reducer=ee.Reducer.variance(),
                    kernel=rect_kernel.rotate(i)
                ).updateMask(directions.eq(2 * i + 1))
            )
            dir_mean = dir_mean.addBands(
                band.reduceNeighborhood(
                    reducer=ee.Reducer.mean(),
                    kernel=diag_kernel.rotate(i)
                ).updateMask(directions.eq(2 * i + 2))
            )
            dir_var = dir_var.addBands(
                band.reduceNeighborhood(
                    reducer=ee.Reducer.variance(),
                    kernel=diag_kernel.rotate(i)
                ).updateMask(directions.eq(2 * i + 2))
            )

        # "Collapse" the stack into single band images.
        dir_mean = dir_mean.reduce(ee.Reducer.sum())
        dir_var = dir_var.reduce(ee.Reducer.sum())

        # Generate the filtered value.
        varX = dir_var.subtract(
            dir_mean.multiply(dir_mean).multiply(sigmaV)
        ).divide(sigmaV.add(1.0))

        b = varX.divide(dir_var)
        result = dir_mean.add(
            b.multiply(band.subtract(dir_mean))
        )

        # Return the result with the new band name.
        result = result.arrayFlatten([['sum']]).rename(bandName.cat('_lee'))
        return result

    # Map over the band names using the per_band function.
    filtered_images = band_names.map(per_band)

    # Convert the list of images into an ImageCollection.
    filtered_collection = ee.ImageCollection(filtered_images)

    # Convert the ImageCollection to a single image.
    filtered_image = filtered_collection.toBands()

    # Collect the new band names.
    new_band_names = band_names.map(lambda name: ee.String(name).cat('_lee'))

    # Rename the bands of the filtered image.
    filtered_image = filtered_image.rename(new_band_names)

    return filtered_image



def apply_RefinedLee(image):
    """Applies the Refined Lee filter to the 'HH' and 'HV' bands.

    Parameters:
    image (ee.Image): Input image containing radar bands.

    Returns:
    ee.Image: Image containing original 'HH' and 'HV' bands plus their filtered versions
              named 'HH_lee' and 'HV_lee'.
    """
    # Get the list of bands in the image
    band_names = image.bandNames()

    # Define the target bands
    target_bands = ee.List(['HH', 'HV'])

    # Filter the target bands to those that exist in the image
    #existing_bands = target_bands.filter(lambda b: band_names.contains(b))

    # Function to process each existing band
    def process_band(band_name):
        band_name = ee.String(band_name)
        band = image.select(band_name)
        #band_natural = toNatural(band)
        band_filtered = RefinedLee(band)
        #band_filtered_db = todb(band_filtered)
        return band_filtered

    # Map the processing function over the existing bands
    filtered_bands_list = target_bands.map(process_band)

    # Convert the list of images to a single image
    filtered_bands_image = ee.ImageCollection(filtered_bands_list).toBands()

    # Add the filtered bands to the original image
    return image.addBands(filtered_bands_image).rename(['HH','HV','HH_lee','HV_lee'])



def todb(image):
    """
    Converts filtered 'HH_lee' and 'HV_lee' bands from natural units back to decibel (dB) scale.

    The conversion formula used is: 10 * log10((DN)^2- 83.0 db

    Parameters:
    image (ee.Image): Image containing 'HH_lee' and 'HV_lee' bands in natural units.

    Returns:
    ee.Image: Image with additional bands 'HH_lee_db' and 'HV_lee_db' in dB scale.
    """

    hh = image.select('HH_lee')
    hhdb = hh.pow(2).log10().multiply(10).subtract(83).rename('HH_lee_db')
    hv = image.select('HV_lee')
    hvdb = hv.pow(2).log10().multiply(10).subtract(83).rename('HV_lee_db')
    return image.addBands(hhdb).addBands(hvdb)



def toInt(image):
    """
    Converts 'HH_lee_db' and 'HV_lee_db' bands from float dB values to 16-bit integer scale.

    The conversion:
    - Scales values from the range [-50, 10] dB to [0, 65535] integer range.
    - Converts scaled values to 32-bit integers.

    Parameters:
    image (ee.Image): Image containing 'HH_lee_db' and 'HV_lee_db' bands.

    Returns:
    ee.Image: Image with additional integer bands 'HH_lee_db_int' and 'HV_lee_db_int'.
    """

    hh = image.select('HH_lee_db')
    hhint = hh.unitScale(-50, 10).multiply(65535).toUint16().rename('HH_lee_db_int')   # keep GLCM will be performed at a 16-bit gray level
    hv = image.select('HV_lee_db')
    hvint = hv.unitScale(-50, 10).multiply(65535).toUint16().rename('HV_lee_db_int')   # keep GLCM will be performed at a 16-bit gray level
    return image.addBands(hhint).addBands(hvint)



def addTexture(image, glcmsize=5):
    """
    Adds GLCM texture measures (average) to specified bands of an image.
    Applies texture calculation only to bands listed in 'bands_to_analyze'.

    Parameters:
    image (ee.Image): Input image containing bands to analyze.
    glcmsize (int): Size of the window (kernel) for GLCM calculation (default 5).

    Returns:
    ee.Image: Image with added texture bands for specified bands.
    """
    bands_to_analyze = ee.List(['HH_lee_db_int', 'HV_lee_db_int'])  # Bands to apply GLCM to

    def applyTexture(band, img):
        band = ee.String(band)
        condition = bands_to_analyze.contains(band)
        return ee.Image(ee.Algorithms.If(
            condition,
            ee.Image(img).addBands(
                ee.Image(img).select([band])
                .glcmTexture(size=glcmsize, average=True)
            ),
            img
        ))

    # Apply GLCM texture to HH and HV bands
    band_names = image.bandNames()
    textured_image = ee.Image(band_names.iterate(applyTexture, image))

    return textured_image


def HH_add_HV(image):
    """
    Adds the 'HH_lee_db' and 'HV_lee_db' bands pixel-wise and appends the result as a new band named 'HH&HV'.

    Parameters:
    image (ee.Image): Input image containing 'HH_lee_db' and 'HV_lee_db' bands.

    Returns:
    ee.Image: Image with an additional band 'HH&HV' representing the sum of 'HH_lee_db' and 'HV_lee_db'.
    """
    hh = image.select('HH_lee_db')
    hv = image.select('HV_lee_db')
    hh_add_hv = hh.add(hv).rename(['HH&HV'])   
    return image.addBands(hh_add_hv)



if __name__ == "__main__":

    # gee assets path
    parent = 'projects/mrv-fates/assets'

    # Ask EE for the list of *direct* children of that folder
    resp = ee.data.listAssets({'parent': parent})  # dict with key 'assets'
    assets_meta = resp.get('assets', [])           # list of dictionaries

    # Keep only assets whose type == 'TABLE'   (vectors/shapefiles)
    table_ids = [a['name'] for a in assets_meta if a['type'] == 'TABLE']
    print('Found', len(table_ids), 'vector assets')

     # Print all found vector asset IDs with their index
    for i, aid in enumerate(table_ids):
        print(f'{i:2d}. {aid}')

    # Loop over all vector assets
    for asset_id in tqdm(table_ids):
        #print(table_ids)

        # Filter to process only assets "fishnet"
        if not asset_id.endswith('fishnet'):
            continue  # Skip if it does not end with '_100'

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
            start = '2019-01-01'   # <-- set start date here
            end   = '2019-12-31'   # <-- set end date here

        print(f'Time range: {start} to {end}') 

        # Build one reducer that returns:  min, max, mean, median, stdDev
        stats_reducer = (ee.Reducer.minMax()               # min & max
                         .combine(ee.Reducer.mean(),    '', True)   # mean
                         .combine(ee.Reducer.median(),  '', True)   # median
                         .combine(ee.Reducer.stdDev(),  '', True))  # std

        # Get the list of features
        uids = fishnet.aggregate_array('UID').getInfo() 

        for uid in tqdm(uids):
            # Get the feature by UID
            feature = fishnet.filter(ee.Filter.eq('UID', uid)).first()

            # Example: Extract geometry and UID property
            geom = feature.geometry().getInfo()

            # Retrieve the UID value (for logging)
            uid_value = feature.get('UID').getInfo()
            print(f"\nProcessing UID: {uid_value}")


            # Build the ImageCollection filtered by spatial bounds and date range
            sar_collection = (ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
                              .filterBounds(geom)
                              .filterDate(start, end)
                              .filter(ee.Filter.And(ee.Filter.listContains('system:band_names', 'HH'), 
                                                    ee.Filter.listContains('system:band_names', 'HV')))
                              .select(['HH', 'HV'])
                              .map(apply_RefinedLee)
                              .map(todb)
                              .map(toInt)
                              .map(addTexture)
                              .map(HH_add_HV)
                              .select(['HH_lee_db_int_diss', 'HH_lee_db_int_savg', 
                                       'HV_lee_db_int_imcorr1', 'HV_lee_db_int_savg', 'HH&HV'])
                              .map(lambda image: image.toFloat())
                              # .map(lambda image: image.toDouble())
                             )
            # Check bands before reduction
            collection_size = sar_collection.size().getInfo()
            print(f"collection_size: {collection_size}")
            if collection_size == 0:
                # No images found for this feature, skip processing
                print(f"No images found for uid {uid}. Skipping...")
                continue  # skip to next chunk

            else:
                # check bands before reduction
                print("Bands before ee.Reducer:", sar_collection.first().bandNames().getInfo())

                # Reduce the ImageCollection to single image with stats_reducer
                sar_stats = (sar_collection.reduce(stats_reducer)
                                            .clip(geom)
                                            .toFloat()
                                            # .toDouble()
                                            .select('HH_lee_db_int_diss_min', 
                                                    'HH_lee_db_int_diss_max', 
                                                    'HH_lee_db_int_savg_min', 
                                                    'HV_lee_db_int_imcorr1_min', 
                                                    'HV_lee_db_int_imcorr1_max', 
                                                    'HV_lee_db_int_savg_min', 
                                                    'HH&HV_min')
                            )

                # print bands after reduction
                print("Bands after ee.Reducer:", sar_stats.bandNames().getInfo())

                # Create a descriptive task name for export
                task_desc = f'palsar2_{basename}_uid_{uid}'

                # Set up and start an export task to Google Drive
                task = ee.batch.Export.image.toDrive(**{
                                            'image': sar_stats,
                                            'description': task_desc, # set file name
                                            # 'folder': '_palsar2_image',  # folder on google drive
                                            # region=geom, 
                                            'scale': 25,
                                            'crs': 'EPSG:3857',
                                            'maxPixels': 1e13, # (Large maxPixels to avoid limit errors) The field "max_pixels" must have a value between 1 and 10000000000000 inclusive. If unspecified, the default value is 100000000.
                                            'fileFormat':'GeoTIFF',
                                            'formatOptions':{'noData':-9999}
                                            })

                # kick off
                task.start()

                print(f'\n    → task: {task_desc} submitted ')

                # Optional: sleep to avoid submitting too many tasks too quickly
                # import time
                # time.sleep(3) # sleep interval to aviod creating multi outdir in google drive


