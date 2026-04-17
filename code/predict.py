#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
import xgboost as xgb
import rasterio
import numpy as np
from tqdm import tqdm
import gc
import re
print(xgb.__version__)


def build_file_dataframe(glo30_path, sentinel_path, palsar_path):

    # Extract the numeric uid from filename
    def extract_uid(filename):
        match = re.search(r'uid_(\d+)\.tif$', filename)
        return match.group(1) if match else None

    # Map uids to files in a given folder
    def get_file_dict(folder_path):
        file_dict = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.tif'):
                uid = extract_uid(filename)
                if uid:
                    file_dict[uid] = os.path.join(folder_path, filename)
        return file_dict

    # Get file dictionaries
    glo30_files = get_file_dict(glo30_path)
    sentinel_files = get_file_dict(sentinel_path)
    palsar_files = get_file_dict(palsar_path)

    # Check how many files were found in each folder
    print(f"Found {len(glo30_files)} glo30 files, {len(sentinel_files)} sentinel files, {len(palsar_files)} palsar files.")

    # Find common uids across glo30 and sentinel folders
    common_uids = set(glo30_files.keys()) & set(sentinel_files.keys())

    # Debugging: print number of common uids found
    print(f"Found {len(common_uids)} common uids.")

    # Create DataFrame from common uids
    data = []
    for uid in sorted(common_uids, key=lambda x: int(x)):
        palsar_file = palsar_files.get(uid)
        data.append({
            'uid': uid,
            'glo30_file': glo30_files[uid],
            'sentinel_file': sentinel_files[uid],
            'palsar_file': palsar_file if palsar_file else None
        })

    df = pd.DataFrame(data)
    return df


def biomass(df, model, save_dir):

    # clean data
    df_ss = df.dropna(subset=['glo30_file', 'sentinel_file', 'palsar_file'])
    display(df_ss)

    for idx, row in tqdm(df_ss.iterrows(), total=len(df_ss), desc="Processing"):

        uid = row['uid']

        print(f"\nProcessing UID: {uid}")

        outpath = f'{save_dir}{uid}_AGBD.tif'
        if os.path.exists(outpath):
            print(f"{outpath} already exist!")
            continue  # Skip existing files

        try:

            # --- GLO30 ---
            with rasterio.open(row['glo30_file']) as src:

                glo30_data = src.read() 
                print(f'→   glo30 shape:{glo30_data.shape}')
                raster_meta = src.profile.copy()  # COPY METADATA
                base_shape = src.read(1).shape

                # for i in range(1, src.count + 1):
                #     glo30_desc = src.descriptions[i - 1]
                #     print(f"\nBand {i}: {glo30_desc}")

                # Create a dictionary mapping band description to band data (2D arrays)
                glo30_bands = {desc: glo30_data[i] for i, desc in enumerate(src.descriptions)}
                # print("glo30_bands", glo30_bands)


            # Clear GLO30 data from memory
            del glo30_data
            gc.collect()


            # --- Sentinel-2 ---
            with rasterio.open(row['sentinel_file']) as src1:

                # load src1               
                sentinel_data = src1.read()
                print(f"→   sentinel2 shape:{sentinel_data.shape}")

                # loop
                # for i in range(1, src1.count + 1):
                #     band_desc = src1.descriptions[i - 1]
                #     print(f"\nBand {i}: {band_desc}")

                 # Create a dictionary mapping band description to band data (2D arrays)
                s2_bands = {desc: sentinel_data[i] for i, desc in enumerate(src1.descriptions)}

                # compute B11_range
                B11_max_array = s2_bands['B11_max']
                B11_min_array = s2_bands['B11_min']
                B11_range = np.full_like(B11_max_array, np.nan)
                np.subtract(B11_max_array, B11_min_array, out=B11_range, where=(~np.isnan(B11_max_array) & ~np.isnan(B11_min_array)))

                # add B11_range back to the s2_bands dictionary
                s2_bands['B11_range'] = B11_range         
                # print("s2_bands", s2_bands)


            # Clear Sentinel data from memory
            del sentinel_data
            gc.collect()


            # --- PALSAR2 ---
            with rasterio.open(row['palsar_file']) as src2:

                palsar_data = src2.read()
                print(f"→   palsar2 shape: {palsar_data.shape}\n")

                # for i in range(1, src2.count + 1):
                #     sar_desc = src2.descriptions[i - 1]
                #     print(f"\nBand {i}: {sar_desc}")

                 # Create a dictionary mapping band description to band data (2D arrays)
                sar_bands = {desc: palsar_data[i] for i, desc in enumerate(src2.descriptions)}

                # Calculate ranges
                HH_diss_max_array = sar_bands['HH_lee_db_int_diss_max']
                HH_diss_min_array = sar_bands['HH_lee_db_int_diss_min']
                HH_diss_range = np.full_like(HH_diss_max_array, np.nan)
                np.subtract(HH_diss_max_array, HH_diss_min_array, out=HH_diss_range, where=(~np.isnan(HH_diss_max_array) & ~np.isnan(HH_diss_min_array)))

                HV_imcorr1_max_array = sar_bands['HV_lee_db_int_imcorr1_max']
                HV_imcorr1_min_array = sar_bands['HV_lee_db_int_imcorr1_min']
                HV_imcorr1_range = np.full_like(HV_imcorr1_max_array, np.nan)
                np.subtract(HV_imcorr1_max_array, HV_imcorr1_min_array, out=HV_imcorr1_range, where=(~np.isnan(HV_imcorr1_max_array) & ~np.isnan(HV_imcorr1_min_array)))

                # add back to the s2_bands dictionary
                sar_bands['HH_diss_range'] = HH_diss_range 
                sar_bands['HV_imcorr1_range'] = HV_imcorr1_range 
                # print("sar_bands", sar_bands)

            # Clear PALSAR data from memory
            del palsar_data
            gc.collect()

            # merge dict
            features = {**glo30_bands, **s2_bands, **sar_bands}

            feature_mapping = {
                'Ele_mean': 'Ele', 'Slope_mean': 'Slope', 'B11_stdDev': 'B11_std',
                'B12_stdDev': 'B12_std', 'B5_stdDev': 'B5_std', 'EVI1_stdDev': 'EVI1_std',
                'MCARI1_stdDev': 'MCARI1_std', 'MTCI1_stdDev': 'MTCI1_std', 'NLI_stdDev': 'NLI_std',
                'HH_lee_db_int_savg_min': 'HH_savg_min', 'HV_lee_db_int_savg_min': 'HV_savg_min',
                'HH&HV_min': 'HH+HV_min'
            }

            for old_name, new_name in feature_mapping.items():
                if old_name in features:
                    features[new_name] = features.pop(old_name)

            # =============================================
            print("⚠️ Check feature/value alignment among 3-data-source and model-requirement:")
            for key, value in features.items():

                # If value is a 2D array/list, take the first row
                first_row = value[0] if len(value) > 0 else []
                # Take first 3 elements of the first row (or fewer if shorter)
                first_ = first_row[:3] if len(first_row) >= 3 else first_row
                print(f"3-data Source Feature: {key}, Values: {first_}")


            # # --- Model Prediction ---
            # # Create array list in MODEL'S feature order
            # Align features with model requirements
            feature_list = []
            for name in model.feature_names:
                if name in features:
                    feature_list.append(features[name].flatten())
                else:
                    print(f"⚠ Warning: Feature '{name}' missing! Filling with NaNs.")
                    feature_list.append(np.full(base_shape[0] * base_shape[1], np.nan))

            for name, feature_array in zip(model.feature_names, feature_list):
                first_ = feature_array[:3] if len(feature_array) >= 5 else feature_array
                print(f"Model Feature: {name}, Values: {first_}")
            # ===============================================

            feature_array = np.vstack(feature_list).T

            # Create mask for valid pixels (no NaNs across all features)
            valid_mask = ~np.any(np.isnan(feature_array), axis=1)

            # Prepare output array filled with NaNs
            preds = np.full(feature_array.shape[0], np.nan, dtype=np.float32)

            # Predict only for valid pixels
            if np.any(valid_mask):
                dtest = xgb.DMatrix(feature_array[valid_mask], feature_names=model.feature_names, missing=np.nan)
                preds_valid = model.predict(dtest)
                preds[valid_mask] = np.maximum(preds_valid, 0)  # Clamp negatives to 0

            # Reshape predictions back to raster shape
            preds = preds.reshape(base_shape)            

            # # feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
            # print(f"stacked feature shape:{feature_array.shape}")

            # dtest = xgb.DMatrix(feature_array, feature_names=model.feature_names)
            # preds = model.predict(dtest).reshape(base_shape)
            # preds = np.maximum(preds, 0)  # Replace negative values

            # --- Save Output ---
            raster_meta.update(dtype=rasterio.float32, 
                               count=1, 
                               nodata=-9999,
                               compress='lzw'
                              )
            with rasterio.open(outpath, 'w', **raster_meta) as dst:
                dst.write(preds.astype(np.float32), 1)

            print(f"uid {uid} saved!")

            # Clean up memory only if everything succeeded
            del features, feature_list, feature_array, dtest, preds, raster_meta

        except Exception as e:
            print(f"Failed {uid} on file: {row['glo30_file']}\n  Error: {str(e)}")
            print(f"Failed {uid} on file: {row['sentinel_file']}\n  Error: {str(e)}")
            print(f"Failed {uid} on file: {row['palsar_file']}\n  Error: {str(e)}")

    return 

# execution 
if __name__ == "__main__":

    # load data path
    glo30_path = f'../data/_glo30/'
    sentinel_path = f'../data/_sentinel2/'
    palsar_path = f'../data/_palsar2/'

    # build data souece df
    df_all = build_file_dataframe(glo30_path, sentinel_path, palsar_path)
    # display(df_all)

    # Load the trained model
    model = xgb.Booster()
    model.load_model(f"../run/cv/final_model_.json")

    # run
    save_dir = f"../pred/"
    biomass(df_all, model, save_dir)  

