import isodisreg 
from isodisreg import idr

import pandas as pd
import numpy as np
import xarray as xr

import calendar

def append_string_to_array_elements(input_array, string_to_append):
    # Use a list comprehension to append the string to each element in the input array
    appended_array = [element + string_to_append for element in input_array]
    return np.asarray(appended_array)

def adjust_lon_for_wave_type(wave_type, up_down, loni, lon_step):
    # Define the mapping of wave types to longitude adjustments
    lon_adjustment_map = {
        'ER': lon_step, 'IG1': lon_step, 'MRG': lon_step, 'TD': lon_step,
        'EIG': -lon_step, 'Kelvin': -lon_step, 'MJO': -lon_step, 
    }
    
    if up_down == 'down':
        for key in lon_adjustment_map:
            lon_adjustment_map[key] *= -1


    if wave_type in lon_adjustment_map:
        # print('Adjusting loni')
        loni= loni + lon_adjustment_map[wave_type]
        return loni
    else:
        # print('Else')
        return loni

def merge_adjusted_datasets(f, up_down, wave_id_after, lati, loni, latAdjustment, lon_step):
    datasets = []
    
    wave_id_original = f.waves.values
    
    for wave_type in wave_id_original:
        dataset = f.sel(waves=wave_type)
        lon_adjusted = adjust_lon_for_wave_type(wave_type, up_down, loni, lon_step)
        # print(lon_adjusted)
        if 'variables' in dataset:
            dataset = dataset.sel(lat=lati+latAdjustment, lon=lon_adjusted).drop_vars('variables')
        else:
            dataset = dataset.sel(lat=lati+latAdjustment, lon=lon_adjusted)
        # print(dataset)
        datasets.append(dataset)

    merged = xr.concat(datasets,  dim='waves')
    merged = merged.assign_coords({'waves': wave_id_after})
    return merged

def mae(y_true, y_pred):
    
    error = y_true - y_pred
    mae = np.nanmean(abs(error))
    
    return mae

def easyUQ(obs, preds, strt, last):
    idr_preds_crps = np.empty((int(obs.shape[0]/92), 92), dtype=float)
    idr_preds_bs = np.empty((int(obs.shape[0]/92), 92), dtype=float)
    idr_preds_q05 = np.empty((int(obs.shape[0]/92), 92), dtype=float)
    idr_preds_q95 = np.empty((int(obs.shape[0]/92), 92), dtype=float)
    idrPIT = np.empty((int(obs.shape[0]/92), 92), dtype=float)

    years = np.arange(2007,2020,1)
    
    for tt, test_year in enumerate(years):
        if test_year >= 2007:
            # print(f'test_year: {test_year}')

            t_train_imerg = pd.to_datetime([])
            t_test_imerg = pd.to_datetime([])

            # train-dates
            train_years = years[years!=test_year]
            for train_year in train_years:
                t0_imerg = pd.date_range(start=strt+str(train_year), end=last+str(train_year))
                if calendar.isleap(train_year):
                    t0_imerg = t0_imerg.drop(pd.Timestamp(train_year, 2, 29), errors='ignore')
                t_train_imerg  = t_train_imerg.union(t0_imerg)

            # test-dates
            t00_imerg = pd.date_range(start=strt+str(test_year), end=last+str(test_year))
            if calendar.isleap(test_year):
                t00_imerg = t00_imerg.drop(pd.Timestamp(test_year, 2, 29), errors='ignore')
            t_test_imerg  = t_test_imerg.union(t00_imerg)
            

            pred_train = preds.sel(time=t_train_imerg)
            pred_test = preds.sel(time=t_test_imerg)

            obs_train = obs.sel(time=t_train_imerg)
            obs_test = obs.sel(time=t_test_imerg)

            fitted_idr = idr(obs_train, pd.DataFrame(pred_train))
            idr_preds_test = fitted_idr.predict(pd.DataFrame(pred_test))

            idr_preds_crps[tt, :] = np.asarray(idr_preds_test.crps(obs_test))
            idr_preds_bs[tt, :] = np.asarray(idr_preds_test.bscore(thresholds=0.2, y=obs_test))
            idr_preds_q05[tt, :] = np.asarray(idr_preds_test.qpred(quantiles=0.05))
            idr_preds_q95[tt, :] = np.asarray(idr_preds_test.qpred(quantiles=0.95))


            idrPIT[tt, :] = idr_preds_test.pit(y=obs_test, seed=42)


    crps = np.nanmean(idr_preds_crps.flatten())
    bs = np.nanmean(idr_preds_bs)
    idr_preds_q05 = idr_preds_q05.flatten()
    idr_preds_q95 = idr_preds_q95.flatten()
    return crps, idr_preds_q05, idr_preds_q95, idrPIT, bs 
