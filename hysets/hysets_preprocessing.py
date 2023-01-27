


#%%
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import warnings
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import xarray as xr

import sys
sys.path.append(r"C:\Users\everett\Documents\GitHub\neuralhydrology\neuralhydrology")
from neuralhydrology.datasetzoo import hysets

data_dir = Path(r"C:\Users\everett\SynologyDrive\LSH\hysets")

def sequence_duration(condition):
    condition = condition.values.flatten()
    if condition.sum() == 0:
        z = 0
    else:
        z = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2].mean()
    return z

def baseflow_lyne_hollick(tt, alpha=.925, direction='f'):
        """ 
        Recursive digital filter for baseflow separation. Based on Lyne and Hollick, 1979.
        https://github.com/hydrogeog/pywws
        series : array of discharge measurements\n
        alpha : filter parameter\n
        direction : (f)orward or (r)everse calculation
        """
        tt = tt[tt.first_valid_index():tt.last_valid_index()].copy()
        Q = tt.values
        f = np.zeros(len(Q))
        if direction[0] == 'f':
            for t in np.arange(1,len(Q)):
                # algorithm
                f[t] = alpha * f[t-1] + (1 + alpha)/2 * (Q[t] - Q[t-1])
                # to prevent negative values
                if Q[t] - f[t] > Q[t]:
                    f[t] = 0
        elif direction[0] == 'r':
            for t in np.arange(len(Q)-2, 1, -1):
                f[t] = alpha * f[t+1] + (1 + alpha)/2 * (Q[t] - Q[t+1])
                if Q[t] - f[t] > Q[t]:
                    f[t] = 0
        # adds the baseflow to self variables so it can be called recursively
        bflow = np.array(Q - f)
        # calls method again if multiple passes are specified
        if len(direction) > 1:
            baseflow_lyne_hollick(alpha, direction=direction[1:])
        return bflow

def calculate_hydromet_attributes(tt):
    x = dict()
    
    tt_interp = tt.copy()
    
    # some parameters require long stretches of continuous data - here we create a continuous copy of the timeseries' using linear interpolation
    for col in tt.columns:    
        tt_col = tt[col].copy()
        tt_adjusted = tt_col[tt_col.first_valid_index():tt_col.last_valid_index()].copy()
        tt_adjusted.interpolate(method='linear', inplace=True)
        tt_col[tt_col.first_valid_index():tt_col.last_valid_index()] = tt_adjusted
        
        tt_interp[col] = tt_col
        del tt_col
    
    if tt_adjusted.shape[0] < 365:
        warnings.warn('insufficient data to calculate hymet stats, returning empty dict.')
        return x
    
    x['precip_mean(mm/d)'] = tt['precip(mm/d)'].mean()
    x['precip_high(mm/d)'] = x['precip_mean(mm/d)'] * 5
    x['precip_low(mm/d)'] = 1
    x['precip_mean(mm/mo)'] = tt_interp['precip(mm/d)'].resample('M').sum().mean()
    x['precip_mean(mm/y)'] = tt_interp['precip(mm/d)'].resample('Y').sum().mean()
    
    days_in_year = np.array([pd.Timestamp(year, 12, 31).dayofyear for year in tt.index.year])
    
    x['precip_high_freq(d/yr)'] = np.sum((tt['precip(mm/d)']  > x['precip_high(mm/d)']) * (1/days_in_year)) 
    x['precip_high_dur(d)'] = sequence_duration(tt_interp['precip(mm/d)'] > x['precip_high(mm/d)'])
    x['precip_low_freq(d/yr)'] = np.sum((tt_interp['precip(mm/d)'] < x['precip_low(mm/d)']) * (1/days_in_year))
    x['precip_low_dur(d)'] = sequence_duration(tt_interp['precip(mm/d)'] < x['precip_low(mm/d)'])
    
    bf = baseflow_lyne_hollick(tt_interp.loc[:,'q(mm/d)'])
    x['baseflow_index(-)'] = np.mean(bf) / tt_interp.mean().values[0]
    
    x['q_mean(mm/d)'] = tt_interp['q(mm/d)'].mean()
    x['q_high(mm/d)'] = x['q_mean(mm/d)'] * 9
    x['q_low(mm/d)'] = x['q_mean(mm/d)'] * 0.2
    x['q_high_freq(d/yr)'] = np.sum(tt_interp['q(mm/d)'] > x['q_high(mm/d)'])
    x['q_high_dur(d)'] = sequence_duration(tt_interp['q(mm/d)'] > x['q_high(mm/d)'])
    x['q_low_dur(d)'] = sequence_duration(tt_interp['q(mm/d)'] < x['q_low(mm/d)'])
    x['q_zero_freq(d/yr)'] = np.sum(tt_interp['q(mm/d)'] == 0)
    
    x['q_95'] = tt_interp['q(mm/d)'].quantile(0.95)
    x['q_5'] = tt_interp['q(mm/d)'].quantile(0.05)
    
    # only grab rows where neither discharge or precip are NaN
    qp = tt.loc[~np.any(tt[['q(mm/d)','precip(mm/d)']].isnull(),axis=1),
           ['q(mm/d)','precip(mm/d)']].sum()
    
    x['runoff_ratio'] = qp['q(mm/d)'].sum() / qp['precip(mm/d)'].sum()
    del qp
    
    result = adfuller(tt['q(mm/d)'].dropna())
    x['q_adf(-)'] = result[0]
    x['q_adf_p(-)'] = result[1]
    x['q_adf_cv1(-)'] = result[4]['1%']
    x['q_adf_cv5(-)'] = result[4]['5%']
    x['q_adf_cv10(-)'] = result[4]['10%']
    
    # need at lest 2 years of data for seasonal decomposition
    x['q_seasonality(-)'] = np.nan
    if tt['q(mm/d)'].dropna().shape[0] >= 730:
        result = seasonal_decompose(tt['q(mm/d)'].dropna(), model='additive', period=365)
        x['q_seasonality(-)'] = np.array([0,1 - result.resid.var()/(result.seasonal + result.resid).var()]).max()
    
    tt['tmean(C)'] = tt[['tmax(C)','tmin(C)']].mean(axis=1)
    x['tmean_seasonality(-)'] = np.nan
    if tt['tmean(C)'].dropna().shape[0] >= 730:
        result = seasonal_decompose(tt['tmean(C)'].dropna(), model='additive', period=365)
        x['tmean_seasonality(-)'] = np.array([0,1 - result.resid.var()/(result.seasonal + result.resid).var()]).max()
    x['tmean_alltime(C)'] = tt['tmean(C)'].mean()
    x['tmax_annual_mean(C)'] = tt.loc[:,'tmax(C)'].resample('Y').max().mean() # annual minimum, then average annual min - doing straight min gives all time low, which doesn't vary much cross canada
    x['tmin_annual_mean(C)'] = tt.loc[:,'tmin(C)'].resample('Y').min().mean()

    return x


# %%



def data_by_year(df:pd.DataFrame) -> pd.DataFrame:
    nans_in_year = df.isnull().groupby(df.index.year).sum()
    days_in_year = np.array([pd.Timestamp(year, 12, 31).dayofyear for year in nans_in_year.index])
    return 1 - nans_in_year / days_in_year.reshape(-1,1)

def preprocess_hysets(data_dir:Path,basins:list=[]):
    k = 0 # conter for data availability xarray
    df =  pd.DataFrame(index=basins) # df for hydromet attributes

    # loop through each basin
    
    for basin in tqdm(basins):

        # calculate amount of available data per year (to filter incomplete basins later)
        ts = hysets.load_hysets_timeseries(data_dir, basin)
        ts_annual = data_by_year(ts)

        if basin is basins[0]:
            arr = np.zeros((ts_annual.shape[0],ts_annual.shape[1],len(basins)))
        arr[:,:,k] = ts_annual.values
        k = k + 1
        xarr = xr.DataArray(data=arr,coords={'year':ts_annual.index.values,'param':ts_annual.columns,'basin':basins},dims=['year','param','basin'])
    

        # calculate hydromet attributes
        hydromet_attributes = calculate_hydromet_attributes(ts)
        df.loc[basin,hydromet_attributes.keys()] = hydromet_attributes.values()

    xarr.to_netcdf(data_dir / 'ADDITIONAL_data_availability_by_year.nc')
    df.to_csv(data_dir / 'ADDITIONAL_hydromet_properties.csv')

    return df, xarr



# %%


df = pd.read_csv(data_dir / "HYSETS_watershed_properties.txt",index_col=0)
df = df[df['Source'] == 'HYDAT']
preprocess_hysets(data_dir=data_dir, basins=df.index.tolist())


# %%


# %%


def augment_hysets_attributes(data_dir:Path, additional_attribute_files:list) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "HYSETS_watershed_properties.txt",index_col=0)
    for filename in additional_attribute_files:
        df_additional = pd.read_csv(data_dir / filename)
        df = df.merge(df_additional,left_index=True,right_index=True)
    return df

df = augment_hysets_attributes(data_dir=data_dir,additional_attribute_files=['ADDITIONAL_hydromet_properties.csv'])
df.to_csv(data_dir / 'HYSETS_watershed_properties_AUGMENTED.txt')


with open(data_dir / 'ADDITIONAL_static_attribute_list.txt','w') as file:
    file.write('\n- '.join(list(df.columns)))

with open(data_dir / 'ADDITIONAL_basin_list.txt','w') as file:
    file.write('\n'.join([str(x) for x in df.index]))



# %%
