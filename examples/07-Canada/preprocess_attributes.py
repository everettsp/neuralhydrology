import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import calendar
from tqdm import tqdm

def calc_fdc_slope(q):
    fdc = np.sort(q.dropna())[::-1]
    #exceedence_prob = np.arange(1, len(fdc) + 1) / (len(fdc) + 1)
    fdc_33 = np.percentile(fdc, 33)
    fdc_66 = np.percentile(fdc, 66)
    fdc_slope = (np.log(fdc_66) - np.log(fdc_33)) / (0.33)
    return fdc_slope


def sequence_duration(condition):
    """
    :param condition: A boolean array or pandas Series where True indicates the condition of interest.
    :type condition: pd.Series or np.ndarray
    :return: The average duration of consecutive True values in the input array. If there are no True values, returns 0.
    :rtype: float
    Example:
        >>> import numpy as np
        >>> condition = np.array([True, True, False, True, True, True, False])
        >>> sequence_duration(condition)
        2.0
    """
    condition = condition.values.flatten()
    if condition.sum() == 0:
        z = 0
    else:
        z = np.diff(np.where(np.concatenate(([condition[0]],condition[:-1] != condition[1:],[True])))[0])[::2].mean()
    return z

cfg = Config("hysets_basins.yml")

from neuralhydrology.datasetzoo import hysets as hs

basins = cfg.basins
res = {}
filename = cfg.data_dir / "additional_attributes.txt"

for basin in tqdm(basins):
    x = dict()

    tt = hs.load_hysets_timeseries(data_dir=cfg.data_dir, basin=basin)


    frequency = "D"  # Daily frequency

    if frequency not in ["D", "H"]:
        raise ValueError("Unsupported frequency. Use 'D' for daily or 'H' for hourly.")



    if frequency == "D":
        hourly_multiplier = 1
    elif frequency == "H":
        hourly_multiplier = 24

    #def hystats(tt, area_km2=np.nan) -> dict:

    # tt.interpolate(method='linear', inplace=True)
    # tt = tt.loc[tt.first_valid_index():tt.last_valid_index(),:]
    tt = tt.ffill()
    tt = tt.dropna()
    #tt_interp = tt.copy()

    if frequency == "D":
        tt_daily = tt.copy()
    elif frequency == "H":
        tt_daily = pd.merge(tt.loc[:,[any(w in x for w in sum_params) for x in tt.columns]].resample('D').sum(min_count=12),
                        tt.loc[:,[all(w not in x for w in sum_params) for x in tt.columns]].resample('D').mean(),left_index=True,right_index=True)


    sum_params = ["total_precipitation"]




    #tt_daily_max = tt.resample('D').max()


    tt_monthly = pd.merge(tt.loc[:,[any(w in x for w in sum_params) for x in tt.columns]].resample('ME').sum(min_count=int(30 * 0.8 * hourly_multiplier)),
                    tt.loc[:,[all(w not in x for w in sum_params) for x in tt.columns]].resample('ME').mean(),left_index=True,right_index=True)


    #tt_monthly['PET(mm)'] = thornthwaite(tt_monthly['temperature(C)'], stations.loc[stn, 'Lat'])


    tt_annual = pd.merge(tt.loc[:,[any(w in x for w in sum_params) for x in tt.columns]].resample('YE').sum(min_count=int(365 * 0.8 * hourly_multiplier)),
                    tt.loc[:,[all(w not in x for w in sum_params) for x in tt.columns]].resample('YE').mean(),left_index=True,right_index=True)

    #for col in tt.columns:
    #    tt_col = tt[col].copy()
    #    tt_adjusted = tt_col[tt_col.first_valid_index():tt_col.last_valid_index()].copy()
    #    tt_adjusted.interpolate(method='linear', inplace=True)
    #    tt_col[tt_col.first_valid_index():tt_col.last_valid_index()] = tt_adjusted
    #    
    #    tt_interp[col] = tt_col
    #    del tt_col, tt_adjusted

    x['p_mean'] = tt_daily['total_precipitation'].mean()

    x['precip_high'] = x['p_mean'] * 5
    x['precip_low'] = 1

    x['precip_mean_monthly'] = tt_monthly['total_precipitation'].mean()#tt_interp['total_precipitation'].resample('ME').sum(min_count=25*24).mean()
    x['precip_mean_annual'] = tt_annual['total_precipitation'].mean()#tt_interp['total_precipitation'].resample('YE').sum(min_count=330*).mean()

    x['high_prec_freq'] = np.sum(tt_daily['total_precipitation'] > x['precip_high']) / len(tt_daily) / 365
    x['high_prec_dur'] = sequence_duration(tt_daily['total_precipitation'] > x['precip_high'])

    x['low_prec_freq'] = np.sum(tt_daily['total_precipitation'] < x['precip_low']) / len(tt_daily) / 365
    x['low_prec_dur'] = sequence_duration(tt_daily['total_precipitation'] < x['precip_low'])

    if tt["total_precipitation"].dropna().shape[0] >= 365*2.1 * hourly_multiplier:
        result = seasonal_decompose(tt['total_precipitation'].dropna(), model='additive', period=365 * hourly_multiplier)
        x['p_seasonality'] = np.array([0,1 - result.resid.var()/(result.seasonal + result.resid).var()]).max()
    else:
        x['p_seasonality'] = np.nan
        

    """
    bf = basestage_lyne_hollick(tt_daily.loc[:,'stage(m)'])
    x['basestage_index'] = np.mean(bf) / tt_daily.mean().values[0]


    x['stage_mean'] = tt_daily['stage(m)'].mean()
    x['stage_high'] = x['stage_mean'] * 9
    x['stage_low'] = x['stage_mean'] * 0.2

    x['high_stage_freq'] = np.sum(tt_daily['stage(m)'] > x['stage_high'])
    x['high_stage_dur'] = sequence_duration(tt_daily['stage(m)'] > x['stage_high'])

    x['low_stage_dur'] = sequence_duration(tt_daily['stage(m)'] < x['stage_low'])

    x['zero_stage_freq'] = np.sum(tt_daily['stage(m)'] == 0)

    x['stage95'] = tt_interp['stage(m)'].quantile(0.95)
    x['stage5'] = tt_interp['stage(m)'].quantile(0.05)

    result = adfuller(tt['stage(m)'].dropna())
    x['stage_adf'] = result[0]
    x['stage_adf_p'] = result[1]
    # x['stage_adf'] = result[0]
    # x['stage_adf'] = result[0]
    x['stage_adf_cv1'] = result[4]['1%']
    x['stage_adf_cv5'] = result[4]['5%']
    x['stage_adf_cv10'] = result[4]['10%']

    if tt['stage(m)'].dropna().shape[0] >= 365*24*4*2.1:
        result = seasonal_decompose(tt['stage(m)'].dropna(), model='additive', period=365*24*4)
        x['stage_seasonality'] = np.array([0,1 - result.resid.var()/(result.seasonal + result.resid).var()]).max()
    else:
        x['stage_seasonality'] = np.nan
    """


    x['q_mean'] = tt_daily['discharge'].mean()
    x['q_high'] = x['q_mean'] * 9
    x['q_low'] = x['q_mean'] * 0.2

    x['high_q_freq'] = np.sum(tt_daily['discharge'] > x['q_high']) / len(tt_daily) / 365
    x['high_q_dur'] = sequence_duration(tt_daily['discharge'] > x['q_high'])

    x['low_q_dur'] = sequence_duration(tt_daily['discharge'] < x['q_low'])
    x['zero_q_freq'] = np.sum(tt_daily['discharge'] == 0) / len(tt_daily) / 365


    x["fdc_slope"] = calc_fdc_slope(tt_daily['discharge'])


    x['q95'] = tt['discharge'].quantile(0.95)
    x['q5'] = tt['discharge'].quantile(0.05)

    #area_km2 = np.nan

    area_km2 = attributes.loc[basin,"Drainage_Area_km2"]
    area_m = area_km2 * 1000000
    tt["discharge(mm)"] = tt["discharge"] * 3600 * 24 * 1000 / area_m

    if area_km2 is not np.nan:   
        
        # select stage and precip
        qp = tt.loc[:,['discharge(mm)','total_precipitation']].copy()
        # remove timesteps with any nans
        #qp = qp.loc[np.all(~np.isnan(qp),axis=1),:]
        #qp['discharge'] = qp['discharge'] * 3600 * 24 * 1000 / area_m
        x['runoff_ratio'] = qp['discharge(mm)'].sum() / qp['total_precipitation'].sum()
        del qp

    #result = adfuller(tt['discharge'].dropna())
    #x['q_adf'] = result[0]
    #x['q_adf_p'] = result[1]
    # x['stage_adf'] = result[0]
    # x['stage_adf'] = result[0]
    #x['q_adf_cv1'] = result[4]['1%']
    #x['q_adf_cv5'] = result[4]['5%']
    #x['q_adf_cv10'] = result[4]['10%']


    #if tt['temperature(C)'].dropna().shape[0] >= 365*24*4*2.1:
    #    result = seasonal_decompose(tt['temperature(C)'].dropna(), model='additive', period=365*24*4)
    #    x['temp_seasonality'] = np.array([0,1 - result.resid.var()/(result.seasonal + result.resid).var()]).max()

    #else:
    #    x['temp_seasonality'] = np.nan


    tt_monthly['days_in_month'] = tt_monthly.index.map(lambda date: calendar.monthrange(date.year, date.month)[1])
    x["pet_mean"] = (tt_monthly['evaporation'].mean() / tt_monthly['days_in_month']).mean()

    x["aridity"] = x["pet_mean"] / x['precip_mean_monthly'] if x['precip_mean_monthly'] > 0 else np.nan
    x["frac_snow"] = tt["total_precipitation"].loc[tt["2m_tasmin"] < 0].sum() / tt["total_precipitation"].sum() if tt["total_precipitation"].sum() > 0 else np.nan
    tt["2m_tasmid"] = (tt['2m_tasmax'] + tt['2m_tasmax'])/2
    x['temp_mean'] = tt["2m_tasmid"].mean()
    res[basin] = x


    if basin == basins[0]:
        with open(filename, "w") as f:
            f.write("basin," + ",".join(x.keys()) + "\n")

    with open(filename, "a") as f:
        f.write(f"{basin}," + ",".join([str(x[k]) for k in x.keys()]) + "\n")


df_hydromet_attr = pd.DataFrame.from_dict(res, orient='index')