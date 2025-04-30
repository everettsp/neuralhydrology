from typing import List, Dict, Union

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


FORCING_FILENAMES = [
    "eccc_idw1_1d",
    "eccc_idw2_1h",
    "trca_15min",
    "camusto_1h",
    "camusto_1d",
    ]

ATTRIBUTE_FILES = ["hydromet.csv", "landuse.csv"]

def load_timeseries(data_dir:Path, basin:str, forcing:str) -> pd.DataFrame:
    if forcing.lower() not in FORCING_FILENAMES:
        raise ValueError(f"forcing {forcing} not available; choice include{FORCING_FILENAMES}")
    
    with xr.open_dataset(data_dir / f"{forcing.lower()}.nc") as ds:
        df = ds.sel(station=basin).to_dataframe().drop(columns=["station"]).copy()
        df.index.name = "date"
    return df


class CamusTO(BaseDataset):
    """Template class for adding a new data set.
    
    Each dataset class has to derive from `BaseDataset`, which implements most of the logic for preprocessing data and 
    preparing data for model training. Only two methods have to be implemented for each specific dataset class: 
    `_load_basin_data()`, which loads the time series data for a single basin, and `_load_attributes()`, which loads 
    the static attributes for the specific data set. 
    
    Usually, we outsource the functions to load the time series and attribute data into separate functions (in the
    same file), which we then call from the corresponding class methods. This way, we can also use specific basin data
    or dataset attributes without these classes.
    
    To make this dataset available for model training, don't forget to add it to the `get_dataset()` function in 
    'neuralhydrology.datasetzoo.__init__.py'

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_train : bool 
        Defines if the dataset is used for training or evaluating. If True (training), means/stds for each feature
        are computed and stored to the run directory. If one-hot encoding is used, the mapping for the one-hot encoding 
        is created and also stored to disk. If False, a `scaler` input is expected and similarly the `id_to_int` input
        if one-hot encoding is used. 
    period : {'train', 'validation', 'test'}
        Defines the period for which the data will be loaded
    basin : str, optional
        If passed, the data for only this basin will be loaded. Otherwise the basin(s) are read from the appropriate
        basin file, corresponding to the `period`.
    additional_features : List[Dict[str, pd.DataFrame]], optional
        List of dictionaries, mapping from a basin id to a pandas DataFrame. This DataFrame will be added to the data
        loaded from the dataset, and all columns are available as 'dynamic_inputs', 'evolving_attributes' and
        'target_variables'
    id_to_int : Dict[str, int], optional
        If the config argument 'use_basin_id_encoding' is True in the config and period is either 'validation' or 
        'test', this input is required. It is a dictionary, mapping from basin id to an integer (the one-hot encoding).
    scaler : Dict[str, Union[pd.Series, xarray.DataArray]], optional
        If period is either 'validation' or 'test', this input is required. It contains the centering and scaling
        for each feature and is stored to the run directory during training (train_data/train_data_scaler.yml).

    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xr.DataArray]] = {}):
        # initialize parent class
        super(CamusTO, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load basin time series data
        
        This function is used to load the time series data (meteorological forcing, streamflow, etc.) and make available
        as time series input for model training later on. Make sure that the returned dataframe is time-indexed.
        
        Parameters
        ----------
        basin : str
            Basin identifier as string.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the time series data (e.g., forcings + discharge).
        """
        """Load input and output data from text files."""

        # get forcings


        dfs = []

        #if not any(f.endswith('_15min') for f in self.cfg.forcings):
        #    raise ValueError('Forcings include no fifteen-minute forcings set.')
        for forcing in self.cfg.forcings:
            
            df = load_timeseries(self.cfg.data_dir, basin=basin, forcing=forcing)

            if forcing.lower() == "camusto_1h":
                df = df.resample('1h').ffill()
            elif forcing.lower() == "camusto_1d":
                df = df.resample('1d').ffill()
            else:
                raise ValueError(f"forcing {forcing} not available; choice include{FORCING_FILENAMES}")

            if len(self.cfg.forcings) > 1:
                # rename columns
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()})
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # collapse all input features to a single list, to check for 'QObs(mm/d)'.
        all_features = self.cfg.target_variables
        if isinstance(self.cfg.dynamic_inputs, dict):
            for val in self.cfg.dynamic_inputs.values():
                all_features = all_features + val
        elif isinstance(self.cfg.dynamic_inputs, list):
            all_features = all_features + self.cfg.dynamic_inputs


        """
        # catch also QObs(mm/d)_shiftX or _copyX features
        if any([x.startswith("QObs(mm/d)") for x in all_features]):
            # add daily discharge from CAMELS, using daymet to get basin area
            _, area = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, "daymet")
            discharge = camelsus.load_camels_us_discharge(self.cfg.data_dir, basin, area)
            discharge = discharge.resample('1H').ffill()
            df["QObs(mm/d)"] = discharge

        # only warn for missing netcdf files once for each forcing product
        self._warn_slow_loading = False
        """
        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        """Load basin attributes
        
        This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
        dataframe with features in columns.
        
        Returns
        -------
        pd.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.
        """
        return load_attributes(self.cfg.data_dir)


def load_attributes(data_dir:Path) -> pd.DataFrame:
    """Load basin attributes from csv files
    
    This function is used to load basin attribute data (e.g. CAMELS catchments attributes) as a basin-indexed 
    dataframe with features in columns.
    
    Parameters
    ----------
    data_dir : Path
        Path to the directory containing the attribute files.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    
    attribute_files = ["hydromet.csv", "landuse.csv"]

    for f in attribute_files:
        if not (data_dir / f).exists():
            raise FileNotFoundError()
        if f not in ATTRIBUTE_FILES:
            raise ValueError(f"attribute file '{f}' not available; choices include{ATTRIBUTE_FILES}")


    dfs= [pd.read_csv(data_dir / f, index_col=0) for f in attribute_files]
    for df in dfs:
        df.index.name = "TRCAID"
    df = pd.concat(dfs, axis=1)
    df = df.loc[~df.isnull().any(axis=1),:] # remove basins with missing values
    df = df.loc[:,~(df.std() < 1E-10)] # remove attributes with std of 0
    return df

