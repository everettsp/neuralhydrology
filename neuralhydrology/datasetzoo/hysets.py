from typing import List, Dict, Union


from pathlib import Path
import pandas as pd
import xarray
import netCDF4 as nc
import numpy as np
import xarray as xr

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class Hysets(BaseDataset):
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
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # initialize parent class
        super(Hysets, self).__init__(cfg=cfg,
                                              is_train=is_train,
                                              period=period,
                                              basin=basin,
                                              additional_features=additional_features,
                                              id_to_int=id_to_int,
                                              scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        df = load_hysets_timeseries(data_dir=self.cfg.data_dir, basin=basin)

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_hysets_attributes(self.cfg.data_dir, basins=self.basins)


def official_to_hysets_id(data_dir:Path) -> dict:
    df = load_hysets_attributes(data_dir=data_dir, metadata=True, augmented=False)
    return df["Watershed_ID"].to_dict()

def hysets_to_official_id(data_dir:Path) -> dict:
    return {v: k for k, v in official_to_hysets_id(data_dir=data_dir).items()}

import geopandas as gpd

def load_hysets_boundaries(data_dir: Path) -> gpd.GeoDataFrame:
    boundaries = gpd.read_file(data_dir / "HYSETS_watershed_boundaries.zip!HYSETS_watershed_boundaries_20200730.shp")
    boundaries = boundaries.set_index("OfficialID", drop=False)
    boundaries.set_crs(epsg=4326, allow_override=True, inplace=True)
    #boundaries["geometry"] = gpd.GeoSeries.from_wkt(boundaries["geometry"])
    attributes = load_hysets_attributes(data_dir=data_dir)
    basins = gpd.GeoDataFrame(boundaries.join(attributes, how="inner"))
    return basins


def load_hysets_attributes(data_dir: Path, basins: List[str] = [], augmented:bool=True, metadata:bool=False, filter:bool=False) -> pd.DataFrame:
    """Load CAMELS GB attributes from the dataset provided by [#]_

    Parameters
    ----------
    data_dir : Path
        Path to the HYSETS directory. This folder must contain an 'attributes' file with the default name.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
        
    Raises
    ------
    FileNotFoundError
        If no subfolder called 'attributes' exists within the root directory of the CAMELS GB data set.

    References
    ----------
    .. Arsenault, R., Brissette, F., Martel, JL. et al. 
        A comprehensive, multisource database for hydrometeorological modeling of 14,425 North American watersheds. 
        Sci Data 7, 243 (2020). https://doi.org/10.1038/s41597-020-00583-2
    """



    attributes_file = data_dir / "HYSETS_watershed_properties.txt"
    if not attributes_file.exists():
        raise FileNotFoundError(f" Attribute file not found at {attributes_file}")

    df = pd.read_csv(attributes_file,index_col=3)
    df.index = [str(x) for x in df.index]
    df.index.name = "Official_ID"

    if augmented:
        additional_file = data_dir / "additional_attributes.txt"

        if not additional_file.exists():
            print('warning: cannot find augmented attribute file, using default hysets attributes')
            pass
        else:
            df_hydromet = pd.read_csv(additional_file, index_col=0, low_memory=False)
            #df_hydromet = df_hydromet.loc[:, keep_cols]
            df = df.merge(df_hydromet,left_index=True,right_index=True)
            #del keep_cols


    if not metadata:
        flags = ['flag','centroid','longitude','latitude','Watershed_ID','Source','Name']
        keep_cols = [col for col in df.columns if not any([flag.lower() in col.lower() for flag in flags])]
        df = df.loc[:,keep_cols]

    if filter:
        # if an attribute is nan for fewer then 20 basins, remove basins
        few_missing_basins = df.columns[(df.isnull().sum() > 0) & (df.isnull().sum() < 20)]
        remove_basins = []
        for attribute in few_missing_basins:
            remove_basins = remove_basins + df.index[df.loc[:,attribute].isnull()].to_list()

        remove_basins = np.unique(remove_basins)
        keep_basins = [basin not in remove_basins for basin in df.index]
        df = df.loc[keep_basins,:]

    # if an attribute is nan for 20 or more basins, remove attributes
        remove_attributes = df.columns[df.isnull().sum() >= 20]
        keep_attributes = [attribute for attribute in df.columns  if attribute not in remove_attributes]
        df = df.loc[:,keep_attributes]



    if basins:
        basins = [str(b) for b in basins]
        missing_basins = [b not in df.index for b in basins]
        if any(missing_basins):
            raise ValueError(f'Static attributes file is missing some basins ({missing_basins})')
        df = df.loc[basins]

    return df


def load_hysets_timeseries(data_dir: Path, basin: str, forcing="ERA5", version="HYSETS_2023_update") -> pd.DataFrame:
    """Load the time series data for one basin of the HYSETS data set.

    Parameters
    ----------
    data_dir : Path
        Path to the HYSETS directory. This folder must contain the HYSETS NETCDF files with default names.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    """


    
    VERSIONS = ["HYSETS_2020", "HYSETS_2023_update"]

    if version not in VERSIONS:
        raise ValueError(f"version '{version}' not available, choices include: {VERSIONS}")

    if not data_dir.is_dir():
        raise OSError(f"{data_dir} does not exist")
    
    FORCINGS = ["QC_stations","non_QCstations","SCDNA","NRCAN","Livneh","ERA5","ETA5Land"]


    nc_file = data_dir / f'{version}_{forcing}.nc'

    if forcing not in FORCINGS:
        raise ValueError(f"forcing '{forcing}' not available, choices include: {FORCINGS}")
    elif not (nc_file).exists():
        raise FileNotFoundError("forcing '{}' among available forcings, but nc file not found")


    hysets_id = official_to_hysets_id(data_dir=data_dir)[basin]

    
    with xr.open_dataset(nc_file) as f:
        watershed_ind = f.watershed[f.watershedID.values == hysets_id]
        data_vars = [var for var in f.data_vars if 'watershed' in f[var].dims and 'time' in f[var].dims]
        df = f.sel(watershed=watershed_ind)[data_vars].to_dataframe()
        #df.index = [x[1] for x in df.index]
        
    dates =  pd.date_range('1950-1-1', periods=27028, freq='D')
    df.index = dates
    
    area = load_hysets_attributes(data_dir=data_dir, basins=[basin])["Drainage_Area_km2"].values
    # convert from m3/s to mm/day
    df['discharge'] *= 86400 * 10**3 / (area * 10**6) 
    df.index.name = 'date'
    return df


