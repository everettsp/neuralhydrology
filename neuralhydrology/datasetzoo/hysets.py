from typing import List, Dict, Union


from pathlib import Path
import pandas as pd
import xarray
import netCDF4 as nc
import numpy as np


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


def load_hysets_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
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
    attributes_file = data_dir / "HYSETS_watershed_properties_AUGMENTED.txt"
    if not attributes_file.exists():
        raise FileNotFoundError(f" Attribute file not found at {attributes_file}")

    df = pd.read_csv(attributes_file,index_col=0)
    df.index = [str(x) for x in df.index]
    
    flags = ['flag','centroid','longitude','latitude']
    #keep_cols = [col for col in df.columns if not any([flag in col.lower() for flag in flags])]
    #df = df.loc[:,keep_cols]
    if basins:
        basins = [str(b) for b in basins]
        if any(b not in df.index for b in basins):
            print('shit')
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]
    return df


def load_hysets_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
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

    if not data_dir.is_dir():
        raise OSError(f"{data_dir} does not exist")

    with nc.Dataset(data_dir / 'HYSETS_2020_QC_stations.nc') as ds:
        hydromet = np.concatenate([ds[x][int(basin) == ds['watershedID'][:].data].data.reshape(-1,1) 
                    for x in ['discharge','pr','tasmax','tasmin']],axis=1)
        
    with nc.Dataset(data_dir / 'HYSETS_2020_ERA5Land_SWE.nc') as ds:
        swe = ds['swe'][int(basin) == ds['watershedID'][:].data].data.reshape(-1,1)
        
    data = np.concatenate([hydromet,swe],axis=1)
    dates =  pd.date_range('1950-1-1', periods=25202, freq='D')
    df = pd.DataFrame(index=dates,data=data, columns=['q(mm/d)','precip(mm/d)','tmax(C)','tmin(C)','swe(mm/d)'])
    area = load_hysets_attributes(data_dir=data_dir, basins=[basin])["Drainage_Area_km2"].values
    # convert from m3/s to mm/day
    df['q(mm/d)'] = df['q(mm/d)'] * 86400 * 10**3 / (area * 10**6) 
    df.index.name = 'date'
    return df



# ADDITIONAL HYSETS FUNCTIONS THAT ARE NOT REQUIRED BY BASEDATASET
import geopandas as gpd
def load_hysets_boundaries(data_dir):
    """
    Load the basin boundaries from default zipped shapefile.
    Additional Geopandas dependancy
    TODO: find an alternative to geopandas that is already included in NH
    """

    gdf = gpd.read_file(data_dir / "HYSETS_watershed_boundaries.zip")
    attributes = load_hysets_attributes(data_dir=data_dir)
    gdf = pd.merge(gdf, attributes, left_on='OfficialID',right_on='Official_ID',suffixes=('', '_duplicate'))
    gdf['Watershed_ID'] = attributes.index
    gdf = gdf.set_index('Watershed_ID', drop=False)
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf.drop(columns=[x for x in gdf.columns if '_duplicate' in x])
    return gdf


def load_static_attribute_list(data_dir):
    """
    Load the curated list of static HYSETS attributes
    These attributes must be appended to the NH config files
    """
    filename = data_dir / 'ADDITIONAL_static_attributes.txt'
    static_attribute_names = list()
    with open(filename,'r') as f:
        for item in f:
            static_attribute_names.append(item.rstrip('\n').lstrip('- '))
    return static_attribute_names

