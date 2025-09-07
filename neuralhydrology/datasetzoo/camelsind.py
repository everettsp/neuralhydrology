import os
from functools import reduce
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from tqdm import tqdm
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config

class CamelsIND(BaseDataset):
    """Data set class for the CAMELS IND dataset by [#]_.

    For more efficient data loading during model training/evaluating, this dataset class expects the CAMELS-IND dataset
    in a processed format. Specifically, this dataset class works with per-basin csv files that contain all 
    timeseries data combined. Use the :func:`preprocess_camels_ind_dataset` function to process the original dataset 
    layout into this format.

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

    References
    ----------
    .. [#] Mangukiya, N. K., Kumar, K. B., Dey, P., Sharma, S., Bejagam, V., Mujumdar, P. P., & Sharma, A. (2025). 
        CAMELS-IND: hydrometeorological time series and catchment attributes for 228 catchments in Peninsular India. 
        Earth System Science Data, 17(2), 461-491.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        # Check for preprocessed data format before initializing the base class
        data_dir = Path(cfg.data_dir)
        preprocessed_dir = data_dir / 'preprocessed'
        attributes_file = data_dir / 'attributes.csv'

        if not preprocessed_dir.is_dir() or not attributes_file.is_file():
            error_msg = (
                f"Directory '{preprocessed_dir.name}' or file '{attributes_file.name}' not found in the "
                f"CAMELS-IND data directory: {data_dir}.\n"
                f"The CamelsIND dataset class expects the data in a preprocessed format.\n"
                "Please run the `preprocess_camels_ind_dataset` function from `neuralhydrology.datasetzoo.camelsind` "
                "to process the original raw data first.")
            raise FileNotFoundError(error_msg)

        super().__init__(cfg=cfg,
                         is_train=is_train,
                         period=period,
                         basin=basin,
                         additional_features=additional_features,
                         id_to_int=id_to_int,
                         scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from csv files."""
        return load_camels_ind_timeseries(data_dir=self.cfg.data_dir, basin=basin)

    def _load_attributes(self) -> pd.DataFrame:
        """Load static catchment attributes."""
        return load_camels_ind_attributes(data_dir=self.cfg.data_dir, basins=self.basins)


def load_camels_ind_timeseries(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the time series data for one basin of the CAMELS IND data set.
        
    Parameters
    ----------
    data_dir : Path
        Path to the processed CAMELS IND data directory. This folder must contain a folder called 'preprocessed' 
        containing the per-basin csv files created by :func:`preprocess_camels_ind_dataset`.
    basin : str
        Basin identifier number as string.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame, containing the time series data (forcings + discharge) data.
    """
    timeseries_dir = data_dir / "preprocessed"

    basin_file = timeseries_dir / f"{basin}.csv"

    if not basin_file.is_file():
        raise FileNotFoundError(f"Time series file for basin {basin} not found at {basin_file}")

    df = pd.read_csv(basin_file, index_col='date', parse_dates=['date'])

    return df


def load_camels_ind_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load CAMELS IND attributes

    Parameters
    ----------
    data_dir : Path
        Path to the processed CAMELS IND data directory. Assumes that a file called 'attributes.csv' exists.
    basins : List[str], optional
        If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all basins
        are returned.

    Returns
    -------
    pd.DataFrame
        Basin-indexed DataFrame, containing the attributes as columns.
    """
    attributes_file = data_dir / 'attributes.csv'

    df = pd.read_csv(attributes_file, index_col="gauge_id")
    df.index = df.index.astype(str)

    # Check for basins that are in the requested list but not in the attribute file.
    missing_basins = [basin for basin in basins if basin not in df.index]

    # If there are any missing basins, raise a single, consolidated error.
    if missing_basins:
        raise ValueError(f"The following {len(missing_basins)} basins are missing from the "
                         f"attribute file: {', '.join(missing_basins)}")

    return df


def _process_and_split_timeseries(base_path: Path, timeseries_output_dir: Path):
    """Processes raw time series data and splits it into per-basin files.

    Handles the aggregation of time series data (forcings, streamflow), merges
    them, converts streamflow units, and splits the result into separate CSV files
    for each basin. It also prints a summary of the available features to aid in
    model configuration.

    Parameters
    ----------
    base_path : Path
        The root directory path of the raw dataset, expected to contain
        subdirectories for 'catchment_mean_forcings',
            'streamflow_timeseries', and 'attributes_csv'.
    timeseries_output_dir : Path
        The path to the directory where the processed, per-basin time series
        CSV files will be saved. It will be created if it doesn't exist.
    """
    # --- Define paths for time series data ---
    forcings_folder = base_path / "catchment_mean_forcings"
    streamflow_file = base_path / "streamflow_timeseries" / "streamflow_observed.csv"
    topo_file = base_path / "attributes_csv" / "camels_ind_topo.csv"

    GAUGE_ID_COL = 'gauge_id'
    AREA_COL_IN_TOPO_FILE = 'cwc_area'  # The area column in the topology file (in km²)
    STREAMFLOW_COL_TO_TRANSFORM = 'streamflow'  # The original streamflow column (in m³/s)

    # --- 1. Aggregate all forcing files ---
    print("\n--- Step 1A: Aggregating Forcing Files ---")
    if not forcings_folder.is_dir():
        raise FileNotFoundError(f"Forcings folder not found at: {forcings_folder}")

    forcing_files = list(forcings_folder.glob('*.csv'))
    dataframes = []
    for file_path in tqdm(forcing_files, desc="Reading forcing files"):
        df = pd.read_csv(file_path)
        df['gauge_id'] = file_path.stem
        dataframes.append(df)

    forcings_df = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully aggregated {len(forcing_files)} forcing files.")

    # --- 2. Process Streamflow and Merge Data ---
    print("\n--- Step 1B: Processing Streamflow and Merging Data ---")
    if not streamflow_file.is_file():
        raise FileNotFoundError(f"Streamflow file not found at: {streamflow_file}")

    stream_flow_df = pd.read_csv(streamflow_file)
    df_streamflow_long = pd.melt(stream_flow_df,
                                 id_vars=['year', 'month', 'day'],
                                 var_name='gauge_id',
                                 value_name='streamflow')

    for df, name in [(forcings_df, "forcings"), (df_streamflow_long, "streamflow")]:
        original_len = len(df)
        df['gauge_id'] = pd.to_numeric(df['gauge_id'], errors='coerce')
        df.dropna(subset=['gauge_id'], inplace=True)
        if len(df) < original_len:
            print(f"Warning: Dropped {original_len - len(df)} rows from {name} data due to non-numeric gauge IDs.")
        df['gauge_id'] = df['gauge_id'].astype(int)

    merged_df = pd.merge(forcings_df, df_streamflow_long, on=['year', 'month', 'day', 'gauge_id'], how='outer')

    if 'pet(mm/day)' in merged_df.columns:
        merged_df = merged_df.drop(columns=['pet(mm/day)'])

    merged_df = merged_df.sort_values(by=['gauge_id', 'year', 'month', 'day']).reset_index(drop=True)
    print("Time series data merged and cleaned successfully.")
    # Converting the streamflow from m³/s to mm/day
    df_forcings = merged_df
    try:
        print(f"Loading topology data from '{topo_file}'...")
        df_topo = pd.read_csv(topo_file)
        print("Data loaded successfully.\n")
    except FileNotFoundError as e:
        print(f"Error: Could not find a required topography file:")

    # Ensure gauge IDs are the same data type (string) for a reliable mapping
    df_forcings[GAUGE_ID_COL] = df_forcings[GAUGE_ID_COL].astype(str)
    df_topo[GAUGE_ID_COL] = df_topo[GAUGE_ID_COL].astype(str)

    # Create a mapping dictionary: {gauge_id: area_in_square_meters}
    # The area is converted from km² to m² by multiplying by 1,000,000 (1e6)
    area_map = pd.Series(df_topo[AREA_COL_IN_TOPO_FILE].values * 1e6, index=df_topo[GAUGE_ID_COL]).to_dict()

    # Add the area in m² to the main forcings dataframe
    df_forcings['area_m2'] = df_forcings[GAUGE_ID_COL].map(area_map)

    # Check if any gauges were not found in the topology file
    missing_areas = df_forcings['area_m2'].isna().sum()
    if missing_areas > 0:
        print(
            f"Warning: {missing_areas} rows had gauge IDs not found in the topology file. Their streamflow cannot be converted."
        )

    # The formula:
    # (m³/s * 86400 s/day) -> m³/day
    # (m³/day / area_m²)   -> m/day
    # (m/day * 1000 mm/m)  -> mm/day
    df_forcings['streamflow_mm_day'] = (df_forcings[STREAMFLOW_COL_TO_TRANSFORM] * 86400 /
                                        df_forcings['area_m2']) * 1000

    # Drop the original streamflow column and the temporary area column
    df_final = df_forcings.drop(columns=[STREAMFLOW_COL_TO_TRANSFORM, 'area_m2'])

    # --- 3. Restructure into Per-Basin Files ---
    print("\n--- Step 1C: Restructuring into Per-Basin Time Series Files ---")
    timeseries_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created time series sub-directory at: {timeseries_output_dir}")

    df_final.columns = df_final.columns.str.replace(r'\(.*\)', '', regex=True).str.strip()

    df_final['date'] = pd.to_datetime({
        'year': df_final['year'],
        'month': df_final['month'],
        'day': df_final['day']
    },
                                      errors='coerce')
    df_final.dropna(subset=['date'], inplace=True)

    cols_to_drop = ['gauge_id', 'year', 'month', 'day']
    for basin_id, basin_df in tqdm(df_final.groupby('gauge_id'), desc="Writing time series files"):
        basin_df = basin_df.copy()
        basin_df.set_index('date', inplace=True)
        basin_df.drop(columns=cols_to_drop, inplace=True)
        basin_df.to_csv(timeseries_output_dir / f"{basin_id}.csv")

    # --- NEW: Print a summary of the available dynamic features ---
    print("\n--- Summary of Available Dynamic Features ---")
    # Define columns that are for utility/indexing, not features
    utility_columns = ['gauge_id', 'year', 'month', 'day', 'date']
    # Get the dynamic feature names by filtering out utility columns
    dynamic_features = [col for col in df_final.columns if col not in utility_columns]

    # Separate into likely inputs and targets for clarity
    potential_targets = ['streamflow_mm_day']
    dynamic_inputs = [feat for feat in dynamic_features if feat not in potential_targets]
    targets = [feat for feat in dynamic_features if feat in potential_targets]

    print("The following columns are now available in each per-basin CSV file.")
    print("Use these names in your NeuralHydrology config.yml file:")
    print(f"  > For 'dynamic_inputs': {dynamic_inputs}")
    if targets:
        print(f"  > For 'target_variables': {targets}")
    else:
        print("  > Warning: No standard target variable like 'streamflow' was found.")

    print(f"\nAll per-basin time series files have been stored in: {timeseries_output_dir}")


def _process_and_merge_attributes(base_path: Path, attributes_output_file: Path):
    """Merges all static attribute CSVs from a directory into a single file.

    This function scans a specified `attributes_csv` subdirectory for all files
    ending with '.csv'. It reads each file and iteratively merges them into a
    single pandas DataFrame using an outer join on the 'gauge_id' column.
    Columns that are entirely empty (all NaN) after the merge are removed.
    The final, consolidated attributes are saved to a new CSV file.

    Parameters
    ----------
    base_path : Path
        The root directory path of the dataset. This function expects it to
        contain an `attributes_csv` subdirectory.
    attributes_output_file : Path
        The full path, including filename, where the final merged attributes
        CSV file will be saved.
    """
    print("\n--- Step 2: Merging All Attribute Files ---")
    attributes_folder = base_path / 'attributes_csv'

    if not attributes_folder.is_dir():
        raise FileNotFoundError(f"Attributes folder not found at: {attributes_folder}")

    csv_files = list(attributes_folder.glob('*.csv'))
    if not csv_files:
        print("Warning: No attribute CSV files found. Skipping attribute processing.")
        return

    dataframes = [pd.read_csv(file) for file in tqdm(csv_files, desc="Reading attribute files")]

    merged_attributes_df = reduce(lambda left, right: pd.merge(left, right, on='gauge_id', how='outer'), dataframes)

    merged_attributes_df = merged_attributes_df.dropna(axis=1, how='all')
    merged_attributes_df.to_csv(attributes_output_file, index=False)
    print(f"Successfully merged {len(csv_files)} attribute files.")
    print(f"Consolidated attributes file saved to: {attributes_output_file}")



def preprocess_camels_ind_dataset(camels_base_path: str, output_dir: str):
    """Orchestrates the full preprocessing of a CAMELS-IND dataset.
    
    The raw dataset can be downloaded from `Zenodo <https://zenodo.org/records/14999580>`_.
    The output_dir is the processed CAMELS IND directory mentioned in the codes above.
    This function performs two main tasks:

    1.  Processes time series data (forcings, streamflow) and saves one CSV
        per basin into a `preprocessed` sub-directory. It will also print a
        summary of the available dynamic features for use in the config file.
    2.  Merges all static attribute files into a single `attributes.csv` file
        saved in the main output directory.

    The final folder structure will look like this::

        output_dir/
        ├── attributes.csv
        └── preprocessed/
            ├── [gauge_id_1].csv
            ├── [gauge_id_2].csv
            └── ...
            
    Parameters
    ----------
    camels_base_path : str
        The path to the root directory of the CAMELS_IND dataset.
    output_dir : str
        The path to the top-level directory where all processed files will be saved.
        This directory must not already exist.

    Raises
    ------
    FileExistsError
        If the `output_dir` already exists.
    FileNotFoundError
        If any of the required CAMELS-IND sub-folders are not found.
    """
    # --- 0. Setup and Input Validation ---
    print("--- Starting Full CAMELS-IND Preprocessing ---")
    base_path = Path(camels_base_path)
    main_output_path = Path(output_dir)

    if main_output_path.exists():
        raise FileExistsError(f"Output directory '{main_output_path}' already exists. "
                              "Please remove it or choose a different one to prevent overwriting.")

    main_output_path.mkdir(parents=True)
    print(f"Created main output directory at: {main_output_path}")

    # --- Define paths for sub-processes ---
    timeseries_output_dir = main_output_path / 'preprocessed'
    attributes_output_file = main_output_path / 'attributes.csv'

    # --- Run the processing functions ---
    _process_and_split_timeseries(base_path, timeseries_output_dir)
    _process_and_merge_attributes(base_path, attributes_output_file)

    print("\n--- Preprocessing finished successfully! ---")
