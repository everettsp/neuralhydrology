"""Utility functions for analyzing and comparing multiple runs."""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
import pickle


from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester

def eval_all_runs(runs: pd.DataFrame|Path) -> None:
    if isinstance(runs, Path):
        runs_df = load_runs_to_dataframe(runs)
    elif isinstance(runs, pd.DataFrame):
        runs_df = runs
    else:
        raise ValueError("Input must be a pandas DataFrame or a Path to a directory containing runs.")
    
    for run_dir in runs_df["run_dir"].unique():
        run_dir = Path(run_dir)
        run_config = Config(run_dir / "config.yml")
        n_epochs = run_config.epochs
        if (run_dir / f"model_epoch{n_epochs:03d}.pt").exists():
            for period in ["train", "validation", "test"]:
                if (run_dir / period).exists():
                    print(f"skipping {run_dir} for period {period} as it already exists")
                    continue
                else:
                    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period=period, init_model=True)
                    tester.evaluate(save_results=True, metrics=run_config.metrics)
        else:
            print(f"skipping {run_dir} as model does not exist")
            continue

def load_runs_to_dataframe(run_directory: Path, 
                          config_patterns: List[str] = ["config.yml"],
                          flatten_nested: bool = True,
                          include_metadata: bool = True) -> pd.DataFrame:
    """Load configuration options from multiple runs into a pandas DataFrame.
    
    This function scans a directory for run folders, loads their configuration files,
    and creates a DataFrame where each row represents a run and columns represent
    configuration options.
    
    Parameters
    ----------
    run_directory : Path
        Path to the directory containing multiple run folders.
    config_patterns : List[str], optional
        List of config file patterns to look for in each run directory.
        Default is ["config.yml"].
    flatten_nested : bool, optional
        If True, nested configuration dictionaries will be flattened using dot notation.
        For example, {'learning_rate': {0: 0.001}} becomes {'learning_rate.0': 0.001}.
        Default is True.
    include_metadata : bool, optional
        If True, includes metadata like run directory path and config file path.
        Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame where each row represents a run and columns represent configuration
        options. The index is the run directory name.
        
    Examples
    --------
    >>> runs_df = load_runs_to_dataframe(Path("runs/"))
    >>> print(runs_df[['model', 'batch_size', 'learning_rate.0']].head())
    """
    if not run_directory.is_dir():
        raise ValueError(f"Directory {run_directory} does not exist.")
    
    run_configs = []
    failed_runs = []
    
    # Find all potential run directories
    run_dirs = [d for d in run_directory.iterdir() if d.is_dir()]
    
    for run_dir in run_dirs:
        config_found = False
        
        # Try each config pattern
        for pattern in config_patterns:
            config_files = list(run_dir.glob(pattern))
            
            if config_files:
                config_file = config_files[0]  # Take the first match
                try:
                    config = Config(config_file)
                    config_dict = config.as_dict().copy()
                    
                    # Add metadata if requested
                    if include_metadata:
                        config_dict['_run_dir'] = str(run_dir)
                        config_dict['_config_file'] = str(config_file)
                        config_dict['_run_name'] = run_dir.name
                    
                    # Flatten nested dictionaries if requested
                    if flatten_nested:
                        config_dict = _flatten_dict(config_dict)
                    
                    run_configs.append(config_dict)
                    config_found = True
                    break
                    
                except Exception as e:
                    failed_runs.append((run_dir.name, str(e)))
                    break
        
        if not config_found:
            failed_runs.append((run_dir.name, f"No config file found matching patterns: {config_patterns}"))
    
    if failed_runs:
        warnings.warn(f"Failed to load {len(failed_runs)} runs: {failed_runs}")
    
    if not run_configs:
        raise ValueError("No valid run configurations found in the directory.")
    
    # Create DataFrame with proper handling of unhashable types
    df = pd.DataFrame(run_configs)
    
    # Set index to run name if metadata is included
    if include_metadata and '_run_name' in df.columns:
        df = df.set_index('_run_name')
    
    return df


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary using dot notation.
    
    Parameters
    ----------
    d : Dict[str, Any]
        Dictionary to flatten.
    parent_key : str, optional
        Parent key for recursive calls.
    sep : str, optional
        Separator to use between nested keys.
        
    Returns
    -------
    Dict[str, Any]
        Flattened dictionary.
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # Special handling for learning_rate - keep as structured data
        if k == 'learning_rate' and isinstance(v, dict):
            # Convert learning_rate dict to a list of (epoch, lr) tuples
            lr_list = [(int(epoch), float(lr)) for epoch, lr in sorted(v.items())]
            items.append((new_key, lr_list))
        elif isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, Path):
            # Convert Path objects to strings for DataFrame compatibility
            items.append((new_key, str(v)))
        elif isinstance(v, pd.Timestamp):
            # Convert timestamps to strings for DataFrame compatibility
            items.append((new_key, v.strftime('%Y-%m-%d')))
        elif isinstance(v, list):
            # Convert lists to strings to make them hashable
            if len(v) == 0:
                items.append((new_key, "[]"))
            elif all(isinstance(item, Path) for item in v):
                # Convert list of Path objects to comma-separated string
                items.append((new_key, ",".join(str(p) for p in v)))
            elif all(isinstance(item, (str, int, float)) for item in v):
                # Convert list of simple types to comma-separated string
                items.append((new_key, ",".join(str(item) for item in v)))
            else:
                # For complex lists, convert to string representation
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    
    return dict(items)


def compare_runs(df: pd.DataFrame, 
                 columns: Optional[List[str]] = None,
                 exclude_metadata: bool = True) -> pd.DataFrame:
    """Compare specific configuration options across runs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_runs_to_dataframe.
    columns : List[str], optional
        Specific columns to compare. If None, compares all columns.
    exclude_metadata : bool, optional
        If True, excludes metadata columns (those starting with '_').
        
    Returns
    -------
    pd.DataFrame
        DataFrame with only the specified columns for comparison.
    """
    if columns is None:
        columns = df.columns.tolist()
    
    if exclude_metadata:
        columns = [col for col in columns if not col.startswith('_')]
    
    # Filter columns to only include those that exist in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    
    return df[existing_columns]


def find_varying_configs(df: pd.DataFrame, exclude_metadata: bool = True) -> List[str]:
    """Find configuration options that vary across runs.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_runs_to_dataframe.
    exclude_metadata : bool, optional
        If True, excludes metadata columns from analysis.
        
    Returns
    -------
    List[str]
        List of column names that have different values across runs.
    """
    columns = df.columns.tolist()
    
    if exclude_metadata:
        columns = [col for col in columns if not col.startswith('_')]
    
    varying_configs = []
    
    for col in columns:
        if col in df.columns:
            try:
                # Handle different data types appropriately
                unique_vals = df[col].dropna().nunique()
                if unique_vals > 1:
                    varying_configs.append(col)
            except (TypeError, ValueError):
                # If we can't compute nunique (e.g., due to unhashable types), 
                # check manually
                try:
                    non_null_values = df[col].dropna().tolist()
                    if len(set(str(v) for v in non_null_values)) > 1:
                        varying_configs.append(col)
                except:
                    # If all else fails, skip this column
                    continue
    
    return varying_configs


def summarize_runs(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a summary of the runs in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_runs_to_dataframe.
        
    Returns
    -------
    Dict[str, Any]
        Summary statistics about the runs.
    """
    summary = {
        'total_runs': len(df),
        'varying_configs': find_varying_configs(df),
        'common_configs': {},
        'config_value_counts': {}
    }
    
    # Find common configuration values
    for col in df.columns:
        if not col.startswith('_'):
            try:
                unique_vals = df[col].dropna().nunique()
                if unique_vals == 1:
                    summary['common_configs'][col] = df[col].dropna().iloc[0]
                elif unique_vals > 1:
                    try:
                        summary['config_value_counts'][col] = df[col].value_counts().to_dict()
                    except (TypeError, ValueError):
                        # For unhashable types, convert to string first
                        summary['config_value_counts'][col] = df[col].astype(str).value_counts().to_dict()
            except (TypeError, ValueError):
                # Skip problematic columns
                continue
    
    return summary

def purge_incomplete_runs(runs_dir: Path):
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not (run_dir / "model_epoch001.pt").exists():
            print(f"Purging incomplete run directory: {run_dir}")
            for item in run_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    import shutil
                    shutil.rmtree(item)
            run_dir.rmdir()



def load_results(run_dir:Path, epoch:int=-1, split:str="test"):
    """
    Load evaluation results for a specific run directory, epoch, and data split.
    Args:
        run_dir (Path): Path to the run directory containing model outputs.
        epoch (int, optional): Epoch number to load results from. If -1, loads results from the last epoch as specified in the config file. Defaults to -1.
        split (str, optional): Data split to load results for (e.g., "test", "val"). Defaults to "test".
    Returns:
        Any: Loaded results object from the specified epoch and split.
    Raises:
        ValueError: If the specified split folder does not exist in the run directory.
    """

    if not (run_dir / split).exists():
        raise ValueError(f"Run directory {run_dir} does not contain a '{split}' folder, make sure to run eval_run.")
    
    if epoch == -1:
        epoch = Config(run_dir / "config.yml").epochs

    with open(run_dir / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as f:
        results = pickle.load(f)
    return results