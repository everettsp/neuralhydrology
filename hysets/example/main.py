# Imports
from pathlib import Path

import pandas as pd
import torch
import sys
sys.path.append(r"C:\Users\everett\Documents\GitHub\neuralhydrology\neuralhydrology")

from neuralhydrology.nh_run import start_run, eval_run, finetune

# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    start_run(config_file=Path("hysets.yml"))

# fall back to CPU-only mode
else:
    start_run(config_file=Path("hysets.yml"), gpu=-1)


# Load validation results from the last epoch
run_dir = Path("runs/cudalstm_hysets_limited_2401_111632/")
df = pd.read_csv(run_dir / "validation" / "model_epoch003" / "validation_metrics.csv", dtype={'basin': str})
df = df.set_index('basin')

# Compute the median NSE from all basins, where discharge observations are available for that period
print(f"Median NSE of the validation period {df['NSE'].median():.3f}")

# Select a random basins from the lower 50% of the NSE distribution
basin = df.loc[df["NSE"] < df["NSE"].median()].sample(n=1).index[0]

print(f"Selected basin: {basin} with an NSE of {df.loc[df.index == basin, 'NSE'].values[0]:.3f}")



# Add the path to the pre-trained model to the finetune config
with open("finetune.yml", "a") as fp:
    fp.write(f"base_run_dir: {run_dir.absolute()}")
    
# Create a basin file with the basin we selected above
with open("finetune_basin.txt", "w") as fp:
    fp.write(basin)


finetune(Path("finetune.yml"))
