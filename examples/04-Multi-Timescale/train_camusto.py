from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


import os
os.chdir(Path(__file__).resolve().parent)
#start_run(config_file=Path("toronto_mts.yml"))
start_run(config_file=Path("toronto_1h.yml"))

#start_run(config_file=Path("toronto_ar.yml"))  