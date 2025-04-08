from pathlib import Path

import sys
sys.path.append(r"C:\Users\everett\Documents\GitHub\neuralhydrology")
#sys.path.append("../../neuralhydrology")

import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


start_run(config_file=Path("toronto_ar.yml"))

