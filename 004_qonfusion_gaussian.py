import hydra
import numpy as np
import pennylane as qml
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from pennylane import numpy as np
import pennylane as qml
import pennylane.numpy as np
from datetime import datetime
import math
import cv2
from PIL import Image
import os
import glob
from PIL import Image
import io
import matplotlib.animation as animation
import pennylane as qml
import pennylane.numpy as np
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from qutip import Bloch
from torch import Tensor
from q.model import *

def get_conf():
    initialize(version_base=None,config_path='conf', job_name="qgaussian_config.yaml")
    cfg = compose(config_name="qgaussian_config")
    return cfg




if __name__ == '__main__':
    cfg=get_conf()
    print(OmegaConf.to_yaml(cfg))
    img_w = int(cfg.model.img_w)
    rgb_c = int(cfg.model.rgb_c)
    n_samples = int(cfg.model.samples)
    batch_size = int(cfg.model.batch_size)
    singlExp_val_global = int(cfg.model.singlExp_val_global)
    sample_shots = int(cfg.model.sample_shots)
    NUMSHOTS = int(cfg.model.NUMSHOTS)
    paper_dir=(cfg.paper.dir)
    os.makedirs(paper_dir, exist_ok=True)

    for qub in [3]:
        for t in [True]:
            q = QuantumRandomGaussianGenerator(cfg, qub, plotCircuit=True, useRot=t)
            q.run(n_samples)


