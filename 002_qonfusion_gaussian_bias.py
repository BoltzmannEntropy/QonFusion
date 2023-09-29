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
    initialize(version_base=None,config_path='conf', job_name="qrand_config.yaml")
    cfg = compose(config_name="qrand_config")
    return cfg




if __name__ == '__main__':
    cfg=get_conf()
    # print(OmegaConf.to_yaml(cfg))
    img_w = int(cfg.model.img_w)
    rgb_c = int(cfg.model.rgb_c)
    singlExp_val_global = int(cfg.model.singlExp_val_global)
    sample_shots = int(cfg.model.sample_shots)
    n_samples = int(cfg.model.samples)
    paper_dir=(cfg.paper.dir)
    os.makedirs(paper_dir, exist_ok=True)

    torch.set_printoptions(precision=8)


    qubit_bias_list = []

    for qub in (5, 6, 7):
        # Create a new instance of the QuantumRandomGaussianGenerator
        q = QuantumRandomGaussianGenerator(cfg, qub, False)

        # Number of samples for checking mean and standard deviation
        n_samples_check = 10000

        # Generate a large number of dx values
        dx_values = [q.forward()[0].numpy() for _ in range(n_samples_check)]
        dx_values = np.array(dx_values)

        # Compute the mean of dx values
        dx_mean = np.mean(dx_values)

        # Add the number of qubits and corresponding bias to the list
        qubit_bias_list.append((qub, dx_mean))

    # Convert the list to a DataFrame for easier viewing
    qubit_bias_df = pd.DataFrame(qubit_bias_list, columns=["Qubits", "Bias"])

    # Print the DataFrame
    print(qubit_bias_df)
