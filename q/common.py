"""
Project: QonFusion
Author: Bolzmann

This module provides common routines for the QonFusion project, including the base class for quantum random number generators.
"""

from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch import Tensor
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
# matplotlib.use('Agg')
import warnings
import pennylane.numpy as np
warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)
options = {
    'node_size': 2,
    'edge_color': 'black',
    'linewidths': 1,
    'width': 0.5
}
CMAP = cm.jet

class Q_Base_RNG(nn.Module):
    """
    Base class for quantum random number generators.

    This class should be inherited by any class that is intended to generate random numbers using quantum circuits.
    It defines the basic interface that such classes should follow.
    """

    def __init__(self) -> None:
        """
        Constructor of the base quantum random number generator class.
        """
        super(Q_Base_RNG, self).__init__()

    @abstractmethod
    def Q_sample(self) -> torch.tensor:
        """
        Abstract method to sample from the quantum circuit.

        Returns:
            A sample from the quantum circuit as a torch tensor.
        """
        raise RuntimeWarning()

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """
        Abstract method to perform a forward pass through the quantum circuit.

        Args:
            *inputs (Tensor): The input tensors.

        Returns:
            The output tensor from the forward pass.
        """
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        """
        Abstract method to compute the loss function.

        Args:
            *inputs (Any): The input data.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss as a torch tensor.
        """
        pass

    @abstractmethod
    def run(self, *inputs: Any, **kwargs) -> None:
        """
        Abstract method to run the quantum circuit.

        Args:
            *inputs (Any): The input data.
            **kwargs: Additional keyword arguments.
        """
        pass

    @property
    def nparams(self):
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params

    def print_parameters(self):
        """
        Print the parameters of the quantum variational autoencoder model.

        This method loops through the named parameters of the model, sets the requires_grad attribute to True,
        and prints the name and shape of each parameter. It also prints the total number of parameters in the model.
        """
        class_name = type(self).__name__
        for name, param in self.named_parameters():
            param.requires_grad = True
            print(name, param.data.shape)
        print("The number of parameters for QCNN:{} is {}".format(class_name, sum(
            p.numel() for p in self.parameters() if p.requires_grad)))

    def save_fig(self,fig_dir, title,dpi=300):
        """
        Helper method for saving figures.

        Args:
            fig_dir (str): The directory where the figure should be saved.
            title (str): The title of the figure.
            dpi (int, optional): The resolution of the saved image in dots per inch.

        This method adjusts the layout of the figure, checks if the provided directory exists (and creates it if not),
        and then saves the figure to the provided directory with the provided title, resolution, and transparency settings.
        """
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if fig_dir is None:
            plt.show()
        else:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(os.path.join(fig_dir, title),
                        bbox_inches='tight',
                        dpi=dpi,
                        transparent=True)
            plt.close()
        return

    def print_properties(self):
        """
        Print the properties of the quantum neural network model.

        This method prints the name of the class and the values of all of its attributes.
        """
        class_name = type(self).__name__
        print(f"Properties of {class_name}:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")
