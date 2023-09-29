import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pennylane.numpy as np
import torch
from scipy.spatial.distance import pdist
# from qutip import Bloch
from scipy.stats import ks_2samp, entropy
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import sklearn

import q.common

# Data = Dict[str, Union[int, float, Sequence[Union[int, float]]]]

# __all__ = ['chow_liu']


# QUBIT_BIAS_DICT = {
#     5: -0.010508,
#     6: -0.005904,
#     7: -0.000578,
#     8: 0.002273,
#     9: 0.008113
# }
#
# # Print the dictionary
# for qub, bias in QUBIT_BIAS_DICT.items():
#     print(f"Qubits: {qub}, Bias: {bias}")

"""
QonFusion Project
Author: Bolzmann

This script contains classes for various quantum random number generators
and functions for calculating metrics and statistical tests between distributions.

The quantum random number generators include:
    1. QuantumRandomRotationGenerator
    2. QuantumBrownianGenerator
    3. QuantumRandomUniformGenerator
    4. QuantumRandomGaussianGenerator

Additionally, the script contains functionality for saving figures and
calculating Maximum Mean Discrepancy (MMD), Kolmogorov–Smirnov test,
and Kullback–Leibler divergence.
"""


class QuantumRandomRotationGenerator(q.common.Q_Base_RNG):
    """
    A class used to represent a Quantum Random Rotation Generator.

    ...

    Attributes
    ----------
    n_qubits : int
        number of qubits to be used in the quantum circuit
    space_span : int
        span of the Hilbert space of the quantum system
    Q_device : pennylane.Device
        the device on which the quantum circuit will be run
    eps : float
        a small number added to avoid division by zero errors
    plotCircuit : bool
        a flag used to decide whether to plot the circuit or not
    cfg : dict
        a dictionary to hold configuration parameters
    anzats : callable
        a quantum circuit that is used to generate the quantum state
    circuit : pennylane.QNode
        a quantum node that represents the quantum circuit

    Methods
    -------
    toBinaryIndex(samples)
        Converts the qubit measurements to a binary index.
    Q_sample()
        Generates a quantum sample.
    forward()
        An alias for the Q_sample method.
    run(n_samples)
        Generates a given number of quantum samples and visualize them on a circle plot.
    """

    def __init__(self, cfg, n_qubits, plotCircuit=False):
        """
        Constructs all the necessary attributes for the QuantumRandomRotationGenerator object.

        Parameters
        ----------
            cfg : dict
                Configurations for the QuantumRandomRotationGenerator.
            n_qubits : int
                Number of qubits.
            plotCircuit : bool, optional
                Whether to plot the circuit or not (default is False).
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.space_span = 2 ** self.n_qubits
        self.Q_device = qml.device('default.qubit', wires=self.n_qubits,shots=5) # Hard coded only for generating rotations
        self.eps = 0.000001  # avoid division by zero
        self.plotCircuit=plotCircuit
        self.cfg=cfg

        # Note, the key difference between:
        # [qml.sample(qml.PauliZ(i)) for i in range(n_qubits)]
        # [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        # Is that the first one returns a sampled measurement outcome, while the second returns the expectation value:
        # qml.sample() will perform a measurement on the qubit in the Pauli-Z basis, collapsing the wavefunction and returning either 0 or 1 randomly based on the qubit state probabilities.
        # qml.expval() calculates the expectation value ⟨Z⟩ of the Pauli-Z operator on the qubit. This gives the average value we would expect to measure, without collapsing the wavefunction.
        # So in summary:
        # qml.sample() gives a random measurement sample (0 or 1).
        # qml.expval() gives the expected average value of a measurement (between 0 and 1).
        # Sampling collapses the state while expectation values allow further quantum processing.
        # The choice depends on if we want to extract a classical result or maintain coherence.
        def Q_HAD_ROT_UNI():
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            _samples = [qml.sample(qml.PauliZ(i)) 
                             for i in range(n_qubits)]
            return _samples
        self.anzats=Q_HAD_ROT_UNI

        self.circuit = qml.QNode(self.anzats, self.Q_device, interface="torch")
        if self.plotCircuit:
            fig, _ = qml.draw_mpl(self.circuit, expansion_strategy='device')()
            class_name = type(self).__name__
            fig_name = f'{class_name}_ansatz_qubits_{self.n_qubits}.png'
            self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)
            # plt.show()
    def toBinaryIndex(self, samples):
        """
        Converts the qubit measurements to a binary index.

        Parameters
        ----------
            samples : list
                Measurements of qubits.

        Returns
        -------
            value : int
                The binary index.
        """
        value = 0
        for i, sample in enumerate(samples):
            if sample == 1:
                value += 2 ** i
        return value

    def Q_sample(self):
        """
        Generates a quantum sample.

        Returns
        -------
            torch.Tensor
                The quantum sample.
        """
        u1 = self.circuit()
        bits1 = [i[0] for i in u1]
        u1 = float(self.toBinaryIndex(bits1) / self.space_span)  # zero to one
        return torch.tensor(2 * np.pi * (self.eps + u1),dtype=float)  # changed pi to 2*pi

    def forward(self):
        """
        An alias for the Q_sample method.

        Returns
        -------
            torch.Tensor
                The quantum sample.
        """
        return self.Q_sample()

    def run(self,n_samples):
        """
        Generates a given number of quantum samples and visualize them on a circle plot.

        Parameters
        ----------
            n_samples : int
                Number of samples to generate.

        Returns
        -------
            None
        """
        self.print_properties()
        samples = []
        for _ in range(n_samples):
            res = self.forward()
            samples.append(res)
        fig, ax = plt.subplots()
        x = np.cos(samples)
        y = np.sin(samples)
        ax.scatter(x, y)

        for xi, yi, si in zip(x, y, samples):
            ax.arrow(0, 0, xi, yi, head_width=0.05, head_length=0.001, fc='blue', ec='blue')
            ax.text(xi, yi, f'{si:.2f}', fontsize=12)

        circ = plt.Circle((0, 0), radius=1, edgecolor='r', facecolor='None')
        ax.add_patch(circ)

        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f'Random Rotations with {self.n_qubits} Qubits, {n_samples} Samples.')
        ax.set_xlabel('Cosine')
        ax.set_ylabel('Sine')
        fig_name = f'fig_qubits_{self.n_qubits}.png'

        self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)
        # fig_path = os.path.join(cfg.paper.dir, fig_name)
        # plt.savefig(fig_path, dpi=300)  # Set the DPI value (e.g., 300 for high quality)
        plt.close(fig)

class QuantumBrownianGenerator(q.common.Q_Base_RNG):
    """
    A class used to represent a Quantum Brownian Motion Generator.

    ...

    Attributes
    ----------
    n_qubits : int
        number of qubits to be used in the quantum circuit
    space_span : int
        span of the Hilbert space of the quantum system
    Q_device : pennylane.Device
        the device on which the quantum circuit will be run
    cfg : dict
        a dictionary to hold configuration parameters
    q_gauss : QuantumRandomGaussianGenerator
        an instance of QuantumRandomGaussianGenerator

    Methods
    -------
    Q_sample()
        Generates a quantum sample using the Gaussian generator.
    forward()
        An alias for the Q_sample method.
    run(n_samples)
        Generates a given number of quantum Brownian samples and plots the motion.
    """

    def __init__(self, cfg, n_qubits):
        """
        Constructs all the necessary attributes for the QuantumBrownianGenerator object.

        Parameters
        ----------
            cfg : dict
                Configurations for the QuantumBrownianGenerator.
            n_qubits : int
                Number of qubits.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.space_span = 2 ** self.n_qubits
        self.Q_device = qml.device('default.qubit', wires=self.n_qubits,shots=cfg.model.sample_shots)
        self.cfg=cfg
        self.q_gauss = QuantumRandomGaussianGenerator(self.cfg, self.n_qubits, False)

    def Q_sample(self):
        """
        Generates a quantum sample using the Gaussian generator.

        Returns
        -------
            torch.Tensor
                The quantum sample.
        """
        return torch.tensor (self.q_gauss.forward())

    def forward(self):
        """
        An alias for the Q_sample method.

        Returns
        -------
            torch.Tensor
                The quantum sample.
        """
        return self.Q_sample()

    def run(self, n_samples):
        """
        Generates a given number of quantum Brownian samples and plots the motion.

        Parameters
        ----------
            n_samples : int
                Number of samples to generate.

        Returns
        -------
            None
        """
        self.print_properties()

        # Create a new instance of the QuantumRandomGaussianGenerator

        dt = 0.001  # Timestep
        T = 1  # Total time
        N = int(T / dt)  # Number of timesteps

        # Initialize the position
        x = 0
        X = [x]  # Position history

        # Get the bias for the current number of qubits
        # bias = QUBIT_BIAS_DICT[qub]

        # Generate the Brownian motion
        for i in range(N):
            dx = self.q_gauss.forward()[0].numpy() # Subtract the bias from dx
            x += dx * np.sqrt(dt)
            X.append(x)

        # Reverse the process
        X_rev = X[::-1]

        # Plot the Brownian motion and its reverse
        plt.figure(figsize=(10, 6))
        plt.plot(X, color="blue", label="Forward Process")
        plt.plot(X_rev, color="green", label="Reverse Process")
        plt.title(f"Quantum Brownian Motion for {self.n_qubits} Qubits")
        plt.xlabel("Time step")
        plt.ylabel("Position")
        plt.grid(True)
        plt.legend()
        fig_name = f'brownian_fig_{self.n_qubits}.png'
        self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)

class QuantumRandomUniformGenerator(q.common.Q_Base_RNG):
    """
    A class used to represent a Quantum Random Uniform Generator.

    ...

    Attributes
    ----------
    n_qubits : int
        number of qubits to be used in the quantum circuit
    useRot : bool
        whether to use quantum rotations or not
    space_span : int
        span of the Hilbert space of the quantum system
    Q_device : pennylane.Device
        the device on which the quantum circuit will be run
    eps : float
        a small constant to avoid division by zero
    plotCircuit : bool
        whether to plot the quantum circuit or not
    cfg : dict
        a dictionary to hold configuration parameters

    Methods
    -------
    toBinaryIndex(samples)
        Converts a list of binary numbers to a decimal number.
    Q_sample(isIndex=False)
        Generates a quantum sample in the uniform distribution.
    Q_uniform_idx()
        Returns the index of the quantum uniform distribution.
    forward()
        An alias for the Q_sample method.
    run(n_samples)
        Generates a given number of quantum uniform samples and plots the histogram.
    """

    def __init__(self, cfg, n_qubits, plotCircuit=False, useRot=False):
        """
        Constructs all the necessary attributes for the QuantumRandomUniformGenerator object.

        Parameters
        ----------
            cfg : dict
                Configurations for the QuantumRandomUniformGenerator.
            n_qubits : int
                Number of qubits.
            plotCircuit : bool, optional
                Whether to plot the quantum circuit (default is False).
            useRot : bool, optional
                Whether to use quantum rotations or not (default is False).
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.useRot = useRot
        self.space_span = 2 ** self.n_qubits
        self.Q_device = qml.device('default.qubit', wires=self.n_qubits,shots=cfg.model.sample_shots)
        self.eps = 0.000001  # avoid division by zero
        self.plotCircuit=plotCircuit
        self.cfg=cfg
        self.var_init = np.random.uniform(high=2 * np.pi, size=(9, 3))

        def Q_PQC_UNIFORM_DIST_ANZATS(weights, useRot=True):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            if useRot:
                qml.Rot(weights[0], weights[1], weights[2], wires=0)
                qml.Rot(weights[3], weights[4], weights[5], wires=1)
                qml.Rot(weights[6], weights[7], weights[8], wires=2)
            _expectations = [qml.sample(qml.PauliZ(i)) 
                             for i in range(n_qubits)]
            return _expectations

        self.anzats=Q_PQC_UNIFORM_DIST_ANZATS

        self.circuit = qml.QNode(self.anzats, self.Q_device, interface="torch")
        if self.plotCircuit:

            fig, _ = qml.draw_mpl(self.circuit, expansion_strategy='device')(self.var_init)
            class_name = type(self).__name__
            fig_name = f'{class_name}_ansatz_qubits_{self.n_qubits}.png'
            self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)
            # plt.show()

    def toBinaryIndex(self, samples):
        """
        Converts a list of binary numbers to a decimal number.

        Parameters
        ----------
            samples : list
                A list of binary numbers.

        Returns
        -------
            int
                The decimal number that the binary list represents.
        """
        value = 0
        for i, sample in enumerate(samples):
            if sample == 1:
                value += 2 ** i
        return value

    def Q_sample(self, isIndex=False):
        """
        Generates a quantum sample in the uniform distribution.

        Parameters
        ----------
            isIndex : bool, optional
                Whether to return the index of the quantum sample (default is False).

        Returns
        -------
            torch.Tensor
                The quantum sample in the uniform distribution.
        """
        rot_ange_num = 9
        var_init = np.zeros(rot_ange_num)
        if self.useRot:
            rot_rng = QuantumRandomRotationGenerator(self.cfg,self.n_qubits)
            for i in range(0, len(var_init)):
                var_init[i] = rot_rng()
        u1 = self.circuit(var_init, self.useRot)
        bits1 = [i[0] for i in u1]
        if not isIndex:
            u1 = float(self.toBinaryIndex(bits1) / self.space_span)
        else:
            u1 = self.toBinaryIndex(bits1)
        return torch.tensor(u1,dtype=float)

    def Q_uniform_idx(self):
        """
        Returns the index of the quantum uniform distribution.

        Returns
        -------
            torch.Tensor
                The index of the quantum uniform distribution.
        """
        rot_ange_num = 9
        var_init = np.zeros(rot_ange_num)
        u1 = self.circuit(var_init, self.useRot)
        return u1

    def forward(self):
        """
        An alias for the Q_sample method.

        Returns
        -------
            torch.Tensor
                The quantum sample in the uniform distribution.
        """
        return self.Q_sample(isIndex=True)

    def run(self, n_samples):
        """
        Generates a given number of quantum uniform samples and plots the histogram.

        Parameters
        ----------
            n_samples : int
                Number of samples to generate.

        Returns
        -------
            None
        """
        self.print_properties()
        samples = []
        for _ in range(n_samples):
            res = self.forward()
            samples.append(res)
        fig, ax = plt.subplots()

        plt.hist(samples, bins=range(2 ** self.n_qubits + 1), alpha=0.7, rwidth=0.85, color='#607c8e', align='left')
        plt.title(f'Uniform distribution:Q={self.n_qubits},Rot={self.useRot},#={n_samples}')
        plt.xlabel('Output')
        plt.ylabel('Counts')
        plt.xticks(range(2 ** self.n_qubits))
        fig_name = f'uni_fig_qubits_{self.n_qubits}.png'

        self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)
        plt.close(fig)


class QuantumRandomGaussianGenerator(q.common.Q_Base_RNG):
    """
    A class used to represent a Quantum Random Gaussian Generator.

    ...

    Attributes
    ----------
    n_qubits : int
        number of qubits to be used in the quantum circuit
    useRot : bool
        whether to use quantum rotations or not
    space_span : int
        span of the Hilbert space of the quantum system
    Q_device : pennylane.Device
        the device on which the quantum circuit will be run
    eps : float
        a small constant to avoid division by zero
    plotCircuit : bool
        whether to plot the quantum circuit or not
    cfg : dict
        a dictionary to hold configuration parameters
    q_uniform : QuantumRandomUniformGenerator
        an instance of the QuantumRandomUniformGenerator class

    Methods
    -------
    Q_sample()
        Generates a quantum sample in the Gaussian distribution.
    forward()
        An alias for the Q_sample method.
    compute_mmd(x, y, kernel_width=None)
        Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.
    run(n_samples)
        Generates a given number of quantum Gaussian samples and compares them with classical Gaussian samples.
    """

    def __init__(self, cfg, n_qubits, plotCircuit=False, useRot=True):
        """
        Constructs all the necessary attributes for the QuantumRandomGaussianGenerator object.

        Parameters
        ----------
            cfg : dict
                Configurations for the QuantumRandomGaussianGenerator.
            n_qubits : int
                Number of qubits.
            plotCircuit : bool, optional
                Whether to plot the quantum circuit (default is False).
            useRot : bool, optional
                Whether to use quantum rotations or not (default is False).
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.useRot = useRot
        self.space_span = 2 ** self.n_qubits
        self.Q_device = qml.device('default.qubit', wires=self.n_qubits,shots=cfg.model.sample_shots)
        self.eps = 0.000001  # avoid division by zero
        self.plotCircuit=plotCircuit
        self.cfg=cfg
        self.q_uniform = QuantumRandomUniformGenerator(self.cfg, self.n_qubits, self.plotCircuit,self.useRot)

    def Q_sample(self):
        """
        Generates a quantum sample in the Gaussian distribution.
        Returns
        -------
            tuple
                The quantum sample in the Gaussian distribution.
        """
        while True:
            u1=(self.q_uniform.Q_sample())
            u2=(self.q_uniform.Q_sample())
            u = 2 * u1 - 1
            v = 2 * u2 - 1
            s = u ** 2 + v ** 2
            if s < 1 and s != 0:
                break
        factor = (-2 * np.log(s) / s) ** 0.5
        z1 = u * factor
        z2 = v * factor
        return torch.tensor(z1,dtype=float),torch.tensor(z2,dtype=float)

    def forward(self):
        """
        An alias for the Q_sample method.

        Returns
        -------
            tuple
                The quantum sample in the Gaussian distribution.
        """
        return self.Q_sample()

    def compute_mmd(self, x, y, kernel_width=None):
        """
        Computes the Maximum Mean Discrepancy (MMD) between two sets of samples.

        Parameters
        ----------
            x : numpy.array
                First set of samples.
            y : numpy.array
                Second set of samples.
            kernel_width : float, optional
                Width of the Gaussian kernel (default is None).

        Returns
        -------
            float
                The MMD between the two sets of samples.
        """
        x_kernel = pdist(x[:, None], 'sqeuclidean')
        y_kernel = pdist(y[:, None], 'sqeuclidean')
        xy_kernel = pairwise_distances(x[:, None], y[:, None], metric='sqeuclidean')
        if kernel_width is None:
            kernel_width = np.median(np.hstack((x_kernel, y_kernel, xy_kernel.flatten())))
        kernel = lambda r: np.exp(-r / kernel_width)
        return np.mean(kernel(x_kernel)) + np.mean(kernel(y_kernel)) - 2 * np.mean(kernel(xy_kernel))



    def permutation_test(self,quantum_samples, classical_samples, n_permutations=4000):
        """
        Perform a permutation test on two samples.

        Parameters
        ----------
            quantum_samples : list
                Quantum Gaussian samples.
            classical_samples : list
                Classical Gaussian samples.
            n_permutations : int, optional
                Number of permutations to perform (default is 1000).

        Returns
        -------
            p_value : float
                p-value indicating the significance of the difference.
        """
        observed_diff = np.mean(quantum_samples) - np.mean(classical_samples)
        combined_samples = np.concatenate([quantum_samples, classical_samples])
        count = 0

        for _ in range(n_permutations):
            permuted_samples = sklearn.utils.resample(combined_samples, replace=False)
            perm_quantum = permuted_samples[:len(quantum_samples)]
            perm_classical = permuted_samples[len(quantum_samples):]
            perm_diff = np.mean(perm_quantum) - np.mean(perm_classical)

            if abs(perm_diff) >= abs(observed_diff):
                count += 1

        p_value = count / n_permutations
        return p_value

    def run(self, n_samples):
        """
        Generates a given number of quantum Gaussian samples and compares them with classical Gaussian samples.

        Parameters
        ----------
            n_samples : int
                Number of samples to generate.

        Returns
        -------
            None
        """
        self.print_properties()

        quantum_samples = [z for _ in range(n_samples) for z in self.forward()]

        # Generate classical Gaussian samples
        classical_samples = np.random.normal(0, 1, n_samples)  # mean = 0, std dev = 1

        # Compute statistical tests
        ks_stat, ks_pvalue = ks_2samp(quantum_samples, classical_samples)
        mmd = self.compute_mmd(np.array(quantum_samples), np.array(classical_samples))
        smooth_const = 1e-10  # Smoothing constant for KL divergence
        kl_div = entropy(np.histogram(quantum_samples, bins=30, density=True)[0] + smooth_const,
                         np.histogram(classical_samples, bins=30, density=True)[0] + smooth_const)

        perm_pvalue = self.permutation_test(quantum_samples, classical_samples)

        # ... rest of the plotting code ...

        # Plot the distributions
        fig, ax = plt.subplots(figsize=(12, 6))  # Set the figure size here
        ax.hist(quantum_samples, bins=30, alpha=0.5, label="Quantum Samples", density=True)
        ax.hist(classical_samples, bins=30, alpha=0.5, label="Classical Samples", density=True)

        # Plot the standard normal distribution
        from scipy.stats import norm
        x = np.linspace(-3.25, 3.25, 1000)
        mu = 0
        sigma = 1
        y = norm.pdf(x, mu, sigma)
        # Mark the standard deviations
        plt.plot(x, y, label='Standard Normal Distribution', color="green")
        vertical_offset = 0.2
        for deviation in range(1, 3):
            plt.axvline(mu - deviation * sigma, color='blue', linestyle='dashed', alpha=0.5)
            plt.axvline(mu + deviation * sigma, color='blue', linestyle='dashed', alpha=0.5)
            plt.text(mu - deviation * sigma, 0.05 + vertical_offset, f'-{deviation}σ', ha='center', va='center',
                     color='blue', alpha=0.5)
            plt.text(mu + deviation * sigma, 0.05 + vertical_offset, f'+{deviation}σ', ha='center', va='center',
                     color='blue', alpha=0.5)

        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(title=f"KS-stat: {ks_stat:.3f}\nKS-pvalue: {ks_pvalue:.3f}\nMMD: {mmd:.3f}\nKL-div: {kl_div:.3f}",
                  loc="upper right")
        ax.legend(
            title=f"KS-stat: {ks_stat:.3f}\nKS-pvalue: {ks_pvalue:.3f}\nMMD: {mmd:.3f}\nKL-div: {kl_div:.3f}\nPerm-pvalue: {perm_pvalue:.3f}",
            loc="upper right")

        ax.set_title(f'Gaussian quantum vs. classical distribution:Q={self.n_qubits},Rot={self.useRot},#={n_samples}')

        # Save the plot to disk
        fig_name = f'gauss_fig_qubits_{self.n_qubits}_rot{self.useRot}.png'
        self.save_fig(self.cfg.paper.dir, fig_name, dpi=300)
        plt.close(fig)  # Close the figure to free up memory



class QuantumRandomGaussianCurruptor(q.common.Q_Base_RNG):


    def __init__(self, cfg, n_qubits, plotCircuit=False, useRot=False):

        super().__init__()
        self.n_qubits = n_qubits
        self.useRot = useRot
        self.space_span = 2 ** self.n_qubits
        self.Q_device = qml.device('default.qubit', wires=self.n_qubits,shots=cfg.model.sample_shots)
        self.plotCircuit=plotCircuit
        self.cfg=cfg
        self.q_gauss = QuantumRandomGaussianGenerator(self.cfg, self.n_qubits, self.plotCircuit,self.useRot)
        # self.q_uniform = QuantumRandomUniformGenerator(self.cfg, self.n_qubits, self.plotCircuit,self.useRot)


    def Q_sample(self):
        return self.q_gauss.forward()

    def forward(self):
        return self.Q_sample()

    def Q_corruptImageWithGaussianNoise(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (img_w, img_w)) / 255.0  # Normalize to 0-1
        img = cv2.resize(img, (self.cfg.model.img_w, self.cfg.model.img_w))  # Normalize to 0-1
        # Make a copy of the original image for later use
        original_img = img.copy()

        quantum_noise = np.zeros((img.shape[0], img.shape[0]))
        # var_init = np.random.uniform(high=2 * np.pi, size=(n_layers, n_qubits, 3))
        # Populate the array with samples from the quantum circuit
        for i in tqdm(range(img.shape[0])):
            for j in range(img.shape[0]):
                quantum_noise[i, j] = self.forward()[0]

        ratio=0.9
        quantum_noisy_img = (1.0-ratio)*img + ratio*quantum_noise
        gaussian_noise = np.random.normal(0, 0.1, img.shape)
        gaussian_noisy_img = (1.0-ratio)*img + ratio*gaussian_noise
        return original_img, quantum_noise, quantum_noisy_img, gaussian_noise, gaussian_noisy_img

    def run(self, n_samples):
        self.print_properties()

        # List to hold all image data
        image_data = []

        # Get a list of all the image files in the target directory
        image_files = glob.glob("./imgs/*.png")

        # Only use the first 10 images
        image_files = image_files[:n_samples]

        for img_path in image_files:
            original_img, quantum_noise, quantum_noisy_img, gaussian_noise, gaussian_noisy_img = self.Q_corruptImageWithGaussianNoise(img_path)
            # Save the results in a dictionary and append it to image_data
            image_data.append({
                "original_img": original_img,
                "quantum_noise": quantum_noise,
                "quantum_noisy_img": quantum_noisy_img,
                "gaussian_noise": gaussian_noise,
                "gaussian_noisy_img": gaussian_noisy_img
            })

            # Loop over each set of images in image_data
        for idx, data in enumerate(image_data):
            # Display the original image
            plt.figure(figsize=(20, 5))
            plt.suptitle(f"Image {idx+1}", fontsize=16)
            plt.subplot(1, 5, 1)
            plt.imshow(data["original_img"], cmap='gray')
            plt.colorbar()
            plt.title('Original Image')

            # Display the quantum noise
            plt.subplot(1, 5, 2)
            plt.imshow(data["quantum_noise"], cmap='gray')
            plt.colorbar()
            plt.title('Quantum Noise')

            # Display the quantum corrupted image
            plt.subplot(1, 5, 3)
            plt.imshow(data["quantum_noisy_img"], cmap='gray')

            # cv2.imwrite(self.cfg.paper.dir +'quantum_noisy_img.png',data["quantum_noisy_img"], cv2.IMREAD_GRAYSCALE)
            plt.colorbar()
            plt.title('Quantum Noisy Image')

            # Display the Gaussian noise
            plt.subplot(1, 5, 4)
            plt.imshow(data["gaussian_noise"], cmap='gray')
            plt.colorbar()
            plt.title('Gaussian Noise')

            # Display the Gaussian corrupted image
            plt.subplot(1, 5, 5)
            plt.imshow(data["gaussian_noisy_img"], cmap='gray')
            plt.colorbar()
            plt.title('Gaussian Noisy Image')

            plt.tight_layout()
            plt.show()





