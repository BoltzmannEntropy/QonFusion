<h1 align="center">QonFusion-Quantum Approaches to Gaussian Random Variables: Applications in Stable Diffusion and Brownian Motion.</h1>

<h1 align="center">
  <img src="https://github.com/BoltzmannEntropy/QonFusion/blob/master/static/gauss_fig_qubits_6_rotFalse.png?raw=true" width="90%"></a>
  
</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#examples">Examples</a> •
  <a href="#author">Author</a> •
</p>

## About 
Code for our paper: https://arxiv.org/abs/2309.16258

Animations are available here: boltzmannentropy.github.io/qonfusion.github.io/

## Reproducing our tesults 
```python
#!/bin/sh
python3 001_qonfusion_random_rotations.py
python3 002_qonfusion_1d-diffusion.py
python3 002_qonfusion_gaussian_bias.py
python3 003_qonfusion_uniform.py
python3 004_qonfusion_gaussian.py

```

## Citation

https://arxiv.org/abs/2309.16258

```
@misc{kashani2023qonfusion,
      title={QonFusion -- Quantum Approaches to Gaussian Random Variables: Applications in Stable Diffusion and Brownian Motion}, 
      author={Shlomo Kashani},
      year={2023},
      eprint={2309.16258},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```