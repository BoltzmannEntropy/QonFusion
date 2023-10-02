<h1 align="center">QonFusion-Quantum Approaches to Gaussian Random Variables: Applications in Stable Diffusion and Brownian Motion.</h1>

<h1 align="center">
  <img src="https://github.com/BoltzmannEntropy/QonFusion/blob/157ecb6487d91a658ca2afd1b7972ffcbbd5156a/static/gauss_fig_qubits_6_rotFalse.png?raw=true" width="90%"></a> 
</h1>

<p align="center">
  <a href="#About">About</a> •
  <a href="#Citation">Reproducing our results</a> •
  <a href="#Citation">Citation</a> •
</p>

## About 
Code for our paper: https://arxiv.org/abs/2309.16258

Animations are available here: boltzmannentropy.github.io/qonfusion.github.io/

<h1 align="center">
  <img src="https://github.com/BoltzmannEntropy/QonFusion/blob/157ecb6487d91a658ca2afd1b7972ffcbbd5156a/static/arch.png?raw=true" width="100%"></a> 
</h1>

## The intersting code ... 
https://github.com/BoltzmannEntropy/QonFusion/blob/main/q/model.py

## Reproducing our results 
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
