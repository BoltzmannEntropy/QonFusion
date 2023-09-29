import os

from hydra import compose, initialize

# from qutip import Bloch
from q.model import *


# The only circuit which is hard coded
# rnd_rot_device = qml.device('default.qubit', wires=3, shots=3)
# @qml.qnode(rnd_rot_device)


def get_conf():
    initialize(version_base=None,config_path='conf', job_name="qrand_config.yaml")
    cfg = compose(config_name="qrand_config")
    return cfg


if __name__ == '__main__':
    cfg=get_conf()
    # print(OmegaConf.to_yaml(cfg))
    img_w = int(cfg.model.img_w)
    rgb_c = int(cfg.model.rgb_c)
    # n_qubits = int(cfg.model.n_qubits)
    singlExp_val_global = int(cfg.model.singlExp_val_global)
    sample_shots = int(cfg.model.sample_shots)

    paper_dir=(cfg.paper.dir)
    os.makedirs(paper_dir, exist_ok=True)


    n_samples = 100
    for qub in range(1,5):
        q=QuantumRandomRotationGenerator(cfg,qub,plotCircuit=True)
        q.run(200)





