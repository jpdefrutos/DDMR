<div align="center">
    <img src="https://user-images.githubusercontent.com/30429725/204778476-4d24c659-9287-48b8-b616-92016ffcf4f6.svg" alt="drawing" width="600">
</div>

<div align="center">

<h1 align="center">DDMR: Deep Deformation Map Registration</h1>
<h3 align="center">Train smarter, not harder: learning deep abdominal CT registration on scarce data</h3>
 
# ⚠️***WARNING: Under construction*** 

**DDMR** was developed by SINTEF Health Research. The corresponding manuscript describing the framework has been accepted for publication in [PLOS ONE](https://journals.plos.org/plosone/) and the preprint is openly available on [arXiv](https://arxiv.org/abs/2211.15717).

 
</div>

## 💻 Getting started

1. Setup virtual environment:
```
virtualenv -ppython3 venv --clear
source venv/bin/activate
```

2. Install requirements:
```
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

Use the "MultiTrain" scripts to launch the trainings, providing the neccesary parameters. Those in the COMET folder accepts a .ini configuration file (see COMET/train_config_files for example configurations).

For instance:
```
python TrainingScripts/Train_3d.py
```

## 🔍 Evaluate

Use Evaluate_network to test the trained models. On the Brain folder, use "Evaluate_network__test_fixed.py" instead.

For instance:
```
python EvaluationScripts/evaluation.py
```

## ✨ How to cite
Please, consider citing our paper, if you find the work useful:
<pre>
@misc{perezdefrutos2022ddmr,
    title = {Train smarter, not harder: learning deep abdominal CT registration on scarce data},
    author = {Pérez de Frutos, Javier and Pedersen, André and Pelanis, Egidijus and Bouget, David and Survarachakan, Shanmugapriya and Langø, Thomas and Elle, Ole-Jakob and Lindseth, Frank},
    year = {2022},
    doi = {10.48550/ARXIV.2211.15717},
    publisher = {arXiv},
    copyright = {Creative Commons Attribution 4.0 International},
    note = {preprint on arXiv at https://arxiv.org/abs/2211.15717}
}
</pre>

## ⭐ Acknowledgements
This project is based on [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library, and its related publication:
<pre>
@article{VoxelMorph2019,
    title={VoxelMorph: A Learning Framework for Deformable Medical Image Registration}, 
    author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R. and Guttag, John and Dalca, Adrian V.},
    journal={IEEE Transactions on Medical Imaging}, 
    year={2019},
    volume={38},
    number={8},
    pages={1788-1800},
    doi={10.1109/TMI.2019.2897538}
}
</pre>
