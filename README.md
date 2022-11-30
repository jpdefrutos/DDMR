<div align="center">
    <img src="https://user-images.githubusercontent.com/30429725/204596812-c59e8fa7-3383-4c52-a7f9-1f215114e5b3.png" alt="drawing" width="600">
</div>
  
<div align="center">

<h1 align="center">DDMR: Deep Deformation Map Registration</h1>
<h3 align="center">Train smarter, not harder: learning deep abdominal CT registration on scarce data</h3>
 
# ⚠️***WARNING: Under construction*** 

**DDMR** was developed by SINTEF Health Research. The corresponding manuscript describing the framework has been submitted to [IJCARS](https://www.springer.com/journal/11548) and the preprint is openly available at [arXiv](https://arxiv.org/abs/2211.15717).

 
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
  doi = {10.48550/ARXIV.2211.15717},
  url = {https://arxiv.org/abs/2211.15717},
  author = {de Frutos, Javier Pérez and Pedersen, André and Pelanis, Egidijus and Bouget, David and Survarachakan, Shanmugapriya and Langø, Thomas and Elle, Ole-Jakob and Lindseth, Frank},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences, I.4.9; I.5.4; J.3; J.6, I.4.9},
  title = {Train smarter, not harder: learning deep abdominal CT registration on scarce data},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}}
</pre>

