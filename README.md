<div align="center">
    <img src="https://user-images.githubusercontent.com/30429725/204596812-c59e8fa7-3383-4c52-a7f9-1f215114e5b3.png" alt="drawing" width="600">
</div>
  
<div align="center">

<h1 align="center">DDMR: Deep Deformation Map Registration</h1>
<h3 align="center">Train smarter, not harder: learning deep abdominal CT registration on scarce data</h3>
 
# ⚠️***WARNING: Under construction*** 

**DDMR** was developed by SINTEF Health Research. A paper is submitted to X and the preprint is openly available at [arXiv](some-url-here.com).

 
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

## 🏆 Acknowledgements

## ✨ How to cite
Please, consider citing our paper, if you find the work useful:
<pre>
  @misc{frutos2022ddmr,
  author={Pérez de Frutos, Javier and Pedersen, André and Pelanis, Egidijus and Bouget, David and Survarachakan, Shanmugapriya and Langø, Thomas and Elle, Ole-Jakob and Lindseth, Frank},
  publisher = {arXiv},
  title={Train smarter, not harder: learning deep abdominal CT registration on scarce data}, 
  year={2022},
  doi = {10.48550/ARXIV.2011.06033},
  url = {https://arxiv.org/abs/2011.06033}}
</pre>

