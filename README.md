<div align="center">
    <img src="https://user-images.githubusercontent.com/29090665/197514781-11c92d1e-a692-4c78-8f0f-6cb863c6f572.png" alt="drawing" width="600">
</div>

<div align="center">
<h1 align="center">DDMR: Deep Deformation Map Registration</h1>
<h3 align="center">Cross-modal transfer learning and adaptive multi-task learning for improved abdominal CT registration</h3>
 
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
  author={Pérez de Frutos, Javier and Pedersen, André and Pelanis, Egidijus and Bouget, David and Survarachakan, Shanmugapriya and Langø, Thomas and Lindseth, Frank and Elle, Ole-Jakob},
  publisher = {arXiv},
  title={Cross-modal transfer learning and adaptive multi-task learning for improved abdominal CT registration}, 
  year={2022},
  doi = {10.48550/ARXIV.2011.06033},
  url = {https://arxiv.org/abs/2011.06033}}
</pre>

