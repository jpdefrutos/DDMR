from setuptools import find_packages, setup
import os

entry_points = {'console_script':['DeepDeformationMapRegistration=DeepDeformationMapRegistration.main:main']}

setup(
    name='DeepDeformationMapRegistration',
    py_modules=['DeepDeformationMapRegistration'],
    packages=find_packages(include=['DeepDeformationMapRegistration', 'DeepDeformationMapRegistration.*']),
    version='1.0',
    description='Deep-registration training toolkit',
    author='Javier Pérez de Frutos',
    classifiers=[
        'Programming language :: Python :: 3',
        'License :: OSI Approveed :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
    install_requires=[
        'tensorflow-gpu==1.14.0',
        'tensorboard==1.14.0',
        'nibabel==3.2.1',
        'numpy==1.18.5',
        'livermask'
    ]
)
