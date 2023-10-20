from setuptools import find_packages, setup
import os

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = [line.strip('\n') for line in f.readlines()]

setup(
    name='DeepDeformationMapRegistration',
    py_modules=['DeepDeformationMapRegistration'],
    packages=find_packages(include=['DeepDeformationMapRegistration', 'DeepDeformationMapRegistration.*'],
                           exclude=['test_images', 'test_images.*']),
    version='1.0',
    description='Deep-registration training toolkit',
    # long_description=long_description,
    author='Javier Pérez de Frutos',
    classifiers=[
        'Programming language :: Python :: 3',
        'License :: OSI Approveed :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
    install_requires=[
        'fastrlock>=0.3',   # required by cupy-cuda110
        'testresources',    # required by launchpadlib
        'scipy',
        'scikit-image',
        'simpleITK',
        'voxelmorph==0.1',
        'pystrum==0.1',
        'tensorflow-gpu==1.14.0',
        'tensorflow-addons',
        'tensorflow-datasets',
        'tensorflow-metadata',
        'tensorboard==1.14.0',
        'nibabel==3.2.1',
        'numpy==1.18.5',
        'h5py==2.10'
    ],
    entry_points={
        'console_scripts': ['ddmr=DeepDeformationMapRegistration.main:main']
    }
)
