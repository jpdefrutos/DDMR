from setuptools import find_packages, setup
import os

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

with open("requirements.txt", "r") as f:
    reqs = [line.strip('\n') for line in f.readlines()]

setup(
    name='ddmr',
    py_modules=['ddmr'],
    packages=find_packages(include=['ddmr', 'ddmr.*'],
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
        'scipy',
        'scikit-image',
        'simpleITK',
        'voxelmorph==0.1',
        'pystrum==0.1',
        'tensorflow==2.13',
        'tensorflow-addons',
        'tensorflow-datasets',
        'tensorflow-metadata',
        'nibabel==3.2.1',
        'numpy',
        'h5py'
    ],
    entry_points={
        'console_scripts': ['ddmr=ddmr.main:main']
    }
)
