from setuptools import setup, find_packages
import os

os.system("conda install -c anaconda cudatoolkit pyqt pywin32")

MAJOR = 0
MINOR = 0
PATCH = 1
VERSION = "{}.{}.{}".format(MAJOR, MINOR, PATCH)

def readme():
    with open('README.rst') as f:
        return f.read()

with open("pecebl/version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))

setup(
    name = "pecebl",
    version = VERSION,
    description = "eBeam Lithography simulation and Proximity Effect Correction",
    long_description = readme(),
    license = "GPLv3",
    author = "Luan Nguyen",
    author_email = "looninho@gmail.com",
    url = "http://github.com/looninho/pecebl",
    packages = ['pecebl',],
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'pecebl = scripts.cli_demo:main',
        ],
    },
    install_requires=[
        'numpy','scipy', 'sympy', 'numba', 'matplotlib', 
        'ipython==7.8.0', 'jupyter', 'pandas', 'pyculib', 
        'imageio', 'pyqtgraph', 'pyopengl', 'pycuda', 'scikit-cuda',
    ],
    dependency_links=[
        "https://anaconda.org/anaconda/cudatoolkit",
        "https://anaconda.org/anaconda/pyqt",
        "https://anaconda.org/anaconda/pywin32",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: POSIX :: Linux",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3.7",
    ],
    keywords = ['lithography', 'proximity effect correction','ebl', 'pec', 
        'fft', 'convolution', 'deconvolution','ebeam-lithography', 
        'monte-carlo-simulation', 'casino3'],
)