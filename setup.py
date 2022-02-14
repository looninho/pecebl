from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
PATCH = 1
VERSION = "{}.{}.{}".format(MAJOR, MINOR, PATCH)

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open("pecebl/version.py", "w") as f:
    f.write("__version__ = '{}'\n".format(VERSION))

setup(
    name = "pecebl",
    version = VERSION,
    description = "eBeam Lithography simulation and Proximity Effect Correction",
    long_description = long_description,
    long_description_content_type='text/markdown',
    license = "GPLv3",
    author = "Luan Nguyen",
    author_email = "looninho@gmail.com",
    url = "http://github.com/looninho/pecebl",
    packages = find_packages(),
    include_package_data=True,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'pecebl = scripts.cli_demo:main',
        ],
    },
    install_requires=[
        'ipython==7.16.3', 'jupyter', 'numpy','scipy', 
        'sympy', 'pandas', 'pyqtgraph', 'pyopengl', 'matplotlib', 
        'imageio', 'pyculib', 'pycuda', 'scikit-cuda',
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
        'monte-carlo-simulation', 'casino', 'casino3'],
)