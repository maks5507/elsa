#
# Created by mae9785 (eremeev@nyu.edu)
#

from setuptools import setup, find_packages
import setuptools.command.build_py as build_py


setup_kwargs = dict(
    name='elsa',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'nltk',
        'pymorphy2',
        'ufal.udpipe',
        'fasttext',
        'spacy'
    ],
    setup_requires=[
    ],

    cmdclass={'build_py': build_py.build_py},
)

setup(**setup_kwargs)
