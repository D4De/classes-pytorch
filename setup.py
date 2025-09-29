import os
from setuptools import find_packages
from setuptools import setup


pack = [p for p in find_packages() if p.startswith("classes_core")]


setup(
    include_package_data=True,
    name="classes",  # Non ho capito a cosa serve, perchè poi usa il nome della cartella come nome effettivo
    version="2.0.3",
    description="A Cross-Layer framework for evaluating reliability of Deep Neural Networks",
    url="",  # Da mettere il link della repo
    author="Design 4 Dependability Group (Polimi)",
    author_email="dario.passarello@polimi.it",
    packages=pack + ["classes_core/models"],
    package_data={  # Per includere altri file diversi da .py nel package
        "classes_core": ["logging.conf"],  # Per il logger
        "classes_core/models": ["*.json"],
    },
    install_requires=[],  # Da provare su un CONDA vuoto quali sono i requirements da fissare qui
    long_description="A fault error simulator to emulate GPU faults on CNN",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
