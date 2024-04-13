
from pathlib import Path

import setuptools

def parse_requirements(requirements: str):
    with open(requirements) as f:
        return [
            l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')
        ]

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="geneticalgorithm2", 
    version=Path('version.txt').read_text(encoding='utf-8').strip(),
    author="Demetry Pascal",
    author_email="qtckpuhdsa@gmail.com",
    maintainer='Demetry Pascal',
    description="Supported highly optimized and flexible genetic algorithm package for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PasaOpasen/geneticalgorithm2",
    license='MIT',
    keywords=[
        'solve', 'solver', 'equation', 
        'optimization', 'problem', 'genetic', 
        'algorithm', 'GA', 'easy', 'fast', 'genetic-algorithm', 
        'combinatorial', 'mixed', 'evolutionary',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=parse_requirements('./requirements.txt'),
    extras_require={
        'full': parse_requirements('./requirements-extra.txt')
    }
)





