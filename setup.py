from setuptools import find_packages
from setuptools import setup
from codecs import open
import os

current_path = os.getcwd()

with open(os.path.join(current_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='simple_othello', 
    packages=find_packages(exclude=('*.pyc',)),
    version='1.2.5',
    license='MIT', 
    install_requires=[],
    author='Michiyasu Uchiyama',
    description='simple_othello', 
    long_description=long_description,
    long_description_content_type='text/markdown', 
    keywords='simple_othello othello reversi', 

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
