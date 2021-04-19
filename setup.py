#!/usr/bin/env python
from setuptools import find_packages, setup

__version__ = "0.3.0"

if __name__ == '__main__':
    setup(
        name='nanodet',
        version=__version__,
        description='Deep Learning Object Detection Toolbox',
        url='https://github.com/RangiLyu/nanodet',
        author='RangiLyu',
        author_email='lyuchqi@gmail.com',
        keywords='deep learning',
        packages=find_packages(exclude=('config', 'tools', 'demo')),
        classifiers=[
            'Development Status :: Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        zip_safe=False)
