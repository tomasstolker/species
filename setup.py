#!/usr/bin/env python

from setuptools import setup

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

reqs = parse_requirements('requirements.txt', session='hack')
reqs = [str(ir.req) for ir in reqs]

setup(
    name='species',
    version='0.0.1',
    description='Spectral and Photometric Examination Code for Investigating Exoplanet and Substellar atmospheres',
    long_description=open('README.rst').read(),
    author='Tomas Stolker',
    author_email='tomas.stolker@phys.ethz.ch',
    url='https://github.com/tomasstolker/SPECIES',
    packages=['species'],
    package_dir={'species':'species'},
    include_package_data=True,
    install_requires=reqs,
    license='GPLv3',
    zip_safe=False,
    keywords='SPECIES',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
)
