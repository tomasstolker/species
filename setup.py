#!/usr/bin/env python

import pkg_resources
import setuptools

with open('requirements.txt') as req_txt:
    parse_req = pkg_resources.parse_requirements(req_txt)
    install_requires = [str(req) for req in parse_req]

setuptools.setup(
    name='species',
    version='0.7.1',
    description='Toolkit for atmospheric characterization of directly imaged exoplanets',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='Tomas Stolker',
    author_email='stolker@strw.leidenuniv.nl',
    url='https://github.com/tomasstolker/species',
    project_urls={'Documentation': 'https://species.readthedocs.io'},
    packages=setuptools.find_packages(include=['species', 'species.*']),
    package_data={'species': ['data/*.json']},
    install_requires=install_requires,
    tests_require=['pytest'],
    license='MIT',
    zip_safe=False,
    keywords='species',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
