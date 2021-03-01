.PHONY: help pypi pypi-test docs coverage test clean

help:
	@echo "pypi - submit to PyPI server"
	@echo "pypi-test - submit to TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run unit tests"
	@echo "clean - remove artifacts"

pypi:
	python setup.py sdist bdist_wheel
	twine upload dist/*

pypi-test:
	python setup.py sdist bdist_wheel
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

docs:
	rm -f docs/species.analysis.rst
	rm -f docs/species.core.rst
	rm -f docs/species.data.rst
	rm -f docs/species.plot.rst
	rm -f docs/species.read.rst
	rm -f docs/species.util.rst
	sphinx-apidoc -o docs species
	cd docs/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

coverage:
	coverage run --source=species -m pytest
	coverage report -m

test:
	pytest --cov=species/

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	rm -f .coverage
	rm -f coverage.xml
	rm -rf .pytest_cache/
	rm -rf docs/_build/
	rm -rf docs/tutorials/data/
	rm -rf docs/tutorials/ultranest/
	rm -rf docs/tutorials/.ipynb_checkpoints
	rm -f docs/tutorials/species_config.ini
	rm -f docs/tutorials/species_database.hdf5
	rm -f docs/tutorials/*.png
	rm -f docs/tutorials/*.fits
	rm -f docs/tutorials/*.dat
	rm -rf build/
	rm -rf dist/
	rm -rf species.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
