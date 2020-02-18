.PHONY: help pypi ppypi-test docs coverage test clean

help:
	@echo "pypi - submit package to the PyPI server"
	@echo "pypi-test - submit package to the TestPyPI server"
	@echo "docs - generate Sphinx documentation"
	@echo "coverage - check code coverage"
	@echo "test - run test cases"
	@echo "clean - remove all artifacts"

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
	coverage run -m pytest
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
	rm -rf docs/tutorials/.ipynb_checkpoints
	rm -f docs/tutorials/species_config.ini
	rm -f docs/tutorials/species_database.hdf5
	rm -f docs/tutorials/*.png
	rm -rf build/
	rm -rf dist/
	rm -rf species.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
