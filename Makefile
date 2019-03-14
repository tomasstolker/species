.PHONY: help pypi docs coverage test tox clean

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
	rm -f docs/species.*rst
	sphinx-apidoc -o docs/ species/
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
	rm -rf docs/_build
	rm -rf build/
	rm -rf dist/
	rm -rf species.egg-info/
	rm -rf htmlcov/
	rm -rf .tox/
