.. _installation:

Installation
============

``species`` is available on `PyPI <https://pypi.org/project/species/>`_ and `GitHub <https://github.com/tomasstolker/species>`_.

It is recommended to install ``species`` within a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_.

Installation from PyPI
----------------------

Using the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_, it is important to first install `Cython <https://cython.org>`_:

.. code-block:: console

    $ pip install cython
    $ pip install species

Or, including all dependencies will enable all functionalities of ``species``:

.. code-block:: console

    $ pip install 'species[full]'

To update ``species`` to the most recent version:

.. code-block:: console

   $ pip install --upgrade species

For running free atmospheric retrievals, it is required to separately install ``petitRADTRANS``, but this step can be skipped otherwise:

.. code-block:: console

   $ pip install petitRADTRANS

.. important::
   The ``PyMultiNest`` package requires the manual installation of ``MultiNest``. Please follow the `instructions <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for the building the library and make sure that the ``LD_LIBRARY_PATH`` (on Linux) or ``DYLD_LIBRARY_PATH`` (on macOS) environment variable is set. It is possible to use many of the functionalities in ``species`` without installing ``MultiNest``.

Installation from GitHub
------------------------

Using pip
^^^^^^^^^

The repository on `GitHub <https://github.com/tomasstolker/species>`_ contains the latest implementations and can also be installed with `pip <https://packaging.python.org/tutorials/installing-packages/>`_. The following will include the minimum of required dependencies:

.. code-block:: console

    $ pip install git+https://github.com/tomasstolker/species.git

Or, including all dependencies will enable all functionalities of ``species``:

.. code-block:: console

    $ pip install 'git+https://github.com/tomasstolker/species.git#egg=species[full]'


Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

It is best to clone the repository in case you want to make changes to the code:

.. code-block:: console

    $ git clone https://github.com/tomasstolker/species.git

Next, the package is installed by running ``pip`` in the local repository folder:

.. code-block:: console

    $ pip install -e .

Or, to install with all dependencies:

.. code-block:: console

    $ pip install ".[full]"

New commits can be pulled from GitHub once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Contributions and pull requests are welcome (see :ref:`contributing` section). In that case, please fork the `species` repository on the GitHub page and clone your own fork instead of the main repository.

Testing `species`
-----------------

To test the installation, we can initialize a ``species`` workflow:

.. code-block:: python

    >>> from species import SpeciesInit
    >>> SpeciesInit()
