.. _installation:

Installation
============

``species`` is compatible with `Python <https://www.python.org>`_ versions 3.10/3.11/3.12 and is available in the `PyPI repository <https://pypi.org/project/species/>`_ and on `Github <https://github.com/tomasstolker/species>`_.

It is recommended to install ``species`` within a `virtual environment <https://docs.python.org/3/tutorial/venv.html>`_ such that the required dependency versions will not cause any conflicts with other installed packages. 

Installation from PyPI
----------------------

The ``species`` toolkit can be installed with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_, but first `Cython <https://cython.org>`_ should be separately installed:

.. code-block:: console

    $ pip install cython
    $ pip install species

Or, to update ``species`` to the most recent version:

.. code-block:: console

   $ pip install --upgrade species

It is also required to separately install ``petitRADTRANS``, although this step can be skipped:

.. code-block:: console

   $ pip install petitRADTRANS

The installation of ``petitRADTRANS`` can be somewhat challenging on some machines. When skipping the installation of `petitRADTRANS <https://petitradtrans.readthedocs.io>`_ it is still possible to use most of the functionalities of ``species``.

.. important::
   The ``PyMultiNest`` package requires the manual installation of ``MultiNest``. Please follow the `instructions <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for the building the library and make sure that the ``LD_LIBRARY_PATH`` (on Linux) or ``DYLD_LIBRARY_PATH`` (on macOS) environment variable is set. It is also possible to use ``species`` without installing ``MultiNest`` (but a warning will appear), apart from the functionalities that rely on ``PyMultiNest``.

Installation from Github
------------------------

Using pip
^^^^^^^^^

The repository on `Github <https://github.com/tomasstolker/species>`_ contains the latest implementations and can also be installed with `pip <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install git+https://github.com/tomasstolker/species.git

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

In case you want to look into the code, it is best to clone the repository:

.. code-block:: console

    $ git clone https://github.com/tomasstolker/species.git

Next, the package is installed by running ``pip`` in the local repository folder:

.. code-block:: console

    $ pip install -e .

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the `species` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are welcome (see :ref:`contributing` section).

Testing `species`
-----------------

The installation can now be tested, for example by starting Python in interactive mode and initializing a workflow in the current working folder:

.. code-block:: python

    >>> from species import SpeciesInit
    >>> SpeciesInit()
