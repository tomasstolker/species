.. _installation:

Installation
============

*species* is compatible with Python 3.6/3.7 and is available in the |pypi| and on |github|.

Installation from PyPI
----------------------

*species* can be installed with the |pip|:

.. code-block:: console

    $ pip install species

And to update to the most recent version:

.. code-block:: console

   $ pip install --upgrade species

.. important::
   Currently it is recommended to install *species* from Github (see below) in order to benefit from the most recent implementations. The available package on PyPI is also stable but it usually trails behind the version on Github.

Installation from Github
------------------------

Installation from Github is done by cloning the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/species.git

And running the setup script to install the package and its dependencies:

.. code-block:: console

    $ python setup.py install

.. important::
   If an error occurs when running ``setup.py`` then possibly ``pip`` needs to be updated the latest version:

   .. code-block:: console

       $ pip install --upgrade pip

Alternatively to running the ``setup.py`` file, the folder where ``species`` is located can also be added to the ``PYTHONPATH`` environment variable such that the package is found by Python. The command may depend on the OS that is used, but is often something like:

.. code-block:: console

    $ export PYTHONPATH=$PYTHONPATH:/path/to/species

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin master

Do you want to make changes to the code? Please fork the `species` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are very welcome (see :ref:`contributing` section).

Testing `species`
-----------------

The installation can be tested by starting Python in interactive mode and printing the `species` version:

.. code-block:: python

    >>> import species
    >>> species.__version__

.. |pypi| raw:: html

   <a href="https://pypi.org/project/species/" target="_blank">PyPI repository</a>

.. |github| raw:: html

   <a href="https://github.com/tomasstolker/species" target="_blank">Github</a>

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>
