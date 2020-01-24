.. _installation:

Installation
============

*species* is compatible with Python 3.6/3.7 and is available in the |pypi| and on |github|.

Virtual Environment
-------------------

It is recommended to use a Python virtual environment to install and run `species` such that the correct versions of the dependencies can be installed without affecting other installed Python packages. First install `virtualenv`, for example with the |pip|:

.. code-block:: console

    $ pip install virtualenv

Then create a virtual environment for Python 3:

.. code-block:: console

    $ virtualenv -p python3 folder_name

And activate the environment with:

.. code-block:: console

    $ source folder_name/bin/activate

A virtual environment can be deactivated with:

.. code-block:: console

    $ deactivate

.. important::
   Make sure to adjust the path where the virtual environment is installed and activated.

Installation from PyPI
----------------------

*species* can be installed from the |pypi| with the |pip|::

    $ pip install species

If you do not use a virtual environment then you may have to add the ``--user`` argument:

.. code-block:: console

    $ pip install --user species

To update the installation to the most recent version:

.. code-block:: console

   $ pip install --upgrade species

Installation from Github
------------------------

Installation from Github is also possible by cloning the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/species.git

In that case, the dependencies can be installed from the `species` repository folder:

.. code-block:: console

    $ pip install -r requirements.txt

Once a local copy of the repository exists, new commits can be pulled from Github with:

.. code-block:: console

    $ git pull origin master

And to update the dependencies for compatibility with `species`:

.. code-block:: console

    $ pip install --upgrade -r requirements.txt 

By adding the path of the repository to the ``PYTHONPATH`` environment variable enables `species` to be imported from any location:

.. code-block:: console

    $ echo "export PYTHONPATH='$PYTHONPATH:/path/to/species'" >> folder_name/bin/activate

.. important::
   Make sure to adjust local path in which `species` will be cloned from the Github repository.

Do you want to makes changes to the code? Then please fork the `species` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are very welcome (see :ref:`contributing` section).

Testing `species`
-----------------

The installation can be tested by starting Python in interactive mode and printing the `species` version:

.. code-block:: python

    >>> import species
    >>> species.__version__

.. tip::
   If the `species` package is not find by Python then possibly the path was not set correctly. The list of folders that are searched by Python for modules can be printed in interactive mode as:

      .. code-block:: python

         >>> import sys
         >>> sys.path

.. |pypi| raw:: html

   <a href="https://pypi.org/project/species/" target="_blank">PyPI repository</a>

.. |github| raw:: html

   <a href="https://github.com/tomasstolker/species" target="_blank">Github</a>

.. |pip| raw:: html

   <a href="https://packaging.python.org/tutorials/installing-packages/" target="_blank">pip package manager</a>
