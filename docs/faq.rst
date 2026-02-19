.. _faq:

Frequent Asked Questions
========================

What dependency versions should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``species`` package is kept reasonably well up to date with recent dependency versions. If you encounter an error then try to update the dependencies.

For example, in your local folder where you may have cloned the repository:

.. code-block:: bash

   pip install --upgrade -e .

Which tools support multiprocessing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The parameter inference tools in ``FitModel``, ``FitEvolution``, ``EmissionLine``, and ``AtmosphericRetrieval`` that make use of ``MultiNest``, ``UltraNest``, and ``Dynesty``.

How do I run my code on multiple CPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, make sure to install ``mpi4py``:

.. code-block:: bash

   pip install mpi4py

Then, to execute you ``species`` script with MPI, for example using 8 CPUs:

.. code-block:: bash

   mpirun -n 8 python run_species.py

.. important::
   Writing to the HDF5 database is not possible with multiprocessing, whereas reading data is. It is important to store any needed data (e.g. companion data, atmospheric models) in the HDF5 database using a single CPU, before running a fit. Next, it is recommended to execute a minimal code with multiprocessing, e.g. only ``SpeciesInit``, ``FitModel``, and ``run_multinest``. The results from the fit will get stored in the database.
