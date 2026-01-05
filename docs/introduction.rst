.. _introduction:

Introduction
============

*species* is a toolkit for the spectral and photometric analysis of directly imaged exoplanets and brown dwarfs. It provides a unified framework for working with a variety of observational data, models, and analysis tools.

The toolkit builds upon a broad collection of publicly available :ref:`data`, such as atmospheric model spectra, evolutionary tracks, spectral and photometric libraries, and filter profiles. Required data are downloaded and stored in a local HDF5 :ref:`database` that acts as the central data storage for a workflow. Databases are reusable by setting their path in the :ref:`configuration` file.

There are tools available for parameter inference, synthetic photometry, interpolation of model grids, color-magnitude and color-color diagrams, spectral classification, flux calibration, emission line analysis, and more. The notebook :ref:`tutorials` demonstrate many of these functionalities.
