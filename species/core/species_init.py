"""
Module for setting up species in the working folder.
"""

import json
import socket
import urllib.request

from configparser import ConfigParser
from os import environ
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import h5py

from typeguard import typechecked

from .._version import __version__, __version_tuple__


class SpeciesInit:
    """
    Class for initiating species by creating the database and
    configuration file in case they are not present in the working
    folder, and creating the data folder for storage of input data.
    """

    @typechecked
    def __init__(
        self,
        config_file: Optional[Union[str, PosixPath, WindowsPath]] = None,
        database_file: Optional[Union[str, PosixPath, WindowsPath]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config_file : str, PosixPath, WindowsPath, None
            Path to the configuration file that is stored as
            `species_config.ini`. Setting the argument enables
            reading a configuration file at a different path
            than the current working directory. The default
            location (i.e. the working folder of your script or
            notebook) is used by setting the argument to ``None``.
        database_file : str, PosixPath, WindowsPath, None
            Path to the HDF5 database that is stored as
            `species_database.hdf5`. Setting the argument will
            overwrite the database path in the configuration file.
            The path from the configuration file is used if the
            argument of ``database_file`` is set to ``None``.

        Returns
        -------
        NoneType
            None
        """

        print("=======\nspecies\n=======")

        # Check if there is a new version available

        species_version = (
            f"{__version_tuple__[0]}."
            f"{__version_tuple__[1]}."
            f"{__version_tuple__[2]}"
        )

        try:
            pypi_url = "https://pypi.org/pypi/species/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                pypi_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            pypi_version = None

        if pypi_version is not None:
            pypi_split = pypi_version.split(".")
            current_split = species_version.split(".")

            new_major = (pypi_split[0] == current_split[0]) & (
                pypi_split[1] > current_split[1]
            )

            new_minor = (
                (pypi_split[0] == current_split[0])
                & (pypi_split[1] == current_split[1])
                & (pypi_split[2] > current_split[2])
            )

            if new_major | new_minor:
                print(f"\n-> species v{pypi_version} is available!")

        print(f"\nVersion: {__version__}")

        working_folder = Path.cwd()
        print(f"Working folder: {working_folder}")

        if config_file is None:
            config_file = working_folder / "species_config.ini"

        elif isinstance(config_file, str):
            config_file = Path(config_file)
            environ["SPECIES_CONFIG"] = str(config_file)

        config = ConfigParser(allow_no_value=True)

        if config_file.exists():
            print(f"\nConfiguration file: {config_file}")

        else:
            print("\nCreating species_config.ini...", end="", flush=True)

            config.add_section("species")
            # config.set('species', '; File with the HDF5 database')
            config.set("species", "database", "species_database.hdf5")
            # config.set('species', '; File with the HDF5 database')
            config.set("species", "data_folder", "./data/")
            # config.set('species', '; File with the HDF5 database')
            config.set("species", "vega_mag", "0.03")

            with open(config_file, "w", encoding="utf-8") as file_obj:
                config.write(file_obj)

            print(" [DONE]")

        config.read(config_file)
        config_update = False

        if database_file is None:
            if "database" in config["species"]:
                database_file = Path(config["species"]["database"])

            else:
                database_file = Path("./species_database.hdf5")
                config.set("species", "database", "species_database.hdf5")
                config_update = True

        else:
            if isinstance(database_file, str):
                config.set("species", "database", database_file)
                database_file = Path(database_file)
            else:
                config.set("species", "database", str(database_file))

            config_update = True

        if "data_folder" in config["species"]:
            data_folder = Path(config["species"]["data_folder"])

        else:
            data_folder = Path("./data/")
            config.set("species", "data_folder", "./data/")
            config_update = True

        if "vega_mag" in config["species"]:
            vega_mag = config["species"]["vega_mag"]

        else:
            vega_mag = 0.03
            config.set("species", "vega_mag", "0.03")
            config_update = True

        if config_update:
            # If condition is needed because the file should
            # not be opened when using MPI and the config
            # file is already present and no need to update
            with open(config_file, "w", encoding="utf-8") as file_obj:
                config.write(file_obj)

        if database_file.exists():
            print(f"Database file: {database_file}")

        else:
            print("Creating species_database.hdf5...", end="", flush=True)
            h5_file = h5py.File(database_file, "w")
            h5_file.close()
            print(" [DONE]")

        try:
            from mpi4py import MPI

            mpi_rank = MPI.COMM_WORLD.Get_rank()
            MPI.COMM_WORLD.Barrier()

        except ImportError:
            mpi_rank = 0

        # Add samples to the database

        if mpi_rank == 0:
            with h5py.File(database_file, "a") as hdf5_file:
                if "configuration" in hdf5_file:
                    del hdf5_file["configuration"]

                config_group = hdf5_file.create_group("configuration")
                config_group.attrs["config_file"] = str(config_file)
                config_group.attrs["database_file"] = str(database_file)
                config_group.attrs["data_folder"] = str(data_folder)

        if data_folder.exists():
            print(f"Data folder: {data_folder}")

        else:
            print("Creating data folder...", end="", flush=True)
            data_folder.mkdir()
            print(" [DONE]")

        print("\nConfiguration settings:")
        print(f"   - Database: {database_file}")
        print(f"   - Data folder: {data_folder}")
        print(f"   - Magnitude of Vega: {vega_mag}")

        try:
            from mpi4py import MPI

            # Rank of this process in a communicator
            mpi_rank = MPI.COMM_WORLD.Get_rank()

            # Number of processes in a communicator
            mpi_size = MPI.COMM_WORLD.Get_size()

            print("\nMultiprocessing: mpi4py installed")
            print(f"Process number {mpi_rank+1:d} out of {mpi_size:d}...")

        except ImportError:
            print("\nMultiprocessing: mpi4py not installed")
