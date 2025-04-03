"""
Module for setting up species in the working folder.
"""

import json
import socket
import urllib.request

from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

from configparser import ConfigParser
from importlib.util import find_spec
from rich import print as rprint
from typeguard import typechecked

import h5py

from .._version import __version__


class SpeciesInit:
    """
    Class for initiating species by creating the database and
    configuration file in case they are not present in the working
    folder, and creating the data folder for storage of input data.
    """

    @typechecked
    def __init__(
        self, database_file: Optional[Union[str, PosixPath, WindowsPath]] = None
    ) -> None:
        """
        Parameters
        ----------
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

        species_init = f"[bold magenta]species v{__version__}[/bold magenta]"
        len_text = len(f"species v{__version__}")

        print(len_text * "=")
        rprint(species_init)
        print(len_text * "=")

        try:
            pypi_url = "https://pypi.org/pypi/species/json"

            with urllib.request.urlopen(pypi_url, timeout=1.0) as open_url:
                url_content = open_url.read()
                url_data = json.loads(url_content)
                latest_version = url_data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and __version__ != latest_version:
            print(f"\n -> A new version ({latest_version}) is available!")
            print(" -> It is recommended to update to the latest version")
            print(" -> See https://github.com/tomasstolker/species for details")

        working_folder = Path.cwd()
        print(f"\nWorking folder: {working_folder}")

        config_file = working_folder / "species_config.ini"

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

        if find_spec("mpi4py") is None:
            print("\nMultiprocessing: mpi4py not installed")

        else:
            from mpi4py import MPI

            # Rank of this process in a communicator
            mpi_rank = MPI.COMM_WORLD.Get_rank()

            # Number of processes in a communicator
            mpi_size = MPI.COMM_WORLD.Get_size()

            print("\nMultiprocessing: mpi4py installed")
            print(f"Process number {mpi_rank+1:d} out of {mpi_size:d}...")
