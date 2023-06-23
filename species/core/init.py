"""
Module for setting up species in the working folder.
"""

import configparser
import json
import os
import socket
import urllib.request

import h5py
import species


class SpeciesInit:
    """
    Class for initiating species by creating the database and
    configuration file in case they are not present in the working
    folder, and creating the data folder for storage of input data.
    """

    def __init__(self):
        """
        Returns
        -------
        NoneType
            None
        """

        species_msg = f"species v{species.__version__}"
        print(len(species_msg) * "=")
        print(species_msg)
        print(len(species_msg) * "=")

        working_folder = os.path.abspath(os.getcwd())
        print(f"Working folder: {working_folder}")

        config_file = os.path.join(working_folder, "species_config.ini")

        try:
            contents = urllib.request.urlopen(
                "https://pypi.org/pypi/species/json", timeout=1.0
            ).read()

            data = json.loads(contents)
            latest_version = data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and species.__version__ != latest_version:
            print(f" -> A new version ({latest_version}) is available!")
            print(" -> It is recommended to update to the latest version")
            print(" -> See https://github.com/tomasstolker/species for details")

        if not os.path.isfile(config_file):
            print("Creating species_config.ini...", end="", flush=True)

            with open(config_file, "w") as file_obj:
                file_obj.write("[species]\n\n")

                file_obj.write("; File with the HDF5 database\n")
                file_obj.write("database = species_database.hdf5\n\n")

                file_obj.write("; Folder where data will be downloaded\n")
                file_obj.write("data_folder = ./data/\n\n")

                file_obj.write("; Method for the grid interpolation\n")
                file_obj.write(
                    "; Options: linear, nearest, slinear, " "cubic, quintic, pchip\n"
                )
                file_obj.write("interp_method = linear\n")

            print(" [DONE]")

        config = configparser.ConfigParser()
        config.read(config_file)

        if "database" in config["species"]:
            database_file = os.path.abspath(config["species"]["database"])

        else:
            database_file = "species_database.hdf5"

            with open(config_file, "a") as file_obj:
                file_obj.write("\n; File with the HDF5 database\n")
                file_obj.write("database = species_database.hdf5\n")

        if "data_folder" in config["species"]:
            data_folder = os.path.abspath(config["species"]["data_folder"])

        else:
            data_folder = "./data/"

            with open(config_file, "a") as file_obj:
                file_obj.write("\n; Folder where data will be downloaded\n")
                file_obj.write("data_folder = ./data/\n")

        if "interp_method" in config["species"]:
            interp_method = config["species"]["interp_method"]

        else:
            interp_method = "linear"

            with open(config_file, "a") as file_obj:
                file_obj.write("\n; Method for the grid interpolation\n")
                file_obj.write(
                    "; Options: linear, nearest, slinear, " "cubic, quintic, pchip\n"
                )
                file_obj.write("interp_method = linear\n")

        if "vega_mag" in config["species"]:
            vega_mag = config["species"]["vega_mag"]

        else:
            vega_mag = 0.03

            with open(config_file, "a") as file_obj:
                file_obj.write("\n; Magnitude of Vega for all filters\n")
                file_obj.write("vega_mag = 0.03\n")

        print("Configuration settings:")
        print(f"   - Database: {database_file}")
        print(f"   - Data folder: {data_folder}")
        print(f"   - Interpolation method: {interp_method}")
        print(f"   - Magnitude of Vega: {vega_mag}")

        if not os.path.isfile(database_file):
            print("Creating species_database.hdf5...", end="", flush=True)
            h5_file = h5py.File(database_file, "w")
            h5_file.close()
            print(" [DONE]")

        if not os.path.exists(data_folder):
            print("Creating data folder...", end="", flush=True)
            os.makedirs(data_folder)
            print(" [DONE]")
