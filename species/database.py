"""
Database module.
"""

import os
import sys
import tarfile
import warnings
import configparser
import urllib

import h5py
import requests
import numpy as np

from astropy.io import fits

import species.parallax
import species.photometry

warnings.simplefilter("ignore", UserWarning)


class Database(object):
    """
    Text.
    """

    def __init__(self):
        """
        :return: None
        """

        config_file = os.path.join(os.getcwd(), "species_config.ini")

        config = configparser.ConfigParser()
        config.read(config_file)

        self.database = config['species']['database']
        self.input_path = config['species']['input']

    def add_model(self,
                  model):
        """
        :param model: Model name.
        :type model: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if "models" not in h5_file:
            h5_file.create_group("models")

        if "models/"+model in h5_file:
            del h5_file["models/"+model]

        if model[0:13] == "drift-phoenix":

            data_file = os.path.join(self.input_path, "drift-phoenix.tgz")
            data_folder = os.path.join(self.input_path, "drift-phoenix/")

            url = "https://people.phys.ethz.ch/~stolkert/species/drift-phoenix.tgz"

            if not os.path.isfile(data_file):
                sys.stdout.write("Downloading DRIFT-PHOENIX atmospheric models, (151 MB)... ")
                sys.stdout.flush()

                urllib.urlretrieve(url, data_file)

                sys.stdout.write("[DONE]\n")
                sys.stdout.flush()

            sys.stdout.write("Unpacking DRIFT-PHOENIX atmospheric models... ")
            sys.stdout.flush()

            tar = tarfile.open(data_file)
            tar.extractall(path=self.input_path)
            tar.close()

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

            sys.stdout.write("Adding DRIFT-PHOENIX atmospheric models... ")
            sys.stdout.flush()

            file_list = []

            for root, _, files in os.walk(data_folder):
                for i, filename in enumerate(files):
                    if i == 0:
                        teff = []
                        logg = []
                        feh = []

                    if filename.startswith("lte_"):
                        teff.append(float(filename[4:8]))
                        logg.append(float(filename[9:12]))
                        feh.append(float(filename[12:16]))
                        file_list.append(filename)

                    else:
                        continue

                teff_sort = sorted(set(teff))
                logg_sort = sorted(set(logg))
                feh_sort = sorted(set(feh))

                modeldata = np.loadtxt(root+files[-1])
                wavelength = modeldata[:, 0]

                size = (len(teff_sort), len(logg_sort), len(feh_sort), len(wavelength))

                grid = np.zeros(size, dtype='float64')

                for i, filename in enumerate(file_list):
                    index_teff = teff_sort.index(teff[i])
                    index_logg = logg_sort.index(logg[i])
                    index_feh = feh_sort.index(feh[i])

                    modeldata = np.loadtxt(root+filename)

                    # [Angstrom] -> [micron]
                    wavelength = modeldata[:, 0]*1e-4

                    # [erg s-1 cm-2 Angstrom-1] -> [W m-2 micron-1]
                    flux = modeldata[:, 1]*1e-7*1e4*1e4

                    grid[index_teff, index_logg, index_feh, :] = flux

            h5_file.create_group("models/"+model)
            h5_file.create_dataset("models/"+model+"/teff", data=np.asarray(teff_sort))
            h5_file.create_dataset("models/"+model+"/logg", data=np.asarray(logg_sort))
            h5_file.create_dataset("models/"+model+"/feh", data=np.asarray(feh_sort))
            h5_file.create_dataset("models/"+model+"/wavelength", data=wavelength)
            h5_file.create_dataset("models/"+model+"/flux", data=grid)

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

            h5_file.close()

    def add_spectrum(self,
                     spectrum):
        """
        :param spectrum: Spectral library.
        :type spectrum: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if "spectra" not in h5_file:
            h5_file.create_group("spectra")

        if spectrum[0:5] == "vega":

            data_file = os.path.join(self.input_path, "alpha_lyr_stis_008.fits")
            url = "ftp://ftp.stsci.edu/cdbs/calspec/alpha_lyr_stis_008.fits"

            if not os.path.isfile(data_file):
                sys.stdout.write("Downloading Vega spectrum (270 kB)... ")
                sys.stdout.flush()

                urllib.urlretrieve(url, data_file)

                sys.stdout.write("[DONE]\n")
                sys.stdout.flush()

            if "spectra/calibration" not in h5_file:
                h5_file.create_group("spectra/calibration")

            if "spectra/calibration/vega" in h5_file:
                del h5_file["spectra/calibration/vega"]

            hdu = fits.open(data_file)
            data = hdu[1].data
            wavelength = data['WAVELENGTH'] # [Angstrom]
            flux = data['FLUX'] # [erg s-1 cm-2 A-1]
            error_stat = data['STATERROR'] # [erg s-1 cm-2 A-1]
            error_sys = data['SYSERROR'] # [erg s-1 cm-2 A-1]
            hdu.close()

            wavelength *= 1e-4 # [Angstrom] -> [micron]
            flux *= 1.e-3*1e4 # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]
            error_stat *= 1.e-3*1e4 # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]
            error_sys *= 1.e-3*1e4 # [erg s-1 cm-2 A-1] -> [W m-2 micron-1]

            sys.stdout.write("Adding Vega spectrum... ")
            sys.stdout.flush()

            h5_file.create_dataset("spectra/calibration/vega",
                                   data=np.vstack((wavelength, flux, error_stat)))

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

        elif spectrum[0:5] == "irtf":

            data_file = os.path.join(self.input_path, "alpha_lyr_stis_008.fits")

            data_file = [os.path.join(self.input_path, "M_fits_091201.tar"),
                         os.path.join(self.input_path, "L_fits_091201.tar"),
                         os.path.join(self.input_path, "T_fits_091201.tar")]

            data_folder = [os.path.join(self.input_path, "M_fits_091201"),
                           os.path.join(self.input_path, "L_fits_091201"),
                           os.path.join(self.input_path, "T_fits_091201")]

            data_type = ["M dwarfs (7.5 MB)",
                         "L dwarfs (850 kB)",
                         "T dwarfs (100 kB)"]

            url_root = "http://irtfweb.ifa.hawaii.edu/~spex/IRTF_Spectral_Library/Data/"

            url = [url_root+"M_fits_091201.tar",
                   url_root+"L_fits_091201.tar",
                   url_root+"T_fits_091201.tar"]

            for i, item in enumerate(data_file):
                if not os.path.isfile(item):
                    sys.stdout.write("Downloading IRTF Spectral Library - "+data_type[i]+"... ")
                    sys.stdout.flush()

                    urllib.urlretrieve(url[i], item)

                    sys.stdout.write("[DONE]\n")
                    sys.stdout.flush()

            sys.stdout.write("Unpacking IRTF Spectral Library... ")
            sys.stdout.flush()

            for i, item in enumerate(data_file):
                tar = tarfile.open(item)
                tar.extractall(path=data_folder[i])
                tar.close()

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

            if "spectra/"+spectrum in h5_file:
                del h5_file["spectra/"+spectrum]

            h5_file.create_group("spectra/"+spectrum)

            sys.stdout.write("Adding IRTF Spectral Library... ")
            sys.stdout.flush()

            for i, item in enumerate(data_folder):
                for root, _, files in os.walk(item):

                    for _, filename in enumerate(files):
                        if filename[-9:] != "_ext.fits":
                            fitsfile = os.path.join(root, filename)
                            spdata, header = fits.getdata(fitsfile, header=True)

                            name = header['OBJECT']
                            spec = header['SPTYPE']

                            if len(spec) == 2 or spec[3] != ".":
                                spec = spec[0:2]
                            else:
                                spec = spec[0:4]

                            distance = species.parallax.get_distance(name) # [pc]

                            dset = h5_file.create_dataset("spectra/"+spectrum+"/"+filename[:-5],
                                                          data=spdata)

                            dset.attrs['name'] = name
                            dset.attrs['sptype'] = spec
                            dset.attrs['distance'] = distance

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

            h5_file.close()

    def add_photometry(self,
                       photometry):
        """
        :param photometry: Photometry library.
        :type photometry: str

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        if "photometry" not in h5_file:
            h5_file.create_group("photometry")

        if photometry[0:7] == "vlm-plx":

            data_file = os.path.join(self.input_path, "vlm-plx-all.fits")
            url = "http://www.as.utexas.edu/~tdupuy/plx/" \
                  "Database_of_Ultracool_Parallaxes_files/vlm-plx-all.fits"

            if not os.path.isfile(data_file):
                sys.stdout.write("Downloading Database of Ultracool Parallaxes (307 kB)... ")
                sys.stdout.flush()

                urllib.urlretrieve(url, data_file)

                sys.stdout.write("[DONE]\n")
                sys.stdout.flush()

            sys.stdout.write("Adding Database of Ultracool Parallaxes... ")
            sys.stdout.flush()

            group = "photometry/"+photometry

            if group in h5_file:
                del h5_file[group]

            h5_file.create_group(group)

            hdulist = fits.open(data_file)
            photdata = hdulist[1].data

            parallax = photdata['PLX'] # [mas]
            distance = 1./(parallax*1e-3) # [pc]

            name = photdata['NAME']
            name = np.core.defchararray.strip(name)

            sptype = photdata['ISPTSTR']
            sptype = np.core.defchararray.strip(sptype)

            flag = photdata['FLAG']
            flag = np.core.defchararray.strip(flag)

            h5_file.create_dataset(group+"/name", data=name)
            h5_file.create_dataset(group+"/sptype", data=sptype)
            h5_file.create_dataset(group+"/flag", data=flag)
            h5_file.create_dataset(group+"/distance", data=distance)

            h5_file.create_dataset(group+"/MKO/NSFCam.Y", data=photdata['YMAG'])
            h5_file.create_dataset(group+"/MKO/NSFCam.J", data=photdata['JMAG'])
            h5_file.create_dataset(group+"/MKO/NSFCam.H", data=photdata['HMAG'])
            h5_file.create_dataset(group+"/MKO/NSFCam.K", data=photdata['KMAG'])
            h5_file.create_dataset(group+"/MKO/NSFCam.Lp", data=photdata['LMAG'])
            h5_file.create_dataset(group+"/MKO/NSFCam.Mp", data=photdata['MMAG'])

            h5_file.create_dataset(group+"/2MASS/2MASS.J", data=photdata['J2MAG'])
            h5_file.create_dataset(group+"/2MASS/2MASS.H", data=photdata['H2MAG'])
            h5_file.create_dataset(group+"/2MASS/2MASS.Ks", data=photdata['K2MAG'])

            sys.stdout.write("[DONE]\n")
            sys.stdout.flush()

        h5_file.close()

    def add_filter(self,
                   filter_id):
        """
        :param filter_id: Filter ID from the SVO Filter Profile Service (e.g., "Paranal/NACO.Lp").
        :type filter_id: str

        :return: None
        """

        filter_split = filter_id.split("/")

        h5_file = h5py.File(self.database, 'a')

        if "filters" not in h5_file:
            h5_file.create_group("filters")

        if "filters/"+filter_split[0] not in h5_file:
            h5_file.create_group("filters/"+filter_split[0])

        if "filters/"+filter_id in h5_file:
            del h5_file["filters/"+filter_id]

        sys.stdout.write("Adding filter "+filter_id+"... ")
        sys.stdout.flush()

        url = "http://svo2.cab.inta-csic.es/svo/theory/fps/getdata.php?format=ascii&id="+filter_id

        session = requests.Session()
        response = session.get(url)
        data = response.content

        wavelength = []
        transmission = []
        for line in data.splitlines():
            if not line.startswith("#"):
                split = line.split(" ")

                wavelength.append(float(split[0])*1e-4) # [micron]
                transmission.append(float(split[1]))

        wavelength = np.array(wavelength)
        transmission = np.array(transmission)

        h5_file.create_dataset("filters/"+filter_id, data=np.vstack((wavelength, transmission)))

        sys.stdout.write("[DONE]\n")
        sys.stdout.flush()

        h5_file.close()

    def add_object(self,
                   object_name,
                   distance,
                   app_mag):
        """
        :param object_name: Object name.
        :type object_name: str
        :param distance: Distance (pc).
        :type distance: float
        :param app_mag: Apparent magnitudes.
        :type app_mag: dict

        :return: None
        """

        h5_file = h5py.File(self.database, 'a')

        sys.stdout.write("Adding "+object_name+"... ")
        sys.stdout.flush()

        if "objects" not in h5_file:
            h5_file.create_group("objects")

        if "objects/"+object_name not in h5_file:
            h5_file.create_group("objects/"+object_name)

        if "objects/"+object_name+"/distance" in h5_file:
            del h5_file["objects/"+object_name+"/distance"]

        h5_file.create_dataset("objects/"+object_name+"/distance", data=distance) # [pc]

        for _, item in enumerate(app_mag):
            if "objects/"+object_name+"/"+item in h5_file:
                del h5_file["objects/"+object_name+"/"+item]

            h5_file.create_dataset("objects/"+object_name+"/"+item, data=app_mag[item]) # [mag]

        sys.stdout.write("[DONE]\n")
        sys.stdout.flush()

        h5_file.close()
