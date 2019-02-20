"""
Text
"""

# TODO HR 8799, kappa And, 2M1207, GJ 504, HD 95086, GQ Lup, RXJ1609, PSO J318, GU Psc

def get_data():
    """
    :return:
    :rtype: dict
    """

    data = {'beta Pic b':{'distance':19.75,
                          'app_mag':{'LCO/VisAO.Ys':(15.53, 0.34), # Males et al. (2014)
                                     'Paranal/NACO.J':(14.0, 0.3), # Bonnefoy et al. (2013)
                                     'Gemini/NICI.ED286':(13.18, 0.15), # Males et al. (2014)
                                     'Paranal/NACO.H':(13.5, 0.2), # Bonnefoy et al. (2013)
                                     'Paranal/NACO.Ks':(12.64, 0.11), # Bonnefoy et al. (2011)
                                     'Paranal/NACO.Lp':(11.30, 0.06), # Stolker et al. (2019)
                                     'Paranal/NACO.NB405':(11.20, 0.23), # Quanz et al. (2010)
                                     'Paranal/NACO.Mp':(11.10, 0.12)}}, # Stolker et al. (2019)

            'AB Pic b':{'distance':50.12,
                        'app_mag':{'Paranal/NACO.J':(16.18, 0.10), # Chauvin et al. 2005
                                   'Paranal/NACO.H':(14.69, 0.10), # Chauvin et al. 2005
                                   'Paranal/NACO.Ks':(14.14, 0.08)}}, # Chauvin et al. 2005

            'HD 206893 B':{'distance':40.81,
                           'app_mag':{'Paranal/SPHERE.IRDIS_B_H':(16.79, 0.06), # Milli et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_1':(15.2, 0.10), # Chauvin et al. 2017
                                      'Paranal/SPHERE.IRDIS_D_K12_2':(14.88, 0.09), # Chauvin et al. 2017
                                      'Paranal/NACO.Lp':(13.43, 0.16)}}, # Milli et al. 2016, TODO update error to +0.17, -0.15?
                                      # 'Paranal/NACO.Mp':(), # Stolker et al. in prep.

            'PZ Tel B':{'distance':47.13,
                        'app_mag':{'Paranal/SPHERE.ZIMPOL_R_PRIM':(17.84, 0.3), # Maire et al. 2015, TODO update error to +0.22, −0.31?
                                   'Paranal/SPHERE.ZIMPOL_I_PRIM':(15.16, 0.12), # Maire et al. 2015
                                   'Paranal/SPHERE.IRDIS_D_H23_2':(11.78, 0.19), # Maire et al. 2015
                                   'Paranal/SPHERE.IRDIS_D_H23_3':(11.65, 0.19), # Maire et al. 2015
                                   'Paranal/SPHERE.IRDIS_D_K12_1':(11.56, 0.09), # Maire et al. 2015
                                   'Paranal/SPHERE.IRDIS_D_K12_2':(11.29, 0.10), # Maire et al. 2015
                                   'Paranal/NACO.J':(12.47, 0.20), # Biller et al. 2010
                                   'Paranal/NACO.H':(11.93, 0.14), # Biller et al. 2010
                                   'Paranal/NACO.Ks':(11.53, 0.07), # Biller et al. 2010
                                   'Paranal/NACO.Lp':(11.05, 0.18), # Beust et al. 2015
                                   # 'Paranal/NACO.NB405':(), # Stolker et al. in prep.
                                   # 'Paranal/NACO.Mp':(), # Stolker et al. in prep.
                                   'Gemini/NICI.ED286':(11.68, 0.14), # Biller et al. 2010
                                   'Gemini/NIRI.H2S1v2-1-G0220':(11.39, 0.14)}}, # Biller et al. 2010

             'HIP 65426 b':{'distance':109.21,
                            'app_mag':{'Paranal/SPHERE.IRDIS_D_H23_2':(17.94, 0.05), # Chauvin et al. 2017
                                       'Paranal/SPHERE.IRDIS_D_H23_3':(17.58, 0.06), # Chauvin et al. 2017
                                       'Paranal/SPHERE.IRDIS_D_K12_1':(17.01, 0.09), # Chauvin et al. 2017
                                       'Paranal/SPHERE.IRDIS_D_K12_2':(16.79, 0.09), # Chauvin et al. 2017
                                       'Paranal/NACO.Lp':(15.26, 0.15), # Cheetham et al. 2018
                                       # 'Paranal/NACO.NB405':(), # Stolker et al. in prep.
                                       'Paranal/NACO.Mp':(15.1, 0.5)}}, # Cheetham et al. 2018

             '51 Eri b':{'distance':29.78,
                         'app_mag':{'MKO/NSFCam.J':(19.04, 0.40), # Rajan et al. 2017
                                    'MKO/NSFCam.H':(18.99, 0.21), # Rajan et al. 2017
                                    'MKO/NSFCam.K':(18.67, 0.19), # Rajan et al. 2017
                                    'Paranal/SPHERE.IRDIS_D_H23_2':(18.41, 0.26), # Samland et al. 2017
                                    'Paranal/SPHERE.IRDIS_D_K12_1':(17.55, 0.14), # Samland et al. 2017
                                    'Keck/NIRC2.Lp':(16.20, 0.11), # Rajan et al. 2017
                                    'Keck/NIRC2.Mp':(16.1, 0.5)}}, # Rajan et al. 2017

             'GSC 06214 B':{'distance':108.84,
                            'app_mag':{'MKO/NSFCam.J':(16.24, 0.04), # Ireland et al. 2011
                                       'MKO/NSFCam.H':(15.55, 0.04), # Ireland et al. 2011
                                       'MKO/NSFCam.Kp':(14.95, 0.05), # Ireland et al. 2011
                                       'MKO/NSFCam.Lp':(13.75, 0.07)}}} # Ireland et al. 2011
                                     # 'Paranal/NACO.Mp':(), # Stolker et al. in prep.

    return data
