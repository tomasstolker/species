"""
Module with a function for data of directly imaged companions.
"""


def get_data():
    """
    Returns
    -------
    dict
        Dictionary with the distances and apparent magnitudes of directly imaged companions.
    """

    data = {'beta Pic b': {'distance': 19.75,
                           'app_mag': {'Paranal/NACO.J': (14.0, 0.3),  # Bonnefoy et al. (2013)
                                       'Gemini/NICI.ED286': (13.18, 0.15),  # Males et al. (2014)
                                       'Paranal/NACO.H': (13.5, 0.2),  # Bonnefoy et al. (2013)
                                       'Paranal/NACO.Ks': (12.64, 0.11),  # Bonnefoy et al. (2011)
                                       'Paranal/NACO.NB374': (11.24, 0.15),  # Stolker et al. in prep.
                                       'Paranal/NACO.Lp': (11.30, 0.06),  # Stolker et al. (2019)
                                       'Paranal/NACO.NB405': (11.03, 0.06),  # Stolker et al. in prep.
                                       'Paranal/NACO.Mp': (11.10, 0.12)}},  # Stolker et al. (2019)

            'HIP 65426 b': {'distance': 109.21,
                            'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (17.94, 0.05),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_H23_3': (17.58, 0.06),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_1': (17.01, 0.09),  # Chauvin et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_2': (16.79, 0.09),  # Chauvin et al. 2017
                                        'Paranal/NACO.NB405': (15.41, 0.30),  # Stolker et al. in prep.
                                        'Paranal/NACO.Lp': (15.34, 0.16),  # Stolker et al. in prep.
                                        'Paranal/NACO.Mp': (14.82, 0.35)}},  # Stolker et al. in prep.

            '51 Eri b': {'distance': 29.78,
                         'app_mag': {'MKO/NSFCam.J': (19.04, 0.40),  # Rajan et al. 2017
                                     'MKO/NSFCam.H': (18.99, 0.21),  # Rajan et al. 2017
                                     'MKO/NSFCam.K': (18.67, 0.19),  # Rajan et al. 2017
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (18.41, 0.26),  # Samland et al. 2017
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (17.55, 0.14),  # Samland et al. 2017
                                     'Keck/NIRC2.Lp': (16.20, 0.11),  # Rajan et al. 2017
                                     'Keck/NIRC2.Ms': (16.1, 0.5)}},  # Rajan et al. 2017

            'HR 8799 b': {'distance': 41.29,
                          'app_mag': {'Subaru/CIAO.z': (21.22, 0.29),  # Currie et al. 2011
                                      'Paranal/SPHERE.IRDIS_B_J': (19.78, 0.09),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (18.05, 0.09),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (18.08, 0.14),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (17.78, 0.10),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (17.03, 0.08),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (17.15, 0.06),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (16.97, 0.09),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Lp': (15.58, 0.10),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (14.92, 0.18),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (16.05, 0.30)}},  # Galicher et al. 2011

            'HR 8799 c': {'distance': 41.29,
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.60, 0.13),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (17.06, 0.13),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (17.09, 0.12),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.78, 0.10),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (16.11, 0.08),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.19, 0.05),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.86, 0.07),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Lp': (15.72, 0.08),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (14.07, 0.08),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (15.03, 0.14)}},  # Galicher et al. 2011

            'HR 8799 d': {'distance': 41.29,
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.59, 0.37),  # Zurlo et al. 2016
                                      'Keck/NIRC2.H': (16.71, 0.24),  # Currie et al. 2012
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (17.02, 0.17),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.68, 0.21),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (16.09, 0.12),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.20, 0.07),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.84, 0.10),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Lp': (14.56, 0.09),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (13.97, 0.14),  # Currie et al. 2014
                                      'Keck/NIRC2.Ms': (14.65, 0.35)}},  # Galicher et al. 2011

            'HR 8799 e': {'distance': 41.29,
                          'app_mag': {'Paranal/SPHERE.IRDIS_B_J': (18.40, 0.21),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_2': (16.91, 0.20),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_H23_3': (16.85, 0.16),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Ks': (15.91, 0.22),  # Marois et al. 2010
                                      'Paranal/SPHERE.IRDIS_D_K12_1': (16.12, 0.10),  # Zurlo et al. 2016
                                      'Paranal/SPHERE.IRDIS_D_K12_2': (15.82, 0.11),  # Zurlo et al. 2016
                                      'Keck/NIRC2.Lp': (14.55, 0.12),  # Currie et al. 2014
                                      'Paranal/NACO.NB405': (13.82, 0.20)}},  # Currie et al. 2014

            'HD 95086 b': {'distance': 86.44,
                           'app_mag': {'Gemini/GPI.H': (20.51, 0.25),  # De Rosa et al. 2016
                                       'Gemini/GPI.K1': (18.99, 0.20),  # De Rosa et al. 2016
                                       'Paranal/NACO.Lp': (16.27, 0.19)}},  # De Rosa et al. 2016

            'PDS 70 b': {'distance': 113.43,
                         'app_mag': {'Paranal/SPHERE.IRDIS_D_H23_2': (17.94, 0.24),  # Keppler et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (17.95, 0.17),  # Keppler et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (16.65, 0.06),  # Müller et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_K12_2': (16.44, 0.05),  # Müller et al. 2018
                                     'Paranal/NACO.Lp': (14.75, 0.62)}},  # Keppler et al. 2018

            '2M1207 b': {'distance': 64.42,
                         'app_mag': {'HST/NICMOS1.F090M': (22.58, 0.35),  # Song et al. 2006
                                     'HST/NICMOS1.F145M': (19.05, 0.03),  # Song et al. 2006
                                     'HST/NICMOS1.F160W': (18.27, 0.02),  # Song et al. 2006
                                     'Paranal/NACO.J': (20.0, 0.20),  # Mohanty et al. 200z
                                     'Paranal/NACO.H': (18.09, 0.21),  # Chauvin et al. 2004
                                     'Paranal/NACO.Ks': (16.93, 0.11),  # Chauvin et al. 2004
                                     # 'Magellan/MagAO.3.3': (15.46, 0.10),  # Skemer et al. 2014
                                     'Paranal/NACO.Lp': (15.28, 0.14)}},  # Chauvin et al. 2004

            'AB Pic B': {'distance': 50.12,
                         'app_mag': {'Paranal/NACO.J': (16.18, 0.10),  # Chauvin et al. 2005
                                     'Paranal/NACO.H': (14.69, 0.10),  # Chauvin et al. 2005
                                     'Paranal/NACO.Ks': (14.14, 0.08)}},  # Chauvin et al. 2005

            'HD 206893 B': {'distance': 40.81,
                            'app_mag': {'Paranal/SPHERE.IRDIS_B_H': (16.79, 0.06),  # Milli et al. 2016
                                        'Paranal/SPHERE.IRDIS_D_K12_1': (15.2, 0.10),  # Delorme et al. 2017
                                        'Paranal/SPHERE.IRDIS_D_K12_2': (14.88, 0.09),  # Delorme et al. 2017
                                        'Paranal/NACO.NB405': (12.79, 0.59),  # Stolker et al. in prep.
                                        'Paranal/NACO.Lp': (14.01, 0.30),  # Stolker et al. in prep.
                                        'Paranal/NACO.Mp': (13.60, 0.54)}},  # Stolker et al. in prep.

            'GQ Lup B': {'distance': 151.82,
                         'app_mag': {'HST/WFPC2.f606w': (19.19, 0.07),  # Marois et al. 2006
                                     'HST/WFPC2.f814w': (17.67, 0.05),  # Marois et al. 2006
                                     'HST/NICMOS2.F171M': (13.84, 0.13),  # Marois et al. 2006
                                     'HST/NICMOS2.F190N': (14.08, 0.20),  # Marois et al. 2006
                                     'HST/NICMOS2.F215N': (13.40, 0.15),  # Marois et al. 2006
                                     'Subaru/CIAO.CH4s': (13.76, 0.26),  # Marois et al. 2006
                                     'Subaru/CIAO.K': (13.37, 0.12),  # Marois et al. 2006
                                     'Subaru/CIAO.Lp': (12.44, 0.22)}},  # Marois et al. 2006

            'PZ Tel B': {'distance': 47.13,
                         'app_mag': {'Paranal/SPHERE.ZIMPOL_R_PRIM': (17.84, 0.31),  # Maire et al. 2015
                                     'Paranal/SPHERE.ZIMPOL_I_PRIM': (15.16, 0.12),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (11.78, 0.19),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (11.65, 0.19),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (11.56, 0.09),  # Maire et al. 2015
                                     'Paranal/SPHERE.IRDIS_D_K12_2': (11.29, 0.10),  # Maire et al. 2015
                                     'Paranal/NACO.J': (12.47, 0.20),  # Biller et al. 2010
                                     'Paranal/NACO.H': (11.93, 0.14),  # Biller et al. 2010
                                     'Paranal/NACO.Ks': (11.53, 0.07),  # Biller et al. 2010
                                     'Paranal/NACO.NB405': (10.94, 0.07),  # Stolker et al. in prep.
                                     'Paranal/NACO.Lp': (11.27, 0.13),  # Stolker et al. in prep
                                     'Paranal/NACO.Mp': (11.08, 0.03),  # Stolker et al. in prep.
                                     'Gemini/NICI.ED286': (11.68, 0.14),  # Biller et al. 2010
                                     'Gemini/NIRI.H2S1v2-1-G0220': (11.39, 0.14)}},  # Biller et al. 2010

            'kappa And b': {'distance': 50.06,
                            'app_mag': {'Subaru/CIAO.J': (15.86, 0.21),  # Bonnefoy et al. 2014
                                        'Subaru/CIAO.H': (14.95, 0.13),  # Bonnefoy et al. 2014
                                        'Subaru/CIAO.Ks': (14.32, 0.09),  # Bonnefoy et al. 2014
                                        'Keck/NIRC2.Lp': (13.12, 0.1),  # Bonnefoy et al. 2014
                                        # 'Keck/NIRC2.NB_4.05': (13.0, 0.2),  # Bonnefoy et al. 2014
                                        'LBT/LMIRCam.M_77K': (13.3, 0.3)}},  # Bonnefoy et al. 2014

            'ROXs 42B b': {'distance': 144.16,
                           'app_mag': {'Keck/NIRC2.J': (16.91, 0.11),  # Daemgen et al. 2017
                                       'Keck/NIRC2.H': (15.88, 0.05),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Ks': (15.01, 0.06),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Lp': (13.97, 0.06),  # Daemgen et al. 2017
                                       # 'Keck/NIRC2.NB_4.05': (13.90, 0.08),  # Daemgen et al. 2017
                                       'Keck/NIRC2.Ms': (14.01, 0.23)}},  # Daemgen et al. 2017

            'GJ 504 b': {'distance': 17.54,
                         'app_mag': {'Paranal/SPHERE.IRDIS_D_Y23_2': (20.98, 0.20),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_Y23_3': (20.14, 0.09),  # Bonnefoy et al. 2018
                                     # 'Paranal/SPHERE.IRDIS_D_J23_2': (>21.28, ),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_J23_3': (19.01, 0.17),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_H23_2': (18.95, 0.30),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_H23_3': (21.81, 0.35),  # Bonnefoy et al. 2018
                                     'Paranal/SPHERE.IRDIS_D_K12_1': (18.77, 0.20),  # Bonnefoy et al. 2018
                                     # 'Paranal/SPHERE.IRDIS_D_K12_2': (>19.96, ),  # Bonnefoy et al. 2018
                                     'Subaru/CIAO.J': (19.78, 0.10),  # Janson et al. 2013
                                     'Subaru/CIAO.H': (20.01, 0.14),  # Janson et al. 2013
                                     'Subaru/CIAO.Ks': (19.38, 0.11),  # Janson et al. 2013
                                     'Subaru/CIAO.CH4s': (19.58, 0.13),  # Janson et al. 2013
                                     # 'Subaru/CIAO.CH4l': (>20.63, ),  # Janson et al. 2013
                                     # 'LBTI/LMIRcam.L_NB6': (17.59, 0.17),  # Skemer et al. 2016
                                     # 'LBTI/LMIRcam.L_NB7': (16.47, 0.19),  # Skemer et al. 2016
                                     # 'LBTI/LMIRcam.L_NB8': (15.85, 0.17),  # Skemer et al. 2016
                                     'Subaru/IRCS.Lp': (16.70, 0.17)}},  # Kuzuhara et al. 2013

            'GU Psc b': {'distance': 47.61,
                         'app_mag': {'Gemini/GMOS-S.z': (21.75, 0.07),  # Naud et al. 2014
                                     'CFHT/Wircam.Y': (19.4, 0.05),  # Naud et al. 2014
                                     'CFHT/Wircam.J': (18.12, 0.03),  # Naud et al. 2014
                                     'CFHT/Wircam.H': (17.70, 0.03),  # Naud et al. 2014
                                     'CFHT/Wircam.Ks': (17.40, 0.03),  # Naud et al. 2014
                                     'WISE/WISE.W1': (17.17, 0.33),  # Naud et al. 2014
                                     'WISE/WISE.W2': (15.41, 0.22)}},  # Naud et al. 2014

            '2M1207 ABb': {'distance': 47.2,
                           'app_mag': {'Paranal/NACO.J': (15.47, 0.30),  # Delorme et al. 2013
                                       'Paranal/NACO.H': (14.27, 0.20),  # Delorme et al. 2013
                                       'Paranal/NACO.Ks': (13.67, 0.20),  # Delorme et al. 2013
                                       'Paranal/NACO.Lp': (12.67, 0.10)}},  # Delorme et al. 2013

            '1RXS 1609 B': {'distance': 139.67,
                            'app_mag': {'Gemini/NIRI.J-G0202w': (17.90, 0.12),  # Lafreniere et al. 2008
                                        'Gemini/NIRI.H-G0203w': (16.87, 0.07),  # Lafreniere et al. 2008
                                        'Gemini/NIRI.K-G0204w': (16.17, 0.18),  # Lafreniere et al. 2008
                                        # 'MMT/Clio.3.1': (15.65, 0.21),  # Bailey et al. 2013
                                        # 'MMT/Clio.3.3': (15.2, 0.16),  # Bailey et al. 2013
                                        'Gemini/NIRI.Lprime-G0207w': (14.8, 0.3)}},  # Lafreniere et al. 2010

            'GSC 06214 B': {'distance': 108.84,
                            'app_mag': {'MKO/NSFCam.J': (16.24, 0.04),  # Ireland et al. 2011
                                        'MKO/NSFCam.H': (15.55, 0.04),  # Ireland et al. 2011
                                        'MKO/NSFCam.Kp': (14.95, 0.05),  # Ireland et al. 2011
                                        'MKO/NSFCam.Lp': (13.75, 0.07),  # Ireland et al. 2011
                                        'LBT/LMIRCam.M_77K': (13.75, 0.3)}}}  # Bailey et al. 2013

    return data
