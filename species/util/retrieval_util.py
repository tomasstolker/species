import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, CubicSpline
from scipy.ndimage.filters import gaussian_filter

from petitRADTRANS_ck_test_speed import nat_cst as nc
from poor_mans_nonequ_chem_FeH.poor_mans_nonequ_chem.poor_mans_nonequ_chem import interpol_abundances


def pt_ret_model(T3, delta, alpha, tint, press, FeH, CO, conv=True):
    """
    Self-luminous retrieval P-T model.

    It has 7 free parameters:

    T3 = np.array([t1, t2, t3]): temperature points to be added on top
      of the radiative Eddington structure (above tau = 0.1).
      Use spline interpolation, t1 < t2 < t3 < tconnect as prior.

    delta: proportionality factor in tau = delta * press_cgs**alpha

    alpha: power law index in tau = delta * press_cgs**alpha
       For the tau model: use proximity to kappa_rosseland photosphere
       as prior.

    tint: internal temperature of the Eddington model

    press: input pressure profile in bar

    FeH: metallicity for the nabla_ad interpolation

    CO: C/O for the nabla_ad interpolation

    conv: enforce convective adiabat yes/no
    """

    # Go grom bar to cgs
    press_cgs = press*1e6

    # Calculate the optical depth
    tau = delta*press_cgs**alpha

    # This is the eddington temperature
    tedd = (3./4.*tint**4.*(2./3.+tau))**0.25

    ab = interpol_abundances(CO*np.ones_like(tedd), FeH*np.ones_like(tedd), tedd, press)

    nabla_ad = ab['nabla_ad']

    # Enforce convective adiabat
    if conv:
        # Calculate the current, radiative temperature gradient
        nab_rad = np.diff(np.log(tedd))/np.diff(np.log(press_cgs))

        # Extend to array of same length as pressure structure
        nabla_rad = np.ones_like(tedd)
        nabla_rad[0] = nab_rad[0]
        nabla_rad[-1] = nab_rad[-1]
        nabla_rad[1:-1] = (nab_rad[1:]+nab_rad[:-1])/2.

        # Where is the atmosphere convectively unstable?
        conv_index = nabla_rad > nabla_ad

        for i in range(10):

            if i == 0:
                t_take = copy.copy(tedd)
            else:
                t_take = copy.copy(tfinal)

            ab = interpol_abundances(CO*np.ones_like(t_take), FeH*np.ones_like(t_take), t_take, press)

            nabla_ad = ab['nabla_ad']

            # Calculate the average nabla_ad between the layers
            nabla_ad_mean = nabla_ad
            nabla_ad_mean[1:] = (nabla_ad[1:]+nabla_ad[:-1])/2.

            # What are the increments in temperature due to convection
            tnew = nabla_ad_mean[conv_index]*np.mean(np.diff(np.log(press_cgs)))

            # What is the last radiative temperature?
            tstart = np.log(t_take[~conv_index][-1])

            # Integrate and translate to temperature from log(temperature)
            tnew = np.exp(np.cumsum(tnew)+tstart)

            # Add upper radiative and
            # lower conective part into one single array
            tfinal = copy.copy(t_take)
            tfinal[conv_index] = tnew

            if np.max(np.abs(t_take-tfinal)/t_take) < 0.01:
                # print('n_ad', 1./(1.-nabla_ad[conv_index]))
                break

    else:
        tfinal = tedd

    # Add the three temperature-point P-T description above tau = 0.1
    def press_tau(tau):
        # Returns the pressure at a given tau, in cgs
        return (tau/delta)**(1./alpha)

    # Where is the uppermost pressure of the Eddington radiative structure?
    p_bot_spline = press_tau(0.1)

    for i_intp in range(2):

        if i_intp == 0:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 4)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the radiative+convective solution
            support_points_high = 1e1**np.arange(np.log10(p_bot_spline), np.log10(press_cgs[-1]), np.diff(np.log10(support_points_low))[0])

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)
            support_points[:4] = support_points_low
            support_points[4:] = support_points_high[1:]

        else:

            # Create the pressure coordinates for the spline support nodes at low pressure
            support_points_low = np.logspace(np.log10(press_cgs[0]), np.log10(p_bot_spline), 7)

            # Create the pressure coordinates for the spline support nodes at high pressure,
            # the corresponding temperatures for these nodes will be taken from the radiative+convective solution
            support_points_high = np.logspace(np.log10(p_bot_spline), np.log10(press_cgs[-1]), 7)

            # Combine into one support node array, don't add the p_bot_spline point twice.
            support_points = np.zeros(len(support_points_low)+len(support_points_high)-1)
            support_points[:7] = support_points_low
            support_points[7:] = support_points_high[1:]

        # Define the temperature values at the node points.
        t_support = np.zeros_like(support_points)

        if i_intp == 0:
            tfintp = interp1d(press_cgs, tfinal)

            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low))-1] = tfintp(p_bot_spline)

            # The temperature at pressures below p_bot_spline (free parameters)
            t_support[:(int(len(support_points_low))-1)] = T3
            # t_support[:3] = tfintp(support_points_low)

            # The temperature at pressures above p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low)):] = tfintp(support_points[(int(len(support_points_low))):])

        else:
            tfintp1 = interp1d(press_cgs, tret)

            t_support[:(int(len(support_points_low))-1)] = tfintp1(support_points[:(int(len(support_points_low))-1)])

            tfintp = interp1d(press_cgs, tfinal)

            # The temperature at p_bot_spline (from the radiative-convectice solution)
            t_support[int(len(support_points_low))-1] = tfintp(p_bot_spline)

            # print('diff', t_connect_calc - tfintp(p_bot_spline))
            t_support[int(len(support_points_low)):] = tfintp(support_points[(int(len(support_points_low))):])

        # Make the temperature spline interpolation to be returned to the user
        # tret = spline(np.log10(support_points), t_support, np.log10(press_cgs), order = 3)

        cs = CubicSpline(np.log10(support_points), t_support)
        tret = cs(np.log10(press_cgs))

    # Return the temperature, the pressure at tau = 1, and the temperature at the connection point.
    # The last two are needed for the priors on the P-T profile.
    return tret, press_tau(1.)/1e6, tfintp(p_bot_spline)


def pt_spline_interp(knot_press,
                     knot_temp,
                     pressure):

    pt_interp = CubicSpline(np.log10(knot_press), knot_temp)

    return pt_interp(np.log10(pressure))


def calc_spectrum_clear(rt_object,
                        press,
                        temp,
                        logg,
                        co,
                        feh,
                        log_p_quench,
                        abund_in=None,
                        half=False):

    if abund_in is None:
        # chemical equilibrium

        # create arrays for constant values of C/O and Fe/H
        co_list = np.full(press.shape, co)
        feh_list = np.full(press.shape, feh)

        # interpolate the abundances, following chemical equilibrium
        abund_out = interpol_abundances(co_list, feh_list, temp, press, Pquench_carbon=10.**log_p_quench)

        # extract the mean molecular weight
        mmw = abund_in['MMW']

    else:
        # free abundances

        # initiate abundance dictionary and total abundance of metals
        abund_out = {}
        metal_sum = 0.

        for item in abund_in:
            # add abundance to dictionary
            abund_out[item] = np.ones_like(press)*10.**abund_in[item]

            # update total metal abundance
            metal_sum += 10.**abund_in[item]

        # abundance of H2 and He
        ab_h2_he = 1. - metal_sum

        # add H2 and He abundances to the dictionary
        abund_out['H2'] = np.ones_like(press)*ab_h2_he*0.75
        abund_out['He'] = np.ones_like(press)*ab_h2_he*0.25

        # calculate the mean moleculair weight
        mmw = return_mmw(abund_out)

    # extract every three levels if half=True

    if half:
        temp = temp[::3]
        press = press[::3]
        mmw = mmw[::3]

    # create a dictionary with the abundances by replacing species ending with _all_iso

    abundances = {}

    if half:
        for item in rt_object.line_species:
            # abundances[item] = abund_out[item.replace('_all_iso', '')][::3]
            abundances[item] = abund_out[item][::3]

        abundances['H2'] = abund_out['H2'][::3]
        abundances['He'] = abund_out['He'][::3]

    else:
        for item in rt_object.line_species:
            # abundances[item] = abund_out[item.replace('_all_iso', '')]
            abundances[item] = abund_out[item]

        abundances['H2'] = abund_out['H2']
        abundances['He'] = abund_out['He']

    # Corretion for the nuclear spin degeneracy that was not included in the partition function
    # See Charnay et al. (2018)

    if abund_in is None and 'FeH' in abundances:
        abundances['FeH'] = abundances['FeH']/2.

    # calculate the emission spectrum
    rt_object.calc_flux(temp, abundances, 10.**logg, mmw)

    # convert frequency (Hz) to wavelength (cm)
    wlen = nc.c/rt_object.freq

    # return wavelength (micron) and flux (W m-2 um-1)
    return 1e4*wlen, 1e-7*rt_object.flux*nc.c/wlen**2.


def calc_spectrum_clouds(rt_object,
                         press,
                         temp,
                         CO,
                         FeH,
                         log_p_quench,
                         log_X_cloud_base_Fe,
                         log_X_cloud_base_MgSiO3,
                         fsed_Fe,
                         fsed_MgSiO3,
                         Kzz,
                         logg,
                         sigma_lnorm,
                         half=False,
                         plotting=False):

    COs = CO * np.ones_like(press)
    FeHs = FeH * np.ones_like(press)

    abundances_interp = interpol_abundances(COs, FeHs, temp, press, Pquench_carbon=1e1**log_p_quench)

    MMW = abundances_interp['MMW']

    P_base_Fe = simple_cdf_Fe(press, temp, FeH, CO, np.mean(MMW), plotting)
    P_base_MgSiO3 = simple_cdf_MgSiO3(press, temp, FeH, CO, np.mean(MMW), plotting=plotting)

    abundances = {}

    abundances['Fe(c)'] = np.zeros_like(temp)

    abundances['Fe(c)'][press < P_base_Fe] = \
          1e1**log_X_cloud_base_Fe * (press[press <= P_base_Fe]/P_base_Fe)**fsed_Fe

    abundances['MgSiO3(c)'] = np.zeros_like(temp)

    abundances['MgSiO3(c)'][press < P_base_MgSiO3] = \
          1e1**log_X_cloud_base_MgSiO3 * (press[press <= P_base_MgSiO3]/P_base_MgSiO3)**fsed_MgSiO3

    if half:
        abundances['Fe(c)'] = abundances['Fe(c)'][::3]
        abundances['MgSiO3(c)'] = abundances['MgSiO3(c)'][::3]

    if half:
        for species in rt_object.line_species:
            abundances[species] = abundances_interp[species.replace('_all_iso', '')][::3]

        abundances['H2'] = abundances_interp['H2'][::3]
        abundances['He'] = abundances_interp['He'][::3]

    else:
        for species in rt_object.line_species:
            abundances[species] = abundances_interp[species.replace('_all_iso', '')]

        abundances['H2'] = abundances_interp['H2']
        abundances['He'] = abundances_interp['He']

    # Corretion for the nuclear spin degeneracy that was not included in the partition function
    # See Charnay et al. (2018)

    if 'FeH' in abundances:
        abundances['FeH'] = abundances['FeH']/2.

    Kzz_use = (1e1**Kzz) * np.ones_like(press)

    if half:
        temp = temp[::3]
        press = press[::3]
        MMW = MMW[::3]
        Kzz_use = Kzz_use[::3]

    fseds = {}
    fseds['Fe(c)'] = fsed_Fe
    fseds['MgSiO3(c)'] = fsed_MgSiO3

    if plotting:
        plt.plot(abundances['CO_all_iso'], press, label='CO')
        plt.plot(abundances['CH4'], press, label='CH4')
        plt.plot(abundances['H2O'], press, label='H2O')
        plt.xlim([1e-10, 1.])
        plt.ylim([press[-1], press[0]])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Mass fraction')
        plt.ylabel('Pressure (bar)')
        plt.axhline(1e1**log_p_quench)
        plt.legend(loc='best')
        plt.savefig('abundances.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(temp, press)
        plt.axhline(P_base_Fe, label='Cloud deck Fe')
        plt.axhline(P_base_MgSiO3, label='Cloud deck MgSiO3')
        plt.yscale('log')
        plt.ylim([1e3, 1e-6])
        plt.xlim([0., 4000.])
        plt.savefig('pt_cloud_deck.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(abundances['Fe(c)'], press)
        plt.axhline(P_base_Fe)
        plt.yscale('log')
        if np.count_nonzero(abundances['Fe(c)']) > 0:
            plt.xscale('log')
        plt.ylim([1e3, 1e-6])
        plt.xlim([1e-10, 1.])
        plt.title('fsed_Fe = '+str(fsed_Fe)+' lgK='+str(Kzz)+' X_b = '+str(log_X_cloud_base_Fe))
        plt.savefig('fe_clouds.pdf', bbox_inches='tight')
        plt.clf()

        plt.plot(abundances['MgSiO3(c)'], press)
        plt.axhline(P_base_MgSiO3)
        plt.yscale('log')
        if np.count_nonzero(abundances['MgSiO3(c)']) > 0:
            plt.xscale('log')
        plt.ylim([1e3, 1e-6])
        plt.xlim([1e-10, 1.])
        plt.title('fsed_MgSiO3 = '+str(fsed_MgSiO3)+' lgK='+str(Kzz)+' X_b = '+str(log_X_cloud_base_MgSiO3))
        plt.savefig('mgsio3_clouds.pdf', bbox_inches='tight')
        plt.clf()

    # Turn off clouds
    # abundances['MgSiO3(c)'] = np.zeros_like(press)
    # abundances['Fe(c)'] = np.zeros_like(press)

    rt_object.calc_flux(temp, abundances, 1e1**logg, MMW, Kzz=Kzz_use, fsed=fseds, sigma_lnorm=sigma_lnorm)

    wlen_micron = nc.c/rt_object.freq/1e-4
    wlen = nc.c/rt_object.freq
    flux = rt_object.flux

    # convert flux f_nu to f_lambda
    f_lambda = flux*nc.c/wlen**2.
    # convert to flux per m^2 (from flux per cm^2) cancels with step below
    # f_lambda = f_lambda * 1e4
    # convert to flux per micron (from flux per cm) cancels with step above
    # f_lambda = f_lambda * 1e-4
    # convert from ergs to Joule
    f_lambda = f_lambda * 1e-7

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([1e2,1e-6])
    # plt.ylabel('P (bar)')
    # plt.xlabel('Average particle size of MgSiO3 particles (microns)')
    # plt.plot(rt_object.r_g[:,rt_object.cloud_species.index('MgSiO3(c)')]/1e-4, press)
    # plt.savefig('mgsio3_size.png')
    # plt.show()
    # plt.clf()

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.ylim([1e2,1e-6])
    # plt.ylabel('P (bar)')
    # plt.xlabel('Average particle size of Fe particles (microns)')
    # plt.plot(rt_object.r_g[:,rt_object.cloud_species.index('Fe(c)')]/1e-4, press)
    # plt.savefig('fe_size.png')
    # plt.show()rt_object
    # plt.clf()

    # return wlen_micron, f_lambda, rt_object.pphot, rt_object.tau_pow, np.mean(rt_object.tau_cloud)
    return wlen_micron, f_lambda


def return_mmw(abundances):

    mol_weight = {}
    mol_weight['H2'] = 2.
    mol_weight['He'] = 4.
    mol_weight['H2O'] = 18.
    mol_weight['CH4'] = 16.
    mol_weight['CO2'] = 44.
    mol_weight['CO'] = 28.
    mol_weight['Na'] = 23.
    mol_weight['K'] = 39.
    mol_weight['NH3'] = 17.
    mol_weight['HCN'] = 27.
    mol_weight['C2H2,acetylene'] = 26.
    mol_weight['PH3'] = 34.
    mol_weight['H2S'] = 34.
    mol_weight['VO'] = 67.
    mol_weight['TiO'] = 64.

    mmw = 0.

    for key in abundances:
        if key == 'CO_all_iso':
            mmw += abundances[key]/mol_weight['CO']
        else:
            mmw += abundances[key]/mol_weight[key]

    return 1./mmw


#############################################################
# To calculate X_Fe from [Fe/H], C/O
#############################################################

# metal species
metals = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'Ca', 'Ti', 'V', 'Fe', 'Ni']

# solar abundances, [Fe/H] = 0, from Asplund+ 2009
nfracs = {}
nfracs['H'] = 0.9207539305
nfracs['He'] = 0.0783688694
nfracs['C'] = 0.0002478241
nfracs['N'] = 6.22506056949881e-05
nfracs['O'] = 0.0004509658
nfracs['Na'] = 1.60008694353205e-06
nfracs['Mg'] = 3.66558742055362e-05
nfracs['Al'] = 2.595e-06
nfracs['Si'] = 2.9795e-05
nfracs['P'] = 2.36670201997668e-07
nfracs['S'] = 1.2137900734604e-05
nfracs['Cl'] = 2.91167958499589e-07
nfracs['K'] = 9.86605611925677e-08
nfracs['Ca'] = 2.01439011429255e-06
nfracs['Ti'] = 8.20622804366359e-08
nfracs['V'] = 7.83688694089992e-09
nfracs['Fe'] = 2.91167958499589e-05
nfracs['Ni'] = 1.52807116806281e-06

# atomic masses
masses = {}
masses['H'] = 1.
masses['He'] = 4.
masses['C'] = 12.
masses['N'] = 14.
masses['O'] = 16.
masses['Na'] = 23.
masses['Mg'] = 24.3
masses['Al'] = 27.
masses['Si'] = 28.
masses['P'] = 31.
masses['S'] = 32.
masses['Cl'] = 35.45
masses['K'] = 39.1
masses['Ca'] = 40.
masses['Ti'] = 47.9
masses['V'] = 51.
masses['Fe'] = 55.8
masses['Ni'] = 58.7


def return_XFe(FeH, CO):

    nfracs_use = copy.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    XFe = masses['Fe']*nfracs_use['Fe']

    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    XFe = XFe / add

    return XFe


def return_XMgSiO3(FeH, CO):

    nfracs_use = copy.copy(nfracs)

    for spec in nfracs.keys():

        if (spec != 'H') and (spec != 'He'):
            nfracs_use[spec] = nfracs[spec]*1e1**FeH

    nfracs_use['O'] = nfracs_use['C']/CO

    nfracs_mgsio3 = np.min([nfracs_use['Mg'], nfracs_use['Si'], nfracs_use['O']/3.])

    masses_mgsio3 = masses['Mg'] + masses['Si'] + 3. * masses['O']

    Xmgsio3 = masses_mgsio3*nfracs_mgsio3

    add = 0.
    for spec in nfracs_use.keys():
        add += masses[spec]*nfracs_use[spec]

    Xmgsio3 = Xmgsio3 / add

    return Xmgsio3


#############################################################
# Fe saturation pressure, from Ackerman & Marley (2001), including erratum (P_vap is in bar, not cgs!)
#############################################################

def return_T_cond_Fe(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: np.exp(15.71 - 47664./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


def return_T_cond_Fe_l(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)
    P_vap = lambda x: np.exp(9.86 - 37120./x)

    XFe = return_XFe(FeH, CO)

    return P_vap(T)/(XFe*MMW/masses['Fe']), T


def return_T_cond_Fe_comb(FeH, CO, MMW = 2.33):

    P1, T1 = return_T_cond_Fe(FeH, CO, MMW)
    P2, T2 = return_T_cond_Fe_l(FeH, CO, MMW)

    retP = np.zeros_like(P1)

    index = P1<P2

    retP[index] = P1[index]
    retP[~index] = P2[~index]

    return retP, T2


def return_T_cond_MgSiO3(FeH, CO, MMW = 2.33):

    T = np.linspace(100.,10000.,1000)

    P_vap = lambda x: np.exp(25.37 - 58663./x)

    Xmgsio3 = return_XMgSiO3(FeH, CO)

    m_mgsio3 =  masses['Mg'] + masses['Si'] + 3. * masses['O']

    return P_vap(T)/(Xmgsio3*MMW/m_mgsio3), T


# if plotting:
#
#     #FeHs = np.linspace(-0.5, 2., 5)
#     #COs = np.linspace(0.3, 1.2, 5)
#     FeHs = [0.]
#     COs = [0.55]
#
#     for FeH in FeHs:
#         for CO in COs:
#             P, T = return_T_cond_Fe(FeH, CO)
#             plt.plot(T,P, label = 'Fe(c), [Fe/H] = '+str(FeH)+', C/O = '+str(CO), color = 'black')
#             P, T = return_T_cond_Fe_l(FeH, CO)
#             plt.plot(T,P, '--', label = 'Fe(l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
#             P, T = return_T_cond_Fe_comb(FeH, CO)
#             plt.plot(T,P, ':', label = 'Fe(c+l), [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
#             P, T = return_T_cond_MgSiO3(FeH, CO)
#             plt.plot(T,P, label = 'MgSiO3, [Fe/H] = '+str(FeH)+', C/O = '+str(CO))
#
#     plt.yscale('log')
#     plt.xlim([0., 5000.])
#     plt.ylim([1e5,1e-10])
#     plt.legend(loc = 'best', frameon = False)
#     plt.savefig('condensation.pdf', bbox_inches='tight')
#     plt.clf()


def simple_cdf_Fe(press, temp, FeH, CO, MMW=2.33, plotting=False):

    Pc, Tc = return_T_cond_Fe_comb(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    #print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color = 'red', linestyle = '--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2,1e-6])
        plt.savefig('condensation_fe.pdf', bbox_inches='tight')
        plt.clf()

    return P_cloud


def simple_cdf_MgSiO3(press, temp, FeH, CO, MMW=2.33, plotting=False):

    Pc, Tc = return_T_cond_MgSiO3(FeH, CO, MMW)
    index = (Pc > 1e-8) & (Pc < 1e5)
    Pc, Tc = Pc[index], Tc[index]
    tcond_p = interp1d(Pc, Tc)
    # print(Pc, press)
    Tcond_on_input_grid = tcond_p(press)

    Tdiff = Tcond_on_input_grid - temp
    diff_vec = Tdiff[1:]*Tdiff[:-1]
    ind_cdf = (diff_vec < 0.)
    if len(diff_vec[ind_cdf]) > 0:
        P_clouds = (press[1:]+press[:-1])[ind_cdf]/2.
        P_cloud = P_clouds[-1]
    else:
        P_cloud = 1e-8

    if plotting:
        plt.plot(temp, press)
        plt.plot(Tcond_on_input_grid, press)
        plt.axhline(P_cloud, color='red', linestyle='--')
        plt.yscale('log')
        plt.xlim([0., 3000.])
        plt.ylim([1e2, 1e-6])
        plt.savefig('condensation_mgsio3.pdf', bbox_inches='tight')
        plt.clf()

    return P_cloud

# if plotting:
#     kappa_IR = 0.01
#     gamma = 0.4
#     T_int = 200.
#     T_equ = 1550.
#     gravity = 1e1**2.45
#
#     pressures = np.logspace(-6, 2, 100)
#
#     temperature = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)
#
#     simple_cdf_Fe(pressures, temperature, 0., 0.55)
#     simple_cdf_MgSiO3(pressures, temperature, 0., 0.55)

def convolve(input_wavelength, input_flux, instrument_res):
    # From talking to Ignas: delta lambda of resolution element
    # is FWHM of the LSF's standard deviation, hence:
    sigma_lsf = 1./instrument_res/(2.*np.sqrt(2.*np.log(2.)))

    # The input spacing of petitRADTRANS is 1e3, but just compute
    # it to be sure, or more versatile in the future.
    # Also, we have a log-spaced grid, so the spacing is constant
    # as a function of wavelength
    spacing = np.mean(2.*np.diff(input_wavelength)/(input_wavelength[1:]+input_wavelength[:-1]))

    # Calculate the sigma to be used in the gauss filter in units
    # of input wavelength bins
    sigma_lsf_gauss_filter = sigma_lsf/spacing

    return gaussian_filter(input_flux, sigma=sigma_lsf_gauss_filter, mode='nearest')
