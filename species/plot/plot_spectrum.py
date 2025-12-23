"""
Module with a function for plotting a spectral energy distribution
that includes photometric and/or spectral data and/or models.
"""

import math

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typeguard import typechecked
from matplotlib.ticker import AutoMinorLocator

from species.core.box import (
    ModelBox,
    ObjectBox,
    PhotometryBox,
    ResidualsBox,
    SpectrumBox,
    SynphotBox,
)
from species.read.read_filter import ReadFilter
from species.util.core_util import print_section
from species.util.data_util import convert_units
from species.util.plot_util import create_model_label, create_param_format


@typechecked
def plot_spectrum(
    boxes: list,
    filters: Optional[List[str]] = None,
    residuals: Optional[ResidualsBox] = None,
    plot_kwargs: Optional[List[Optional[dict]]] = None,
    envelope: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ylim_res: Optional[Tuple[float, float]] = None,
    scale: Optional[Tuple[str, str]] = None,
    title: Optional[str] = None,
    offset: Optional[Tuple[float, float]] = None,
    legend: Optional[
        Union[
            str,
            dict,
            Tuple[float, float],
            List[Optional[Union[dict, str, Tuple[float, float]]]],
        ]
    ] = None,
    figsize: Optional[Tuple[float, float]] = (6.0, 3.0),
    object_type: str = "planet",
    quantity: str = "flux density",
    output: Optional[str] = None,
    leg_param: Optional[List[str]] = None,
    param_fmt: Optional[Dict[str, str]] = None,
    grid_hspace: float = 0.1,
    inc_model_name: bool = False,
    units: Tuple[str, str] = ("um", "W m-2 um-1"),
    font_size: Optional[Dict[str, float]] = None,
) -> mpl.figure.Figure:
    """
    Function for plotting a spectral energy distribution and combining
    various data such as spectra, photometric fluxes, model spectra,
    synthetic photometry, fit residuals, and filter profiles.

    Parameters
    ----------
    boxes : list(species.core.box)
        Boxes with data that will be included in the plot.
    filters : list(str), None
        Filter names for which the transmission profile is plotted.
        Not plotted if set to ``None``.
    residuals : species.core.box.ResidualsBox, None
        Box with residuals of a fit. Not plotted if set to ``None``.
    plot_kwargs : list(dict), None
        List with dictionaries of keyword arguments for each box.
        For example, if the ``boxes`` are a ``ModelBox`` and
        ``ObjectBox``:

        .. code-block:: python

            plot_kwargs=[{'ls': '-', 'lw': 1., 'color': 'black'},
                         {'spectrum_1': {'marker': 'o', 'ms': 3., 'color': 'tab:brown', 'ls': 'none'},
                          'spectrum_2': {'marker': 'o', 'ms': 3., 'color': 'tab:blue', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_H23_3': {'marker': 's', 'ms': 4., 'color': 'tab:cyan', 'ls': 'none'},
                          'Paranal/SPHERE.IRDIS_D_K12_1': [{'marker': 's', 'ms': 4., 'color': 'tab:orange', 'ls': 'none'},
                                                           {'marker': 's', 'ms': 4., 'color': 'tab:red', 'ls': 'none'}],
                          'Paranal/NACO.Lp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'},
                          'Paranal/NACO.Mp': {'marker': 's', 'ms': 4., 'color': 'tab:green', 'ls': 'none'}}]

        For an ``ObjectBox``, the dictionary contains items for the
        different spectrum and filter names stored with
        :func:`~species.data.database.Database.add_object`. In case
        both and ``ObjectBox`` and a ``SynphotBox`` are provided,
        then the latter can be set to ``None`` in order to use the
        same (but open) symbols as the data from the ``ObjectBox``.
        Note that if a filter name is duplicated in an ``ObjectBox``
        (Paranal/SPHERE.IRDIS_D_K12_1 in the example) then a list
        with two dictionaries should be provided. Colors are
        automatically chosen if ``plot_kwargs`` is set to ``None``.
    envelope : bool
        Plot an envelope instead of the individual samples in case
        the list of ``boxes`` contains a list with
        :class:`~species.core.box.ModelBox` objects from
        :func:`~species.data.database.Database.get_mcmc_spectra`
        or :func:`~species.data.database.Database.get_retrieval_spectra`.
        The envelopes show the 68 and 99.7 percent confidence intervals,
        so :math:`1\\sigma` and :math:`3\\sigma` in case of Gaussian
        distributions.
    xlim : tuple(float, float)
        Limits of the wavelength axis.
    ylim : tuple(float, float)
        Limits of the flux axis.
    ylim_res : tuple(float, float), None
        Limits of the residuals axis. Automatically chosen
        (based on the minimum and maximum residual value)
        if set to ``None``.
    scale : tuple(str, str), None
        Scale of the x and y axes ('linear' or 'log').
        The scale is set to ``('linear', 'linear')`` if
        set to ``None``.
    title : str
        Title.
    offset : tuple(float, float), None
        Offset for the label of the x- and y-axis. Default offset is
        used when the argument is set to ``None``.
    legend : str, tuple, dict, list(dict, dict), None
        Location of the legend (str or tuple(float, float))
        or a dictionary with the ``**kwargs`` of
        ``matplotlib.pyplot.legend``, for example
        ``{'loc': 'upper left', 'fontsize: 12.}``. Alternatively,
        a list with two values can be provided to separate the
        model and data handles in two legends. Each of these two
        elements can be set to ``None``. For example,
        ``[None, {'loc': 'upper left', 'fontsize: 12.}]``, if
        only the data points should be included in a legend.
    figsize : tuple(float, float)
        Figure size.
    object_type : str
        Object type ('planet' or 'star'). With 'planet', the radius
        and mass are expressed in Jupiter units. With 'star', the
        radius and mass are expressed in solar units.
    quantity : str
        The quantity of the y-axis ('flux density', 'flux',
        or 'magnitude').
    output : str, None
        Output filename for the plot. The plot is shown in an
        interface window if the argument is set to ``None``.
    leg_param : list(str), None
        List with the parameters to include in the legend of the
        model spectra. Apart from atmospheric parameters (e.g.
        'teff', 'logg', 'radius') also parameters such as 'mass',
        'log_lum', log_lum_atm', and 'log_lum_disk' can be
        included. The default atmospheric parameters are included
        in the legend if the argument is set to ``None``.
    param_fmt : dict(str, str), None
        Dictionary with formats that will be used for the model
        parameter. The parameters are included in the ``legend``
        when plotting the model spectra. Default formats are
        used if the argument of ``param_fmt`` is set to ``None``.
        Formats should provided for example as '.2f' for two
        decimals, '.0f' for zero decimals, and '.1e' for
        exponential notation with one decimal.
    grid_hspace : float
        The relative height spacing between subplots, expressed
        as a fraction of the average axis height. The default
        value is set to 0.1.
    inc_model_name : bool
        Include the model name in the legend of any
        :class:`~species.core.box.ModelBox`.
    units : tuple(str, str), None
        Tuple with the wavelength and flux units. Supported
        units can be found in the docstring of
        :func:`~species.util.data_util.convert_units`.
    font_size : dict(str, float), None
        Dictionary with the font sizes. The keys can be set to
        'xlabel', 'ylabel', 'title', and 'legend'. The values
        should be set to the font sizes. Default font size are
        used when setting the argument to ``None``. The legend
        font size is not used if it is also set with the
        ``legend`` parameter.

    Returns
    -------
    matplotlib.figure.Figure
        The ``Figure`` object that can be used for further
        customization of the plot.
    """

    print_section("Plot spectrum")

    print("Boxes:")
    for item in boxes:
        if isinstance(item, list):
            item_type = item[0].__class__.__name__
            print(f"   - List with {len(item)} x {item_type}")
        else:
            print(f"   - {item.__class__.__name__}")

    print(f"\nObject type: {object_type}")
    print(f"Quantity: {quantity}")
    print(f"Units: {units}")
    print(f"Filter profiles: {filters}")

    print(f"\nFigure size: {figsize}")
    print(f"Legend parameters: {leg_param}")
    print(f"Include model name: {inc_model_name}")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.axisbelow"] = False

    if plot_kwargs is None:
        plot_kwargs = []

    elif plot_kwargs is not None and len(boxes) != len(plot_kwargs):
        raise ValueError(
            f"The number of 'boxes' ({len(boxes)}) should be equal to the "
            f"number of items in 'plot_kwargs' ({len(plot_kwargs)})."
        )

    if leg_param is None:
        leg_param = []

    param_fmt = create_param_format(param_fmt)

    if residuals is not None and filters is not None:
        fig = plt.figure(figsize=figsize)
        grid_sp = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 3, 1])
        grid_sp.update(wspace=0, hspace=grid_hspace, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(grid_sp[1, 0])
        ax2 = plt.subplot(grid_sp[0, 0])
        ax3 = plt.subplot(grid_sp[2, 0])

    elif residuals is not None:
        fig = plt.figure(figsize=figsize)
        grid_sp = mpl.gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        grid_sp.update(wspace=0, hspace=grid_hspace, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(grid_sp[0, 0])
        ax2 = None
        ax3 = plt.subplot(grid_sp[1, 0])

    elif filters is not None:
        fig = plt.figure(figsize=figsize)
        grid_sp = mpl.gridspec.GridSpec(2, 1, height_ratios=[1, 4])
        grid_sp.update(wspace=0, hspace=grid_hspace, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(grid_sp[1, 0])
        ax2 = plt.subplot(grid_sp[0, 0])
        ax3 = None

    else:
        fig = plt.figure(figsize=figsize)
        grid_sp = mpl.gridspec.GridSpec(1, 1)
        grid_sp.update(wspace=0, hspace=grid_hspace, left=0, right=1, bottom=0, top=1)

        ax1 = plt.subplot(grid_sp[0, 0])
        ax2 = None
        ax3 = None

    if residuals is not None:
        labelbottom = False
    else:
        labelbottom = True

    if scale is None:
        scale = ("linear", "linear")

    ax1.set_xscale(scale[0])
    ax1.set_yscale(scale[1])

    if filters is not None:
        ax2.set_xscale(scale[0])

    if residuals is not None:
        ax3.set_xscale(scale[0])

    ax1.tick_params(
        axis="both",
        which="major",
        colors="black",
        labelcolor="black",
        direction="in",
        width=1,
        length=5,
        labelsize=12,
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelbottom=labelbottom,
    )

    ax1.tick_params(
        axis="both",
        which="minor",
        colors="black",
        labelcolor="black",
        direction="in",
        width=1,
        length=3,
        labelsize=12,
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelbottom=labelbottom,
    )

    if filters is not None:
        ax2.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

        ax2.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
            labelbottom=False,
        )

    if residuals is not None:
        ax3.tick_params(
            axis="both",
            which="major",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=5,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
        )

        ax3.tick_params(
            axis="both",
            which="minor",
            colors="black",
            labelcolor="black",
            direction="in",
            width=1,
            length=3,
            labelsize=12,
            top=True,
            bottom=True,
            left=True,
            right=True,
        )

    if scale[0] == "linear":
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

    if scale[1] == "linear":
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    # ax1.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    # ax3.set_yticks([-2., 0., 2.])

    if filters is not None:
        if scale[0] == "linear":
            ax2.xaxis.set_minor_locator(AutoMinorLocator(5))

    if residuals is not None:
        if scale[0] == "linear":
            ax3.xaxis.set_minor_locator(AutoMinorLocator(5))

    if units[0] in ["um", "µm"]:
        x_label = "Wavelength (µm)"
    elif units[0] in ["angstrom", "A", "AA", "Å"]:
        x_label = r"Wavelength ($\AA$)"
    elif units[0] == "Hz":
        x_label = "Frequency (Hz)"
    elif units[0] == "GHz":
        x_label = "Frequency (GHz)"
    else:
        x_label = f"Wavelength ({units[0]})"

    if units[1] in ["W m-2 um-1", "W m-2 µm-1"]:
        y_unit = "W m$^{-2}$ µm$^{-1}$"
    elif units[1] == "W m-2 m-1":
        y_unit = r"W m$^{-2}$ m$^{-1}$"
    elif units[1] == "W m-2 Hz-1":
        y_unit = r"W m$^{-2}$ Hz$^{-1}$"
    elif units[1] in [
        "erg s-1 cm-2 angstrom-1",
        "erg s-1 cm-2 A-1",
        "erg s-1 cm-2 AA-1",
        "erg s-1 cm-2 Å-1",
    ]:
        y_unit = r"erg s$^{-2}$ cm$^{-2}$ $\AA$$^{-1}$"
    elif units[1] == "erg s-1 cm-2 Hz-1":
        y_unit = r"erg s$^{-2}$ cm$^{-2}$ Hz$^{-1}$"
    else:
        y_unit = units[1]

    if font_size is None:
        font_size = {}

    if "xlabel" not in font_size:
        font_size["xlabel"] = 11.0

    if "ylabel" not in font_size:
        font_size["ylabel"] = 11.0

    if "title" not in font_size:
        font_size["title"] = 13.0

    if "legend" not in font_size:
        font_size["legend"] = 9.0

    print(f"Font sizes: {font_size}")

    if residuals is not None and filters is not None:
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.set_xlabel(x_label, fontsize=font_size["xlabel"])

    elif residuals is not None:
        ax1.set_xlabel("")
        ax3.set_xlabel(x_label, fontsize=font_size["xlabel"])

    elif filters is not None:
        ax1.set_xlabel(x_label, fontsize=font_size["xlabel"])
        ax2.set_xlabel("")

    else:
        ax1.set_xlabel(x_label, fontsize=font_size["xlabel"])

    if filters is not None:
        ax2.set_ylabel(r"$T_\lambda$", fontsize=font_size["ylabel"])

    if residuals is not None:
        if quantity == "flux density":
            ax3.set_ylabel(
                r"$\Delta$$F_\lambda$ ($\sigma$)", fontsize=font_size["ylabel"]
            )

        elif quantity == "flux":
            ax3.set_ylabel(
                r"$\Delta$$F_\lambda$ ($\sigma$)", fontsize=font_size["ylabel"]
            )

    if quantity == "magnitude":
        scaling = 1.0
        ax1.set_ylabel("Contrast (mag)", fontsize=font_size["ylabel"])

        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])

    else:
        if ylim is not None:
            ax1.set_ylim(ylim[0], ylim[1])

            ylim = ax1.get_ylim()

            if scale[1] == "linear":
                exponent = math.floor(math.log10(ylim[1]))
                scaling = 10.0**exponent

            else:
                exponent = None
                scaling = 1.0

            if quantity == "flux density":
                if exponent is None:
                    ylabel = rf"$F_\lambda$ ({y_unit})"

                else:
                    ylabel = (
                        r"$F_\lambda$ (10$^{" + str(exponent) + r"}$" + f" {y_unit})"
                    )

            elif quantity == "flux":
                if exponent is None:
                    ylabel = r"$\lambda$$F_\lambda$ (W m$^{-2}$)"

                else:
                    ylabel = (
                        r"$\lambda$$F_\lambda$ (10$^{"
                        + str(exponent)
                        + r"}$ W m$^{-2}$)"
                    )

            ax1.set_ylabel(ylabel, fontsize=font_size["ylabel"])
            ax1.set_ylim(ylim[0] / scaling, ylim[1] / scaling)

            if ylim[0] < 0.0:
                ax1.axhline(
                    0.0, ls="--", lw=0.7, color="gray", dashes=(2, 4), zorder=0.5
                )

        else:
            if quantity == "flux density":
                ax1.set_ylabel(
                    rf"$F_\lambda$ ({y_unit})",
                    fontsize=font_size["ylabel"],
                )

            elif quantity == "flux":
                ax1.set_ylabel(
                    r"$\lambda$$F_\lambda$ (W m$^{-2}$)", fontsize=font_size["ylabel"]
                )

            scaling = 1.0

    if offset is not None and residuals is not None and filters is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and filters is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax2.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None and residuals is not None:
        ax3.get_xaxis().set_label_coords(0.5, offset[0])

        ax1.get_yaxis().set_label_coords(offset[1], 0.5)
        ax3.get_yaxis().set_label_coords(offset[1], 0.5)

    elif offset is not None:
        ax1.get_xaxis().set_label_coords(0.5, offset[0])
        ax1.get_yaxis().set_label_coords(offset[1], 0.5)

    labels_data = []

    for j, box_item in enumerate(boxes):
        flux_scaling = 1.0

        if j < len(boxes):
            plot_kwargs.append(None)

        if isinstance(box_item, (SpectrumBox, ModelBox)):
            wavelength = box_item.wavelength
            flux = box_item.flux

            if isinstance(wavelength[0], (np.float32, np.float64)):
                data_in = np.column_stack([wavelength, flux])
                data_out = convert_units(data_in, units, convert_from=False)

                data_wavel = data_out[:, 0]
                data_flux = data_out[:, 1]

                data_flux = np.array(data_flux, dtype=np.float64)
                flux_masked = np.ma.array(data_flux, mask=np.isnan(data_flux))

                if isinstance(box_item, ModelBox):
                    param = box_item.parameters.copy()

                    label = create_model_label(
                        model_param=param,
                        object_type=object_type,
                        model_name=box_item.model,
                        inc_model_name=inc_model_name,
                        leg_param=leg_param,
                        param_fmt=param_fmt,
                    )

                else:
                    label = None

                if plot_kwargs[j]:
                    kwargs_copy = plot_kwargs[j].copy()

                    if "label" in kwargs_copy:
                        if kwargs_copy["label"] is None:
                            label = None
                        else:
                            label = kwargs_copy["label"]

                        del kwargs_copy["label"]

                    if quantity == "flux":
                        flux_scaling = data_wavel

                    if "zorder" not in kwargs_copy:
                        kwargs_copy["zorder"] = 2.0

                    ax1.plot(
                        data_wavel,
                        flux_scaling * flux_masked / scaling,
                        label=label,
                        **kwargs_copy,
                    )

                else:
                    if quantity == "flux":
                        flux_scaling = data_wavel

                    ax1.plot(
                        data_wavel,
                        flux_scaling * flux_masked / scaling,
                        lw=0.5,
                        label=label,
                        zorder=2,
                    )

            elif isinstance(wavelength[0], (np.ndarray)):
                for i in range(len(wavelength)):
                    data_in = np.column_stack([wavelength[i], flux[i]])
                    data_out = convert_units(data_in, units, convert_from=False)

                    data_wavel = data_out[:, 0]
                    data_flux = data_out[:, 1]

                    data_flux = np.array(data_flux, dtype=np.float64)
                    flux_masked = np.ma.array(data_flux, mask=np.isnan(data_flux))

                    if isinstance(box_item.name[i], bytes):
                        label = box_item.name[i].decode("utf-8")
                    else:
                        label = box_item.name[i]

                    if quantity == "flux":
                        flux_scaling = wavelength

                    ax1.plot(
                        data_wavel,
                        flux_scaling * flux_masked / scaling,
                        lw=0.5,
                        label=label,
                    )

        elif isinstance(box_item, list):
            if envelope:
                spec_list = np.zeros((len(box_item), box_item[0].flux.size))
            else:
                spec_list = None

            for i, item in enumerate(box_item):
                wavelength = item.wavelength
                flux = item.flux

                data_in = np.column_stack([wavelength, flux])
                data_out = convert_units(data_in, units, convert_from=False)

                wavelength = data_out[:, 0]
                flux = data_out[:, 1]

                # data = np.array(flux, dtype=np.float64)
                data = flux.astype(np.float64)
                flux_masked = np.ma.array(data, mask=np.isnan(data))

                if quantity == "flux":
                    flux_scaling = wavelength

                if envelope:
                    spec_list[i] = flux

                else:
                    if plot_kwargs[j]:
                        if "zorder" not in plot_kwargs[j]:
                            plot_kwargs[j]["zorder"] = 1.0

                        ax1.plot(
                            wavelength,
                            flux_scaling * flux_masked / scaling,
                            **plot_kwargs[j],
                        )
                    else:
                        ax1.plot(
                            wavelength,
                            flux_scaling * flux_masked / scaling,
                            color="gray",
                            lw=0.2,
                            alpha=0.5,
                            zorder=1,
                        )

            if envelope:
                spec_percent = np.percentile(spec_list, [0.3, 16.0, 84.0, 99.7], axis=0)

                if plot_kwargs[j]:
                    if "zorder" not in plot_kwargs[j]:
                        plot_kwargs[j]["zorder"] = 1.0

                    if "alpha" in plot_kwargs[j]:
                        del plot_kwargs[j]["alpha"]

                    ax1.fill_between(
                        x=wavelength,
                        y1=flux_scaling * spec_percent[0] / scaling,
                        y2=flux_scaling * spec_percent[3] / scaling,
                        alpha=0.4,
                        **plot_kwargs[j],
                    )

                    ax1.fill_between(
                        x=wavelength,
                        y1=flux_scaling * spec_percent[1] / scaling,
                        y2=flux_scaling * spec_percent[2] / scaling,
                        alpha=1.0,
                        **plot_kwargs[j],
                    )

                else:
                    ax1.fill_between(
                        x=wavelength,
                        y1=flux_scaling * spec_percent[0] / scaling,
                        y2=flux_scaling * spec_percent[3] / scaling,
                        color="peachpuff",
                        alpha=0.4,
                        zorder=1,
                        linewidth=0.0,
                    )

                    ax1.fill_between(
                        x=wavelength,
                        y1=flux_scaling * spec_percent[1] / scaling,
                        y2=flux_scaling * spec_percent[2] / scaling,
                        color="peachpuff",
                        alpha=1.0,
                        zorder=1,
                        linewidth=0.0,
                    )

        elif isinstance(box_item, PhotometryBox):
            label_check = []

            for i, item in enumerate(box_item.wavelength):
                transmission = ReadFilter(box_item.filter_name[i])
                fwhm = transmission.filter_fwhm()

                if quantity == "flux":
                    flux_scaling = item

                if plot_kwargs[j]:
                    if (
                        "label" in plot_kwargs[j]
                        and plot_kwargs[j]["label"] not in label_check
                    ):
                        label_check.append(plot_kwargs[j]["label"])

                    elif (
                        "label" in plot_kwargs[j]
                        and plot_kwargs[j]["label"] in label_check
                    ):
                        del plot_kwargs[j]["label"]

                    if box_item.flux[i][1] is None:
                        if "zorder" not in plot_kwargs[j]:
                            plot_kwargs[j]["zorder"] = 3.0

                        ax1.errorbar(
                            item,
                            flux_scaling * box_item.flux[i][0] / scaling,
                            xerr=fwhm / 2.0,
                            yerr=None,
                            **plot_kwargs[j],
                        )

                    else:
                        if "zorder" not in plot_kwargs[j]:
                            plot_kwargs[j]["zorder"] = 3.0

                        ax1.errorbar(
                            item,
                            flux_scaling * box_item.flux[i][0] / scaling,
                            xerr=fwhm / 2.0,
                            yerr=flux_scaling * box_item.flux[i][1] / scaling,
                            **plot_kwargs[j],
                        )

                else:
                    if box_item.flux[i][1] is None:
                        ax1.errorbar(
                            item,
                            flux_scaling * box_item.flux[i][0] / scaling,
                            xerr=fwhm / 2.0,
                            yerr=None,
                            marker="s",
                            ms=6,
                            color="black",
                            zorder=3,
                        )

                    else:
                        ax1.errorbar(
                            item,
                            flux_scaling * box_item.flux[i][0] / scaling,
                            xerr=fwhm / 2.0,
                            yerr=flux_scaling * box_item.flux[i][1] / scaling,
                            marker="s",
                            ms=6,
                            color="black",
                            zorder=3,
                        )

        elif isinstance(box_item, ObjectBox):
            if box_item.spectrum is not None:
                spec_list = []
                wavel_list = []

                for item in box_item.spectrum:
                    spec_list.append(item)
                    wavel_list.append(box_item.spectrum[item][0][0, 0])

                sort_index = np.argsort(wavel_list)
                spec_sort = []

                for i in range(sort_index.size):
                    spec_sort.append(spec_list[sort_index[i]])

                for spec_key in spec_sort:
                    masked = np.ma.array(
                        box_item.spectrum[spec_key][0],
                        mask=np.isnan(box_item.spectrum[spec_key][0]),
                    )

                    masked = convert_units(masked, units, convert_from=False)

                    if quantity == "flux":
                        flux_scaling = masked[:, 0]

                    if plot_kwargs[j] and spec_key in plot_kwargs[j]:
                        if "label" in plot_kwargs[j][spec_key]:
                            labels_data.append(plot_kwargs[j][spec_key]["label"])

                    if not plot_kwargs[j] or spec_key not in plot_kwargs[j]:
                        plot_obj = ax1.errorbar(
                            masked[:, 0],
                            flux_scaling * masked[:, 1] / scaling,
                            yerr=flux_scaling * masked[:, 2] / scaling,
                            ms=2,
                            marker="s",
                            zorder=2.5,
                            ls="none",
                        )

                        if plot_kwargs[j] is None:
                            plot_kwargs[j] = {}

                        plot_kwargs[j][spec_key] = {
                            "marker": "s",
                            "ms": 2.0,
                            "ls": "none",
                            "color": plot_obj[0].get_color(),
                        }

                    elif (
                        "marker" not in plot_kwargs[j][spec_key]
                        or plot_kwargs[j][spec_key]["marker"] == "none"
                    ):
                        # Plot the spectrum as a line without error bars
                        # (e.g. when the spectrum has a high spectral resolution)
                        plot_obj = ax1.plot(
                            masked[:, 0],
                            flux_scaling * masked[:, 1] / scaling,
                            **plot_kwargs[j][spec_key],
                        )

                    else:
                        if "zorder" not in plot_kwargs[j][spec_key]:
                            plot_kwargs[j][spec_key]["zorder"] = 2.5

                        ax1.errorbar(
                            masked[:, 0],
                            flux_scaling * masked[:, 1] / scaling,
                            yerr=flux_scaling * masked[:, 2] / scaling,
                            xerr=None,
                            **plot_kwargs[j][spec_key],
                        )

            if box_item.flux is not None:
                filter_list = []
                wavel_list = []

                for filter_item in box_item.flux:
                    read_filt = ReadFilter(filter_item)
                    filter_list.append(filter_item)
                    wavel_list.append(read_filt.mean_wavelength())

                sort_index = np.argsort(wavel_list)
                filter_sort = []

                for i in range(sort_index.size):
                    filter_sort.append(filter_list[sort_index[i]])

                for filter_item in filter_sort:
                    transmission = ReadFilter(filter_item)
                    wavel_micron = transmission.mean_wavelength()
                    fwhm_micron = transmission.filter_fwhm()

                    if isinstance(box_item.flux[filter_item][0], np.ndarray):
                        wavel_array = np.full(
                            box_item.flux[filter_item].shape[1], wavel_micron
                        )

                        data_in = np.column_stack(
                            [
                                wavel_array,
                                box_item.flux[filter_item][0],
                                box_item.flux[filter_item][1],
                            ]
                        )

                        data_out = convert_units(data_in, units, convert_from=False)

                        wavelength = data_out[:, 0]
                        flux_conv = data_out[:, 1]
                        sigma_conv = data_out[:, 2]

                    else:
                        data_in = np.column_stack(
                            [
                                [wavel_micron],
                                [box_item.flux[filter_item][0]],
                                [box_item.flux[filter_item][1]],
                            ]
                        )
                        data_out = convert_units(data_in, units, convert_from=False)

                        wavelength = data_out[:, 0]
                        flux_conv = data_out[:, 1]
                        sigma_conv = data_out[:, 2]

                    if fwhm_micron is not None:
                        # Convert FWHM of filter to requested units
                        data_in = np.column_stack(
                            [[wavel_micron + fwhm_micron / 2.0], [1.0]]
                        )
                        data_out = convert_units(data_in, units, convert_from=False)

                        # Absolute value because could be negative when frequency
                        hwhm_up = np.abs(data_out[0, 0] - wavelength[0])

                        # Convert FWHM of filter to requested units
                        data_in = np.column_stack(
                            [[wavel_micron - fwhm_micron / 2.0], [1.0]]
                        )
                        data_out = convert_units(data_in, units, convert_from=False)

                        # Absolute value because could be negative when frequency
                        hwhm_down = np.abs(data_out[0, 0] - wavelength[0])

                        # Calculate the FWHM, which will be identical
                        # to 2*hwhm_up and 2*hwhm_down when working with
                        # wavelengths but hwhm_up and hwhm_down will
                        # be different when converting a FWHM from
                        # wavelength to frequency
                        fwhm = hwhm_up + hwhm_down

                    else:
                        fwhm = None

                    if not plot_kwargs[j] or filter_item not in plot_kwargs[j]:
                        if not plot_kwargs[j]:
                            plot_kwargs[j] = {}

                        if isinstance(box_item.flux[filter_item][0], np.ndarray):
                            if quantity == "flux":
                                flux_scaling = wavelength[0]

                            scale_tmp = flux_scaling / scaling

                            for phot_idx in range(box_item.flux[filter_item].shape[1]):
                                if fwhm is None:
                                    xerr = None
                                else:
                                    xerr = fwhm / 2.0

                                plot_obj = ax1.errorbar(
                                    wavelength[phot_idx],
                                    scale_tmp * box_item.flux[filter_item][0, phot_idx],
                                    xerr=xerr,
                                    yerr=scale_tmp
                                    * box_item.flux[filter_item][1, phot_idx],
                                    marker="s",
                                    ms=5,
                                    zorder=3,
                                    color="black",
                                )

                        else:
                            if quantity == "flux":
                                flux_scaling = wavelength

                            scale_tmp = flux_scaling / scaling

                            if fwhm is None:
                                xerr = None
                            else:
                                xerr = fwhm / 2.0

                            plot_obj = ax1.errorbar(
                                wavelength,
                                scale_tmp * flux_conv,
                                xerr=xerr,
                                yerr=scale_tmp * sigma_conv,
                                marker="s",
                                ms=5,
                                zorder=3,
                                color="black",
                            )

                        plot_kwargs[j][filter_item] = {
                            "marker": "s",
                            "ms": 5.0,
                            "color": plot_obj[0].get_color(),
                        }

                    else:
                        if isinstance(box_item.flux[filter_item][0], np.ndarray):
                            if quantity == "flux":
                                flux_scaling = wavelength[0]

                            if not isinstance(plot_kwargs[j][filter_item], list):
                                raise ValueError(
                                    f"A list with {box_item.flux[filter_item].shape[1]} "
                                    f"dictionaries is required because the filter "
                                    f"{filter_item} has {box_item.flux[filter_item].shape[1]} "
                                    f"values."
                                )

                            for phot_idx in range(box_item.flux[filter_item].shape[1]):
                                if (
                                    "zorder"
                                    not in plot_kwargs[j][filter_item][phot_idx]
                                ):
                                    plot_kwargs[j][filter_item][phot_idx][
                                        "zorder"
                                    ] = 3.0

                                if plot_kwargs[j] and filter_item in plot_kwargs[j]:
                                    if "label" in plot_kwargs[j][filter_item][phot_idx]:
                                        labels_data.append(
                                            plot_kwargs[j][filter_item][phot_idx][
                                                "label"
                                            ]
                                        )

                                if fwhm is None:
                                    xerr = None
                                else:
                                    xerr = fwhm / 2.0

                                ax1.errorbar(
                                    wavelength[phot_idx],
                                    flux_scaling
                                    * box_item.flux[filter_item][0, phot_idx]
                                    / scaling,
                                    xerr=xerr,
                                    yerr=flux_scaling
                                    * box_item.flux[filter_item][1, phot_idx]
                                    / scaling,
                                    **plot_kwargs[j][filter_item][phot_idx],
                                )

                        else:
                            if quantity == "flux":
                                flux_scaling = wavelength

                            if plot_kwargs[j] and filter_item in plot_kwargs[j]:
                                if "label" in plot_kwargs[j][filter_item]:
                                    labels_data.append(
                                        plot_kwargs[j][filter_item]["label"]
                                    )

                            if box_item.flux[filter_item][1] == 0.0:
                                if "zorder" not in plot_kwargs[j][filter_item]:
                                    plot_kwargs[j][filter_item]["zorder"] = 3.0

                                if fwhm is None:
                                    xerr = None
                                else:
                                    xerr = fwhm / 2.0

                                ax1.errorbar(
                                    wavelength,
                                    flux_scaling * flux_conv / scaling,
                                    xerr=xerr,
                                    yerr=None,
                                    uplims=True,
                                    capsize=2.0,
                                    capthick=0.0,
                                    **plot_kwargs[j][filter_item],
                                )

                            else:
                                if "zorder" not in plot_kwargs[j][filter_item]:
                                    plot_kwargs[j][filter_item]["zorder"] = 3.0

                                if fwhm is None:
                                    xerr = None
                                else:
                                    xerr = fwhm / 2.0

                                ax1.errorbar(
                                    wavelength,
                                    flux_scaling * flux_conv / scaling,
                                    xerr=xerr,
                                    yerr=flux_scaling * sigma_conv / scaling,
                                    **plot_kwargs[j][filter_item],
                                )

        elif isinstance(box_item, SynphotBox):
            obj_index = None

            for box_idx, find_item in enumerate(boxes):
                if isinstance(find_item, ObjectBox):
                    obj_index = box_idx
                    break

            for filter_item in box_item.flux:
                transmission = ReadFilter(filter_item)
                wavel_micron = transmission.mean_wavelength()
                fwhm_micron = transmission.filter_fwhm()

                data_in = np.column_stack(
                    [
                        [wavel_micron],
                        [box_item.flux[filter_item]],
                        0.0,
                    ]
                )

                data_out = convert_units(data_in, units, convert_from=False)

                wavelength = data_out[:, 0]
                flux_conv = data_out[:, 1]

                # Convert FWHM of filter to requested units
                data_in = np.column_stack([[wavel_micron + fwhm_micron / 2.0], [1.0]])
                data_out = convert_units(data_in, units, convert_from=False)

                # Absolute value because could be negative when frequency
                hwhm_up = np.abs(data_out[0, 0] - wavelength[0])

                # Convert FWHM of filter to requested units
                data_in = np.column_stack([[wavel_micron - fwhm_micron / 2.0], [1.0]])
                data_out = convert_units(data_in, units, convert_from=False)

                # Absolute value because could be negative when frequency
                hwhm_down = np.abs(data_out[0, 0] - wavelength[0])

                # Calculate the FWHM, which will be identical
                # to 2*hwhm_up and 2*hwhm_down when working with
                # wavelengths but hwhm_up and hwhm_down will
                # be different when converting a FWHM from
                # wavelength to frequency
                fwhm = hwhm_up + hwhm_down

                if quantity == "flux":
                    flux_scaling = wavelength

                if plot_kwargs[j] is not None and filter_item in plot_kwargs[j]:
                    kwargs_copy = plot_kwargs[j][filter_item].copy()

                    if "zorder" not in kwargs_copy:
                        kwargs_copy["zorder"] = 2.8

                    ax1.errorbar(
                        wavelength,
                        flux_scaling * flux_conv / scaling,
                        xerr=fwhm / 2.0,
                        yerr=None,
                        **kwargs_copy,
                    )

                elif (
                    obj_index is None
                    or not plot_kwargs[obj_index]
                    or filter_item not in plot_kwargs[obj_index]
                ):
                    ax1.errorbar(
                        wavelength,
                        flux_scaling * flux_conv / scaling,
                        xerr=fwhm / 2.0,
                        yerr=None,
                        alpha=0.7,
                        marker="s",
                        ms=5,
                        zorder=4,
                        mfc="white",
                    )

                else:
                    if isinstance(plot_kwargs[obj_index][filter_item], list):
                        # In case of multiple photometry values for the
                        # same filter, use the plot_kwargs of the first
                        # data point

                        kwargs_copy = plot_kwargs[obj_index][filter_item][0].copy()

                        if "label" in kwargs_copy:
                            del kwargs_copy["label"]

                        if "zorder" in kwargs_copy:
                            zorder_synphot = kwargs_copy["zorder"] - 0.2
                            del kwargs_copy["zorder"]
                        else:
                            zorder_synphot = 2.8

                        ax1.errorbar(
                            wavelength,
                            flux_scaling * flux_conv / scaling,
                            xerr=fwhm / 2.0,
                            yerr=None,
                            mfc="white",
                            zorder=zorder_synphot,
                            **kwargs_copy,
                        )

                    else:
                        kwargs_copy = plot_kwargs[obj_index][filter_item].copy()

                        if "label" in kwargs_copy:
                            del kwargs_copy["label"]

                        if "mfc" in kwargs_copy:
                            del kwargs_copy["mfc"]

                        if "zorder" in kwargs_copy:
                            zorder_synphot = kwargs_copy["zorder"] - 0.2
                            del kwargs_copy["zorder"]
                        else:
                            zorder_synphot = 2.8

                        ax1.errorbar(
                            wavelength,
                            flux_scaling * flux_conv / scaling,
                            xerr=fwhm / 2.0,
                            yerr=None,
                            mfc="white",
                            zorder=zorder_synphot,
                            **kwargs_copy,
                        )

    if filters is not None:
        for filter_item in filters:
            transmission = ReadFilter(filter_item)
            data = transmission.get_filter()

            data_in = np.ones(data.shape)
            data_in[:, 0] = data[:, 0]

            data_out = convert_units(data_in, units, convert_from=False)
            data[:, 0] = data_out[:, 0]

            ax2.plot(data[:, 0], data[:, 1], "-", lw=0.7, color="tab:gray", zorder=1)

    if residuals is not None:
        obj_index = None

        for i, find_item in enumerate(boxes):
            if isinstance(find_item, ObjectBox):
                obj_index = i
                break

        if obj_index is None:
            raise ValueError(
                "ObjectBox not found so cannot create "
                "residuals. Please add an ObjectBox to "
                "the list of boxes."
            )

        res_max = 0.0

        if residuals.photometry is not None:
            for item in residuals.photometry:
                data_in = np.array(
                    [[residuals.photometry[item][0], residuals.photometry[item][1]]]
                )
                data_out = convert_units(data_in, units, convert_from=False)
                residuals.photometry[item][0] = data_out[0, 0]

                if not plot_kwargs[obj_index] or item not in plot_kwargs[obj_index]:
                    ax3.plot(
                        residuals.photometry[item][0],
                        residuals.photometry[item][1],
                        marker="s",
                        ms=5,
                        linestyle="none",
                        zorder=2,
                    )

                else:
                    if residuals.photometry[item].ndim == 1:
                        if "zorder" not in plot_kwargs[obj_index][item]:
                            plot_kwargs[obj_index][item]["zorder"] = 2.0

                        ax3.errorbar(
                            residuals.photometry[item][0],
                            residuals.photometry[item][1],
                            **plot_kwargs[obj_index][item],
                        )

                    elif residuals.photometry[item].ndim == 2:
                        for i in range(residuals.photometry[item].shape[1]):
                            if isinstance(plot_kwargs[obj_index][item], list):
                                if "zorder" not in plot_kwargs[obj_index][item][i]:
                                    plot_kwargs[obj_index][item][i]["zorder"] = 2.0

                                ax3.errorbar(
                                    residuals.photometry[item][0, i],
                                    residuals.photometry[item][1, i],
                                    **plot_kwargs[obj_index][item][i],
                                )

                            else:
                                if "zorder" not in plot_kwargs[obj_index][item]:
                                    plot_kwargs[obj_index][item]["zorder"] = 2.0

                                ax3.errorbar(
                                    residuals.photometry[item][0, i],
                                    residuals.photometry[item][1, i],
                                    **plot_kwargs[obj_index][item],
                                )

                finite = np.isfinite(residuals.photometry[item][1])

                max_tmp = np.max(np.abs(residuals.photometry[item][1][finite]))
                res_max = max(res_max, max_tmp)

        if residuals.spectrum is not None:
            for spec_key, spec_val in residuals.spectrum.items():
                data_out = convert_units(spec_val, units, convert_from=False)
                spec_val[:, 0] = data_out[:, 0]

                if not plot_kwargs[obj_index] or spec_key not in plot_kwargs[obj_index]:
                    ax3.errorbar(
                        spec_val[:, 0],
                        spec_val[:, 1],
                        marker="o",
                        ms=2,
                        ls="none",
                        zorder=1,
                    )

                else:
                    if "zorder" not in plot_kwargs[obj_index][spec_key]:
                        plot_kwargs[obj_index][spec_key]["zorder"] = 1.0

                    ax3.errorbar(
                        spec_val[:, 0],
                        spec_val[:, 1],
                        **plot_kwargs[obj_index][spec_key],
                    )

                max_tmp = np.nanmax(np.abs(spec_val[:, 1]))
                res_max = max(res_max, max_tmp)

        res_lim = math.ceil(1.1 * res_max)

        if res_lim == 0.0:
            res_lim = 5.0

        if res_lim > 10.0:
            res_lim = 5.0

        ax3.axhline(0.0, ls="--", lw=0.7, color="gray", dashes=(2, 4), zorder=0.5)

        sigma_line = [5.0, 10.0, 15.0, 20.0]

        for sigma_item in sigma_line:
            if res_lim > sigma_item or (
                ylim_res is not None
                and ylim_res[0] < -sigma_item
                and ylim_res[1] > sigma_item
            ):
                ax3.axhline(
                    -sigma_item, ls=":", lw=0.7, color="gray", dashes=(1, 4), zorder=0.5
                )
                ax3.axhline(
                    sigma_item, ls=":", lw=0.7, color="gray", dashes=(1, 4), zorder=0.5
                )

        if ylim_res is None:
            ax3.set_ylim(-res_lim, res_lim)

        else:
            ax3.set_ylim(ylim_res[0], ylim_res[1])

    if filters is not None:
        ax2.set_ylim(0.0, 1.1)

    if title is not None:
        if filters:
            ax2.set_title(title, y=1.02, fontsize=font_size["title"])
        else:
            ax1.set_title(title, y=1.02, fontsize=font_size["title"])

    handles, labels = ax1.get_legend_handles_labels()

    if handles and legend is not None:
        if isinstance(legend, list):
            model_handles = []
            data_handles = []
            model_labels = []
            data_labels = []

            for handle_idx, handle_item in enumerate(handles):
                if labels[handle_idx] in labels_data:
                    data_handles.append(handle_item)
                    data_labels.append(labels[handle_idx])
                else:
                    model_handles.append(handle_item)
                    model_labels.append(labels[handle_idx])

            if legend[0] is not None:
                if isinstance(legend[0], (str, tuple)):
                    leg_1 = ax1.legend(
                        model_handles,
                        model_labels,
                        loc=legend[0],
                        fontsize=font_size["legend"],
                        frameon=False,
                    )
                else:
                    if "fontsize" not in legend[0]:
                        legend[0]["fontsize"] = font_size["legend"]

                    leg_1 = ax1.legend(model_handles, model_labels, **legend[0])

            else:
                leg_1 = None

            if legend[1] is not None:
                if isinstance(legend[1], (str, tuple)):
                    ax1.legend(
                        data_handles,
                        data_labels,
                        loc=legend[1],
                        fontsize=font_size["legend"],
                        frameon=False,
                    )
                else:
                    if "fontsize" not in legend[1]:
                        legend[1]["fontsize"] = font_size["legend"]

                    ax1.legend(data_handles, data_labels, **legend[1])

            if leg_1 is not None:
                ax1.add_artist(leg_1)

        elif isinstance(legend, (str, tuple)):
            ax1.legend(loc=legend, fontsize=font_size["legend"], frameon=False)

        else:
            if "fontsize" not in legend:
                legend["fontsize"] = font_size["legend"]

            ax1.legend(**legend)

    # if scale[0] == "log":
    #     ax1.xaxis.set_major_formatter(ScalarFormatter())
    #
    #     if ax2 is not None:
    #         ax2.xaxis.set_major_formatter(ScalarFormatter())
    #
    #     if ax3 is not None:
    #         ax3.xaxis.set_major_formatter(ScalarFormatter())

    if units[0] in ["Hz", "GHz"] and xlim is None:
        ax1.invert_xaxis()

        if filters is not None:
            ax2.invert_xaxis()

        if residuals is not None:
            ax3.invert_xaxis()

    if xlim is None:
        xlim = ax1.get_xlim()
    else:
        ax1.set_xlim(xlim[0], xlim[1])

    if filters is not None:
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylim(0.0, 1.0)

    if residuals is not None:
        ax3.set_xlim(xlim[0], xlim[1])

    # if scale[1] == "log":
    #     ax1.yaxis.set_major_locator()

    # filters = ['Paranal/SPHERE.ZIMPOL_N_Ha',
    #            'MUSE/Hbeta',
    #            'ALMA/855']
    #
    # filters = ['Paranal/SPHERE.IRDIS_B_Y',
    #            'MKO/NSFCam.J',
    #            'Paranal/SPHERE.IRDIS_D_H23_2',
    #            'Paranal/SPHERE.IRDIS_D_H23_3',
    #            'Paranal/SPHERE.IRDIS_D_K12_1',
    #            'Paranal/SPHERE.IRDIS_D_K12_2',
    #            'Paranal/NACO.Lp',
    #            'Paranal/NACO.NB405',
    #            'Paranal/NACO.Mp']
    #
    # for i, item in enumerate(filters):
    #     readfilter = ReadFilter(item)
    #     filter_wavelength = readfilter.mean_wavelength()
    #     filter_width = readfilter.filter_fwhm()
    #
    #     # if i == 5:
    #     #     ax1.errorbar(filter_wavelength, 1.3e4, xerr=filter_width/2., color='dimgray', elinewidth=2.5, zorder=10)
    #     # else:
    #     #     ax1.errorbar(filter_wavelength, 6e3, xerr=filter_width/2., color='dimgray', elinewidth=2.5, zorder=10)
    #
    #     if i == 0:
    #         ax1.text(filter_wavelength, 1e-2, r'H$\alpha$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 1:
    #         ax1.text(filter_wavelength, 1e-2, r'H$\beta$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 2:
    #         ax1.text(filter_wavelength, 1e-2, 'ALMA\nband 7 rms', ha='center', va='center', fontsize=8, color='black')
    #
    #     if i == 0:
    #         ax1.text(filter_wavelength, 1.4, 'Y', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 1:
    #         ax1.text(filter_wavelength, 1.4, 'J', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 2:
    #         ax1.text(filter_wavelength-0.04, 1.4, 'H2', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 3:
    #         ax1.text(filter_wavelength+0.04, 1.4, 'H3', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 4:
    #         ax1.text(filter_wavelength, 1.4, 'K1', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 5:
    #         ax1.text(filter_wavelength, 1.4, 'K2', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 6:
    #         ax1.text(filter_wavelength, 1.4, 'L$\'$', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 7:
    #         ax1.text(filter_wavelength, 1.4, 'NB4.05', ha='center', va='center', fontsize=10, color='black')
    #     elif i == 8:
    #         ax1.text(filter_wavelength, 1.4, 'M$\'}$', ha='center', va='center', fontsize=10, color='black')
    #
    # ax1.text(1.26, 0.58, 'VLT/SPHERE', ha='center', va='center', fontsize=8., color='slateblue', rotation=43.)
    # ax1.text(2.5, 1.28, 'VLT/SINFONI', ha='left', va='center', fontsize=8., color='darkgray')

    if output is None:
        plt.show()
    else:
        print(f"\nOutput: {output}")
        plt.savefig(output, bbox_inches="tight")

    return fig
