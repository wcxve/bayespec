"""Visualize fit and analysis results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from elisa.infer.helper import check_params
from elisa.plot.data import MLEPlotData, PosteriorPlotData
from elisa.plot.misc import plot_corner, plot_trace
from elisa.plot.scale import LinLogScale, get_scale
from elisa.plot.util import get_colors, get_markers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from matplotlib.pyplot import Axes, Figure

    from elisa.infer.results import FitResult, MLEResult, PosteriorResult
    from elisa.plot.data import PlotData
    from elisa.util.typing import Array, NumPyArray


class PlotConfig:
    """Plotting configuration."""

    def __init__(
        self,
        alpha: float = 0.8,
        palette: Any = 'colorblind',
        xscale: Literal['linear', 'log'] = 'log',
        yscale: Literal['linear', 'log', 'linlog'] = 'linlog',
        lin_frac: float = 0.15,
        cl: tuple[float, ...] = (0.683, 0.95),
        residuals: Literal['deviance', 'pearson', 'quantile'] = 'quantile',
        random_quantile: bool = False,
        mark_outlier_residuals: bool = False,
        residuals_ci_with_sign: bool = True,
        plot_comps: bool = False,
    ):
        self.alpha = alpha
        self.palette = palette
        self.xscale = xscale
        self.yscale = yscale
        self.lin_frac = lin_frac
        self.cl = cl
        self.residuals = residuals
        self.random_quantile = random_quantile
        self.mark_outlier_residuals = mark_outlier_residuals
        self.residuals_ci_with_sign = residuals_ci_with_sign
        self.plot_comps = plot_comps

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        alpha = float(alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        self._alpha = alpha

    @property
    def palette(self) -> Any:
        return self._palette

    @palette.setter
    def palette(self, palette: Any):
        self._palette = palette

    @property
    def xscale(self) -> Literal['linear', 'log']:
        return self._xscale

    @xscale.setter
    def xscale(self, xscale: Literal['linear', 'log']):
        if xscale not in {'linear', 'log'}:
            raise ValueError('xscale must be "linear" or "log"')
        self._xscale = xscale

    @property
    def yscale(self) -> Literal['linear', 'log', 'linlog']:
        return self._yscale

    @yscale.setter
    def yscale(self, yscale: Literal['linear', 'log', 'linlog']):
        if yscale not in {'linear', 'log', 'linlog'}:
            raise ValueError('yscale must be "linear", "log", or "linlog"')
        self._yscale = yscale

    @property
    def lin_frac(self) -> float:
        return self._lin_frac

    @lin_frac.setter
    def lin_frac(self, lin_frac: float):
        lin_frac = float(lin_frac)
        if not 0.0 < lin_frac <= 0.5:
            raise ValueError('lin_frac must be in (0, 0.5]')
        self._lin_frac = lin_frac

    @property
    def cl(self) -> NumPyArray:
        return self._cl

    @cl.setter
    def cl(self, cl: float | Sequence[float]):
        cl = np.sort(np.atleast_1d(cl)).astype(float)
        for c in cl:
            if not 0.0 < c < 1.0:
                raise ValueError('cl must be in (0, 1)')
        self._cl = cl

    @property
    def residuals(self) -> Literal['deviance', 'pearson', 'quantile']:
        return self._residuals

    @residuals.setter
    def residuals(self, residuals: Literal['deviance', 'pearson', 'quantile']):
        if residuals not in {'deviance', 'pearson', 'quantile'}:
            raise ValueError(
                'residuals must be "deviance", "pearson", or "quantile"'
            )
        self._residuals = residuals

    @property
    def random_quantile(self) -> bool:
        return self._random_quantile

    @random_quantile.setter
    def random_quantile(self, random_quantile: bool):
        self._random_quantile = bool(random_quantile)

    @property
    def mark_outlier_residuals(self) -> bool:
        return self._mark_outlier_residuals

    @mark_outlier_residuals.setter
    def mark_outlier_residuals(self, mark_outlier_residuals: bool):
        self._mark_outlier_residuals = bool(mark_outlier_residuals)

    @property
    def residuals_ci_with_sign(self) -> bool:
        return self._residuals_ci_with_sign

    @residuals_ci_with_sign.setter
    def residuals_ci_with_sign(self, residuals_ci_with_sign: bool):
        self._residuals_ci_with_sign = bool(residuals_ci_with_sign)

    @property
    def plot_comps(self) -> bool:
        return self._plot_comps

    @plot_comps.setter
    def plot_comps(self, plot_comps: bool):
        self._plot_comps = bool(plot_comps)


class Plotter(ABC):
    """Plotter to visualize analysis results."""

    _palette: Any | None = None
    data: dict[str, PlotData] | None = None

    def __init__(self, result: FitResult, config: PlotConfig = None):
        self._result = result
        self.data = self.get_plot_data(result)
        self.config = config
        markers = get_markers(len(self.data))
        self._markers = dict(zip(self.data.keys(), markers))

    # def __call__(
    #     self,
    #     plots=
    #     '(data ne ene eene fv vfv) (pit) (qq) (pvalue) (trace) (corner)',
    #     residuals=True/False/deviance pearson quantile,

    # ):
    #     ...
    #
    # def plot_corner(self):
    #     # correlation map, bootstrap distribution, posterior distribution
    #     ...

    @staticmethod
    @abstractmethod
    def get_plot_data(result: FitResult) -> dict[str, PlotData]:
        """Get PlotData from FitResult."""
        pass

    @property
    def config(self) -> PlotConfig:
        """Plotting configuration."""
        return self._config

    @config.setter
    def config(self, config: PlotConfig):
        if config is None:
            config = PlotConfig()
        elif not isinstance(config, PlotConfig):
            raise TypeError('config must be a PlotConfig instance')

        self._config = config

    @property
    def colors(self):
        if self._palette != self.config.palette:
            colors = get_colors(len(self.data), palette=self.config.palette)
            self._colors = dict(zip(self.data.keys(), colors))
        return self._colors

    @property
    def ndata(self):
        ndata = {name: data.ndata for name, data in self.data.items()}
        ndata['total'] = sum(ndata.values())
        return ndata

    def plot(self, *args, r=None, **kwargs) -> tuple[Figure, np.ndarray[Axes]]:
        config = self.config
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex='all',
            height_ratios=[1.618, 1.0],
            gridspec_kw={'hspace': 0.03},
            figsize=(8, 6),
        )

        fig.align_ylabels(axs)

        for ax in axs:
            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

        plt.rcParams['axes.formatter.min_exponent'] = 3

        axs[-1].set_xlabel(r'$\mathrm{Energy\ [keV]}$')

        ylabels = {
            'ce': r'$C_E\ \mathrm{[s^{-1}\ keV^{-1}]}$',
            'residuals': r'$r_D\ [\mathrm{\sigma}]$',
            'ne': r'$N_E\ \mathrm{[s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'ene': r'$E N_E\ \mathrm{[erg\ s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'eene': r'$E^2 N_E\ \mathrm{[erg\ s^{-1}\ cm^{-2}]}$',
            'Fv': r'$F_{\nu}\ \mathrm{[erg\ s^{-1}\ cm^{-2}\ keV^{-1}]}$',
            'vFv': r'$\nu F_{\nu}\ \mathrm{[erg\ s^{-1}\ cm^{-2}]}$',
        }
        axs[0].set_ylabel(ylabels['ce'])
        axs[1].set_ylabel(ylabels['residuals'])

        self.plot_ce_model(axs[0])
        self.plot_ce_data(axs[0])
        self.plot_residuals(axs[1], r)

        axs[0].set_xscale(config.xscale)
        ax = axs[0]
        xmin, xmax = ax.dataLim.intervalx
        ax.set_xlim(xmin * 0.97, xmax * 1.06)

        yscale = config.yscale
        assert yscale in {'linear', 'log', 'linlog'}
        if yscale in {'linear', 'log'}:
            ax.set_yscale(yscale)
        else:
            ax.set_yscale('log')
            lin_thresh = ax.get_ylim()[0]
            lin_frac = config.lin_frac
            dmin, dmax = ax.get_yaxis().get_data_interval()
            scale = LinLogScale(
                axis=None,
                base=10.0,
                lin_thresh=lin_thresh,
                lin_scale=get_scale(10.0, lin_thresh, dmin, dmax, lin_frac),
            )
            ax.set_yscale(scale)
            ax.axhline(lin_thresh, c='k', lw=0.15, ls=':', zorder=-1)

        axs[0].legend()

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        return fig, axs

    def plot_ce_model(self, ax: Axes):
        config = self.config
        colors = self.colors
        cl = config.cl
        step_kwargs = {'lw': 1.618, 'alpha': config.alpha}
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.2 * config.alpha}

        for name, data in self.data.items():
            color = colors[name]

            _plot_step(
                ax,
                data.ch_emin,
                data.ch_emax,
                data.ce_model,
                color=color,
                **step_kwargs,
            )

            quantiles = []
            for i_cl in cl:
                if (q := data.ce_model_ci(i_cl)) is not None:
                    quantiles.append(q)

            if quantiles:
                _plot_ribbon(
                    ax,
                    data.ch_emin,
                    data.ch_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )

    def plot_ce_data(self, ax: Axes):
        config = self.config
        colors = self.colors
        alpha = config.alpha
        xlog = config.xscale == 'log'

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            ax.errorbar(
                x=data.ch_mean if xlog else data.ch_emid,
                xerr=data.ch_error if xlog else 0.5 * data.ch_width,
                y=data.ce_data,
                yerr=data.ce_error,
                alpha=alpha,
                color=color,
                fmt=f'{marker} ',
                label=name,
                lw=0.75,
                ms=2.4,
                mec=color,
                mfc='#FFFFFFCC',
            )

    def plot_residuals(
        self,
        ax: Axes,
        rtype: Literal['deviance', 'pearson', 'quantile'] | None = None,
        seed: int | None = None,
    ):
        config = self.config
        colors = self.colors
        cl = config.cl
        random_quantile = config.random_quantile
        with_sign = config.residuals_ci_with_sign
        mark_outlier = config.mark_outlier_residuals
        ribbon_kwargs = {'lw': 0.618, 'alpha': 0.15 * config.alpha}

        if rtype is None:
            rtype = config.residuals

        alpha = config.alpha
        xlog = config.xscale == 'log'

        normal_q = stats.norm.isf(0.5 * (1.0 - cl))

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            x = data.ch_mean if xlog else data.ch_emid
            xerr = data.ch_error if xlog else 0.5 * data.ch_width

            quantiles = []
            for i_cl in cl:
                q = data.residuals_ci(
                    rtype, i_cl, seed, random_quantile, with_sign
                )
                if q is not None:
                    quantiles.append(q)

            if quantiles:
                _plot_ribbon(
                    ax,
                    data.ch_emin,
                    data.ch_emax,
                    quantiles,
                    color=color,
                    **ribbon_kwargs,
                )
            else:
                for q in normal_q:
                    ax.fill_between(
                        [data.ch_emin[0], data.ch_emax[-1]],
                        -q,
                        q,
                        color=color,
                        **ribbon_kwargs,
                    )

            use_mle = True if quantiles else False
            r = data.residuals(rtype, seed, config.random_quantile, use_mle)
            if rtype == 'quantile':
                r, lower, upper = r
            else:
                lower = upper = False
            ax.errorbar(
                x=x,
                y=r,
                yerr=1.0,
                xerr=xerr,
                color=color,
                alpha=alpha,
                linewidth=0.75,
                linestyle='',
                marker=marker,
                markersize=2.4,
                markeredgecolor=color,
                markerfacecolor='#FFFFFFCC',
                lolims=lower,
                uplims=upper,
            )

            if mark_outlier:
                if quantiles:
                    q = quantiles[-1]
                else:
                    q = [-normal_q[-1], normal_q[-1]]
                mask = (r < q[0]) | (r > q[1])
                ax.scatter(x[mask], r[mask], marker='x', c='r')

        for q in normal_q:
            ax.axhline(q, ls=':', lw=1, c='gray', zorder=0)
            ax.axhline(-q, ls=':', lw=1, c='gray', zorder=0)

        ax.axhline(0, ls='--', lw=1, c='gray', zorder=0)
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    def plot_qq(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'] | None = None,
        seed: int | None = None,
        detrend: bool = True,
    ):
        """Quantile-Quantile plot."""
        config = self.config
        random_quantile = config.random_quantile
        if rtype is None:
            rtype = config.residuals

        rsim = {
            name: data.residuals_sim(rtype, seed, random_quantile)
            for name, data in self.data.items()
        }
        if any(i is None for i in rsim.values()):
            rsim['total'] = None
        else:
            rsim['total'] = np.hstack(list(rsim.values()))

        use_mle = True if rsim else False
        r = {
            name: data.residuals(rtype, seed, random_quantile, use_mle)
            for name, data in self.data.items()
        }
        if rtype == 'quantile':
            r = {k: v[0] for k, v in r.items()}
        r['total'] = np.hstack(list(r.values()))

        n_subplots = len(self.data)
        if n_subplots == 1:
            ncols = 1
        else:
            ncols = n_subplots // 2
            if n_subplots % 2:
                ncols += 1

        fig = plt.figure(figsize=(4 + ncols * 2.25, 4), tight_layout=True)
        gs1 = fig.add_gridspec(1, 2, width_ratios=[4, ncols * 2.25])
        gs2 = gs1[0, 1].subgridspec(2, ncols, wspace=0.35)
        ax1 = fig.add_subplot(gs1[0, 0])
        axs = gs2.subplots(squeeze=False)
        ax1.set_xlabel('Normal Theoretical Quantiles')
        ax1.set_ylabel('Residuals')

        alpha = config.alpha
        ha = 'center' if detrend else 'left'
        text_x = 0.5 if detrend else 0.03

        axs = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)
        for ax, name, color in zip(axs, names, colors):
            theor, q, line, lo, up = get_qq(r[name], detrend, 0.95, rsim[name])
            ax.scatter(theor, q, s=5, color=color, alpha=alpha)
            ax.plot(theor, line, ls='--', color=color, alpha=alpha)
            ax.plot(theor, lo, ls=':', color=color, alpha=alpha)
            ax.plot(theor, up, ls=':', color=color, alpha=alpha)
            ax.fill_between(
                theor, lo, up, alpha=0.2 * alpha, color=color, lw=0.0
            )
            ax.annotate(
                name,
                xy=(text_x, 0.97),
                xycoords='axes fraction',
                ha=ha,
                va='top',
                color=color,
            )
        if n_subplots % 2:
            axs[-1].set_visible(False)

    def plot_pit(self, detrend=True):
        """Probability integral transformation empirical CDF plot."""
        config = self.config

        pit = {name: data.pit()[1] for name, data in self.data.items()}
        pit['total'] = np.hstack(list(pit.values()))

        n_subplots = len(self.data)
        if n_subplots == 1:
            ncols = 1
        else:
            ncols = n_subplots // 2
            if n_subplots % 2:
                ncols += 1

        fig = plt.figure(figsize=(4 + ncols * 2.25, 4), tight_layout=True)
        gs1 = fig.add_gridspec(1, 2, width_ratios=[4, ncols * 2.25])
        gs2 = gs1[0, 1].subgridspec(2, ncols, wspace=0.35)
        ax1 = fig.add_subplot(gs1[0, 0])
        axs = gs2.subplots(squeeze=False)
        ax1.set_xlabel('Scaled Rank')
        ax1.set_ylabel('PIT ECDF')

        alpha = config.alpha
        ha = 'right' if detrend else 'left'
        text_x = 0.97 if detrend else 0.03

        ax_list = [ax1] + axs.ravel().tolist()
        names = ['total'] + list(self.ndata.keys())
        colors = ['k'] + get_colors(n_subplots, config.palette)

        for ax, name, color in zip(ax_list, names, colors):
            x, y, line, lower, upper = get_pit_ecdf(pit[name], 0.95, detrend)
            ax.plot(x, line, ls='--', color=color, alpha=alpha)
            ax.fill_between(
                x, lower, upper, alpha=0.2 * alpha, color=color, step='mid'
            )
            ax.step(x, y, alpha=alpha, color=color, where='mid')
            ax.annotate(
                text=name,
                xy=(text_x, 0.97),
                xycoords='axes fraction',
                ha=ha,
                va='top',
                color=color,
            )
        if n_subplots % 2:
            ax_list[-1].set_visible(False)

    # def plot_ne(self, ax: Axes):
    #     pass
    #
    # def plot_ene(self, ax: Axes):
    #     pass
    #
    # def plot_eene(self, ax: Axes):
    #     pass
    #
    # def plot_ufspec(self):
    #     pass
    #
    # def plot_eufspec(self):
    #     pass
    #
    # def plot_eeufspec(self):
    #     pass


class MLEResultPlotter(Plotter):
    data: dict[str, MLEPlotData]
    _result: MLEResult

    @staticmethod
    def get_plot_data(result: MLEResult) -> dict[str, MLEPlotData]:
        helper = result._helper
        keys = jax.random.split(
            jax.random.PRNGKey(helper.seed['resd']), len(helper.data_names)
        )
        data = {
            name: MLEPlotData(name, result, int(key[0]))
            for name, key in zip(helper.data_names, keys)
        }
        return data

    # def plot_corner(self):
    #     # profile and contour
    #     ...


class PosteriorResultPlotter(Plotter):
    data: dict[str, PosteriorPlotData]
    _result: PosteriorResult

    @staticmethod
    def get_plot_data(result: PosteriorResult) -> dict[str, PosteriorPlotData]:
        helper = result._helper
        keys = jax.random.split(
            jax.random.PRNGKey(helper.seed['resd']), len(helper.data_names)
        )
        data = {
            name: PosteriorPlotData(name, result, int(key[0]))
            for name, key in zip(helper.data_names, keys)
        }
        return data

    def plot_trace(
        self,
        params: str | Sequence[str] | None = None,
        fig_path: str | None = None,
    ):
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        labels = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            + (f'\n[{u}]' if (u := helper.params_unit[p]) else '')
            for p in params
        ]
        fig = plot_trace(self._result._idata, params, axes_scale, labels)
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')

    def plot_corner(
        self,
        params: str | Sequence[str] | None = None,
        color: str | None = None,
        divergences: bool = True,
        fig_path: str | None = None,
    ):
        helper = self._result._helper
        params = check_params(params, helper)
        axes_scale = [
            'log' if helper.params_log[p] else 'linear' for p in params
        ]
        titles = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            for p in params
        ]
        labels = [
            f'${helper.params_comp_latex[p]}$  ${helper.params_latex[p]}$'
            + (f'\n[{u}]' if (u := helper.params_unit[p]) else '')
            for p in params
        ]
        fig = plot_corner(
            idata=self._result._idata,
            params=params,
            axes_scale=axes_scale,
            levels=self.config.cl,
            titles=titles,
            labels=labels,
            color=color,
            divergences=divergences,
        )
        if fig_path:
            fig.savefig(fig_path, bbox_inches='tight')

    def plot_khat(self):
        config = self.config
        colors = self.colors
        alpha = config.alpha
        xlog = config.xscale == 'log'

        fig, ax = plt.subplots(1, 1, squeeze=True)

        khat = self._result.loo.pareto_k
        if np.any(khat.values > 0.7):
            ax.axhline(0.7, color='r', lw=0.5, ls=':')

        for name, data in self.data.items():
            color = colors[name]
            marker = self._markers[name]
            khat_data = khat.sel(channel=data.channel).values
            x = data.ch_mean if xlog else data.ch_emid
            ax.errorbar(
                x=x,
                xerr=data.ch_error if xlog else 0.5 * data.ch_width,
                y=khat_data,
                alpha=alpha,
                color=color,
                fmt=f'{marker} ',
                label=name,
                lw=0.75,
                ms=2.4,
                mec=color,
                mfc='#FFFFFFCC',
            )

            mask = khat_data > 0.7
            if np.any(mask):
                ax.scatter(x=x[mask], y=khat_data[mask], marker='x', c='r')

        ax.set_xscale('log')
        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel(r'Shape Parameter $\hat{k}$')


def _plot_step(
    ax: Axes, x_left: Array, x_right: Array, y: Array, **step_kwargs
) -> None:
    assert len(y) == len(x_left) == len(x_right)

    step_kwargs['where'] = 'post'

    mask = x_left[1:] != x_right[:-1]
    idx = np.insert(np.flatnonzero(mask) + 1, 0, 0)
    idx = np.append(idx, len(y))
    for i in range(len(idx) - 1):
        i_slice = slice(idx[i], idx[i + 1])
        x_slice = np.append(x_left[i_slice], x_right[i_slice][-1])
        y_slice = y[i_slice]
        y_slice = np.append(y_slice, y_slice[-1])
        ax.step(x_slice, y_slice, **step_kwargs)


def _plot_ribbon(
    ax,
    x_left: Array,
    x_right: Array,
    y_ribbons: Sequence[Array],
    **ribbon_kwargs,
) -> None:
    y_ribbons = list(map(np.asarray, y_ribbons))
    shape = y_ribbons[0].shape
    assert len(shape) == 2 and shape[0] == 2
    assert shape[1] == len(x_left) == len(x_right)
    assert all(ribbon.shape == shape for ribbon in y_ribbons)

    ribbon_kwargs['step'] = 'post'

    mask = x_left[1:] != x_right[:-1]
    idx = np.insert(np.flatnonzero(mask) + 1, 0, 0)
    idx = np.append(idx, shape[1])
    for i in range(len(idx) - 1):
        i_slice = slice(idx[i], idx[i + 1])
        x_slice = np.append(x_left[i_slice], x_right[i_slice][-1])

        for ribbon in y_ribbons:
            lower = ribbon[0]
            lower_slice = lower[i_slice]
            lower_slice = np.append(lower_slice, lower_slice[-1])
            upper = ribbon[1]
            upper_slice = upper[i_slice]
            upper_slice = np.append(upper_slice, upper_slice[-1])
            ax.fill_between(x_slice, lower_slice, upper_slice, **ribbon_kwargs)


def get_qq(
    q: NumPyArray,
    detrend: bool,
    cl: float,
    qsim: NumPyArray | None = None,
) -> tuple[NumPyArray, ...]:
    """Get the Q-Q and pointwise confidence/credible interval.

    References
    ----------
    .. [1] doi:10.1080/00031305.2013.847865
    """
    # https://stats.stackexchange.com/a/9007
    # https://stats.stackexchange.com/a/152834
    alpha = np.pi / 8  # 3/8 is also ok
    n = len(q)
    theor = stats.norm.ppf((np.arange(1, n + 1) - alpha) / (n - 2 * alpha + 1))

    q = np.sort(q)
    if qsim is not None:
        line, lower, upper = np.quantile(
            np.sort(qsim, axis=1),
            q=[0.5, 0.5 - 0.5 * cl, 0.5 + 0.5 * cl],
            axis=0,
        )
    else:
        line = np.array(theor)
        grid = np.arange(1, n + 1)
        lower = stats.beta.ppf(0.5 - cl * 0.5, grid, n + 1 - grid)
        upper = stats.beta.ppf(0.5 + cl * 0.5, grid, n + 1 - grid)
        lower = stats.norm.ppf(lower)
        upper = stats.norm.ppf(upper)

    if detrend:
        q -= theor
        line -= theor
        lower -= theor
        upper -= theor

    return theor, q, line, lower, upper


def get_pit_ecdf(
    pit: NumPyArray,
    cl: float,
    detrend: bool,
) -> tuple[NumPyArray, ...]:
    """Get the empirical CDF of PIT and pointwise confidence/credible interval.

    References
    ----------
    .. [1] doi:10.1007/s11222-022-10090-6
    """
    n = len(pit)

    # See ref [1] for the following
    scaled_rank = np.linspace(0.0, 1.0, n + 1)
    # Since binomial is discrete, we need to have lower and upper bounds with
    # a confidence/credible level >= cl to ensure the nominal coverage,
    # that is, we require that (cdf <= 0.5 - 0.5 * cl) for lower bound
    # and (0.5 + 0.5 * cl <= cdf) for upper bound
    lower_q = 0.5 - cl * 0.5
    lower = stats.binom.ppf(lower_q, n, scaled_rank)
    mask = stats.binom.cdf(lower, n, scaled_rank) > lower_q
    lower[mask] -= 1.0
    lower = np.clip(lower / n, 0.0, 1.0)

    upper_q = 0.5 + cl * 0.5
    upper = stats.binom.ppf(upper_q, n, scaled_rank)
    mask = stats.binom.cdf(upper, n, scaled_rank) < upper_q
    upper[mask] += 1.0
    upper = np.clip(upper / n, 0.0, 1.0)

    line = scaled_rank
    pit_ecdf = np.count_nonzero(pit <= scaled_rank[:, None], axis=1) / n

    if detrend:
        lower -= line
        upper -= line
        pit_ecdf -= line
        line = np.zeros_like(line)

    return scaled_rank, pit_ecdf, line, lower, upper

    # x = np.hstack([0.0, np.sort(pit), 1.0])
    # pit_ecdf = np.hstack([0.0, np.arange(n) / n, 1.0])
    # line = scaled_rank
    #
    # if detrend:
    #     pit_ecdf -= x
    #     lower -= scaled_rank
    #     upper -= scaled_rank
    #     line = np.zeros_like(scaled_rank)
    #
    # return x, pit_ecdf, scaled_rank, line, lower, upper


# def get_pit_pdf(pit_intervals: NumPyArray) -> NumPyArray:
#     """Get the pdf of PIT.
#
#     References
#     ----------
#     .. [1] doi:10.1111/j.1541-0420.2009.01191.x
#     """
#     assert len(pit_intervals.shape) == 2 and pit_intervals.shape[1] == 2
#
#     grid = np.unique(pit_intervals)
#     if grid[0] > 0.0:
#         grid = np.insert(grid, 0, 0)
#     if grid[-1] < 1.0:
#         grid = np.append(grid, 1.0)
#
#     n = len(pit_intervals)
#     mask = pit_intervals[:, 0] != pit_intervals[:, 1]
#     cover_mask = np.bitwise_and(
#         pit_intervals[:, :1] <= grid[:-1],
#         grid[1:] <= pit_intervals[:, 1:],
#     )
#     pdf = np.zeros((n, len(grid) - 1))
#     pdf[cover_mask] = np.repeat(
#         1.0 / (pit_intervals[mask, 1] - pit_intervals[mask, 0]),
#         np.count_nonzero(cover_mask[mask], axis=1),
#     )
#     idx = np.clip(grid.searchsorted(pit_intervals[~mask, 0]) - 1, 0, None)
#     pdf[~mask, idx] = 1.0 / (grid[idx + 1] - grid[idx])
#     return pdf.mean(0)
