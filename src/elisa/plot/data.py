"""Data classes for plotting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache, wraps
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as stats

from elisa.infer.likelihood import (
    _STATISTIC_BACK_NORMAL,
    _STATISTIC_SPEC_NORMAL,
    _STATISTIC_WITH_BACK,
)
from elisa.plot.residuals import (
    pearson_residuals,
    pit_poisson,
    pit_poisson_normal,
    pit_poisson_poisson,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Callable, Literal

    from elisa.infer.results import (
        BootstrapResult,
        FitResult,
        MLEResult,
        PosteriorResult,
        PPCResult,
    )
    from elisa.util.typing import Array, NumPyArray


def _cache_method(bound_method: Callable) -> Callable:
    """Cache instance method."""
    return cache(bound_method)


def _cache_method_with_check(
    instance: Any, bound_method: Callable, check_fields: Sequence[str]
) -> Callable:
    """Cache instance method with computation dependency check."""

    def get_id():
        return {field: id(getattr(instance, field)) for field in check_fields}

    cached_method = cache(bound_method)
    old_id = get_id()

    @wraps(bound_method)
    def _(*args, **kwargs):
        if (new_id := get_id()) != old_id:
            cached_method.cache_clear()
            old_id.update(new_id)
        return cached_method(*args, **kwargs)

    return _


def _get_cached_method_decorator(storage: list):
    def decorator(method: Callable):
        storage.append(method.__name__)
        return method

    return decorator


def _get_cached_method_with_check_decorator(
    storage: list, check_fields: str | Sequence[str]
):
    if isinstance(check_fields, str):
        check_fields = [check_fields]
    else:
        check_fields = list(check_fields)

    def decorator(method: Callable):
        name = method.__name__
        storage.append((name, check_fields))
        return method

    return decorator


class PlotData(ABC):
    _cached_method: list[str]
    _cached_method_with_check: list[tuple[str, list[str]]]

    def __init__(self, name: str, result: FitResult, seed: int):
        self.name = str(name)
        self.result = result
        self.seed = seed
        self.data = result._helper.data[self.name]
        self.statistic = result._helper.statistic[self.name]

        for f in self._cached_method:
            method = getattr(self, f)
            setattr(self, f, _cache_method(method))

        for f, fields in self._cached_method_with_check:
            method = getattr(self, f)
            setattr(self, f, _cache_method_with_check(self, method, fields))

    @property
    def channel(self) -> NumPyArray:
        return self.data.channel

    @property
    def ch_emin(self) -> NumPyArray:
        return self.data.ch_emin

    @property
    def ch_emax(self) -> NumPyArray:
        return self.data.ch_emax

    @property
    def ch_emid(self) -> NumPyArray:
        return self.data.ch_emid

    @property
    def ch_width(self) -> NumPyArray:
        return self.data.ch_width

    @property
    def ch_mean(self) -> NumPyArray:
        return self.data.ch_mean

    @property
    def ch_error(self) -> NumPyArray:
        return self.data.ch_error

    @property
    def ce_data(self) -> Array:
        return self.data.ce

    @property
    def ce_error(self) -> Array:
        return self.data.ce_error

    @property
    def spec_counts(self) -> Array:
        return self.data.spec_counts

    @property
    def spec_error(self) -> Array:
        return self.data.spec_error

    @property
    def back_ratio(self) -> float | Array:
        return self.data.back_ratio

    @property
    def back_counts(self) -> Array | None:
        return self.data.back_counts

    @property
    def back_error(self) -> Array | None:
        return self.data.back_error

    @property
    def net_counts(self) -> Array:
        return self.data.net_counts

    @property
    def net_error(self) -> Array:
        return self.data.net_error

    @property
    def ndata(self) -> int:
        return len(self.data.channel)

    @property
    @abstractmethod
    def ce_model(self) -> Array:
        """Point estimate of the folded source model."""
        pass

    @abstractmethod
    def ce_model_ci(self, cl: float = 0.683) -> Array | None:
        """Confidence/Credible intervals of the folded source model."""
        pass

    @abstractmethod
    def pit(self) -> tuple:
        """Probability integral transform."""
        pass

    @abstractmethod
    def residuals(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None,
        random_quantile: bool,
        mle: bool,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        """Residuals between the data and the fitted models."""
        pass

    @abstractmethod
    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None,
        random_quantile: bool,
    ) -> Array | None:
        """Residuals bootstrap/ppc samples."""
        pass

    @abstractmethod
    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        cl: float,
        seed: int | None,
        random_quantile: bool,
        with_sign: bool,
    ) -> Array | None:
        """Confidence/Credible intervals of the residuals."""
        pass


_cached_method = []
_cached_method_with_check = []
_to_cached_method = _get_cached_method_decorator(_cached_method)
_to_cached_method_with_check = _get_cached_method_with_check_decorator(
    _cached_method_with_check, 'boot'
)


class MLEPlotData(PlotData):
    result: MLEResult
    _cached_method = _cached_method
    _cached_method_with_check = _cached_method_with_check

    @property
    def boot(self) -> BootstrapResult:
        return self.result._boot

    def get_model_mle(self, name: str) -> Array:
        return self.result._model_values[name]

    def get_model_boot(self, name: str) -> Array | None:
        boot = self.boot
        if boot is None:
            return None
        else:
            return boot.models[name]

    def get_data_boot(self, name: str) -> Array | None:
        boot = self.boot
        if boot is None:
            return None
        else:
            return boot.data[name]

    @property
    def ce_model(self) -> Array:
        return self.get_model_mle(self.name)

    @_to_cached_method_with_check
    def ce_model_ci(self, cl: float = 0.683) -> Array | None:
        if self.boot is None:
            return None

        assert 0.0 < cl < 1.0
        ci = np.quantile(
            self.get_model_boot(self.name),
            q=0.5 + cl * np.array([-0.5, 0.5]),
            axis=0,
        )
        return ci

    @property
    def sign(self) -> dict[str, Array | None]:
        """Sign of the difference between the data and the fitted models."""
        return {'mle': self._sign_mle(), 'boot': self._sign_boot()}

    @_to_cached_method
    def _sign_mle(self) -> Array:
        return np.where(self.ce_data >= self.ce_model, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_boot(self) -> Array | None:
        boot = self.get_model_boot(self.name)
        if boot is not None:
            boot = np.where(self.get_data_boot(self.name) >= boot, 1.0, -1.0)
        return boot

    @property
    def on_models(self) -> dict[str, Array | None]:
        """Point estimate and bootstrap sample of the on measurement model."""
        on_name = f'{self.name}_Non_model'
        return {
            'mle': self.get_model_mle(on_name),
            'boot': self.get_model_boot(on_name),
        }

    @property
    def off_models(self) -> dict[str, Array | None]:
        """Point estimate and bootstrap sample of the off measurement model."""
        if self.statistic not in _STATISTIC_WITH_BACK:
            return {'mle': None, 'boot': None}

        off_name = f'{self.name}_Noff_model'
        return {
            'mle': self.get_model_mle(off_name),
            'boot': self.get_model_boot(off_name),
        }

    @property
    def deviance(self) -> dict[str, Array | None]:
        """MLE and bootstrap deviance."""
        mle = self.result._deviance['point'][self.name]
        if self.boot is not None:
            boot = self.boot.deviance['point'][self.name]
        else:
            boot = None
        return {'mle': mle, 'boot': boot}

    @property
    def _nsim(self) -> int:
        return 10000

    @_to_cached_method
    def pit(self) -> tuple[Array, Array]:
        stat = self.statistic

        if stat in _STATISTIC_SPEC_NORMAL:
            on_data = self.net_counts
        else:
            on_data = self.spec_counts
        on_model = self.on_models['mle']

        if stat in _STATISTIC_SPEC_NORMAL:  # chi2
            pit = stats.norm.cdf((on_data - on_model) / self.net_error)
            return pit, pit

        if stat in _STATISTIC_WITH_BACK:
            off_data = self.back_counts
            off_model = self.off_models['mle']

            if stat in _STATISTIC_BACK_NORMAL:  # pgstat
                pit = pit_poisson_normal(
                    k=on_data,
                    lam=on_model,
                    v=off_data,
                    mu=off_model,
                    sigma=self.back_error,
                    ratio=self.back_ratio,
                    seed=self.seed + 1,
                    nsim=self._nsim,
                )
                return pit, pit

            else:  # wstat
                return pit_poisson_poisson(
                    k1=on_data,
                    k2=off_data,
                    lam1=on_model,
                    lam2=off_model,
                    ratio=self.data.back_ratio,
                    minus=True,
                    seed=self.seed + 1,
                    nsim=self._nsim,
                )

        else:  # cstat, or pstat
            return pit_poisson(k=on_data, lam=on_model, minus=True)

    def residuals(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = True,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        if rtype == 'deviance':
            return self.deviance_residuals_mle()
        elif rtype == 'pearson':
            return self.pearson_residuals_mle()
        elif rtype == 'quantile':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals_mle(seed, random_quantile)
        else:
            raise NotImplementedError(f'{rtype} residual')

    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.boot is None or rtype == 'quantile':
            return None

        if rtype == 'deviance':
            r = self.deviance_residuals_boot()
        elif rtype == 'pearson':
            r = self.pearson_residuals_boot()
        else:
            raise NotImplementedError(f'{rtype} residual')

        return r

    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.boot is None or rtype == 'quantile':
            return None

        assert 0 < cl < 1

        r = self.residuals_sim(rtype, seed, random_quantile)

        if with_sign:
            return np.quantile(r, q=0.5 + cl * np.array([-0.5, 0.5]), axis=0)
        else:
            q = np.quantile(np.abs(r), q=cl, axis=0)
            return np.row_stack([-q, q])

    @_to_cached_method
    def deviance_residuals_mle(self) -> Array:
        return self._deviance_residuals('mle')

    @_to_cached_method_with_check
    def deviance_residuals_boot(self) -> Array | None:
        return self._deviance_residuals('boot')

    def _deviance_residuals(
        self, rtype: Literal['mle', 'boot']
    ) -> Array | None:
        if rtype == 'boot' and self.boot is None:
            return None

        # NB: if background is present, then this assumes the background is
        #     being profiled out, so that each src & bkg data pair has ~1 dof
        return self.sign[rtype] * np.sqrt(self.deviance[rtype])

    @_to_cached_method
    def pearson_residuals_mle(self) -> Array:
        return self._pearson_residuals('mle')

    @_to_cached_method_with_check
    def pearson_residuals_boot(self) -> Array | None:
        return self._pearson_residuals('boot')

    def _pearson_residuals(
        self, rtype: Literal['mle', 'boot']
    ) -> Array | None:
        if rtype == 'boot' and self.boot is None:
            return None

        stat = self.statistic

        if rtype == 'mle':
            if stat in _STATISTIC_SPEC_NORMAL:
                on_data = self.net_counts
            else:
                on_data = self.spec_counts
        else:
            on_data = self.get_data_boot(f'{self.name}_Non')

        if stat in _STATISTIC_SPEC_NORMAL:
            std = self.net_error
        else:
            std = None

        r = pearson_residuals(on_data, self.on_models[rtype], std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype == 'mle':
                off_data = self.back_counts
            else:
                off_data = self.get_data_boot(f'{self.name}_Noff')

            if self.statistic in _STATISTIC_BACK_NORMAL:
                std = self.back_error
            else:
                std = None

            r_b = pearson_residuals(off_data, self.off_models[rtype], std)

            # NB: this assumes the background is being profiled out,
            #     so that each src & bkg data pair has ~1 dof
            r = self.sign[rtype] * np.sqrt(r * r + r_b * r_b)

        return r

    def quantile_residuals_mle(
        self, seed: int, random: bool
    ) -> tuple[Array, Array | bool, Array | bool]:
        pit_minus, pit = self.pit()

        if random:
            pit = np.random.default_rng(seed).uniform(pit_minus, pit)
        r = stats.norm.ppf(pit)

        lower = upper = False

        if self.statistic in {'pgstat', 'wstat'}:
            upper_mask = pit == 0.0
            if np.any(upper_mask):
                r[upper_mask] = stats.norm.ppf(1.0 / self._nsim)
                upper = np.full(r.shape, False)
                upper[upper_mask] = True

            lower_mask = pit == 1.0
            if np.any(lower_mask):
                r[lower_mask] = stats.norm.ppf(1.0 - 1.0 / self._nsim)
                lower = np.full(r.shape, False)
                lower[lower_mask] = True

        return r, lower, upper


# clean up helpers
del (
    _cached_method,
    _cached_method_with_check,
    _to_cached_method,
    _to_cached_method_with_check,
)

_cached_method = []
_cached_method_with_check = []
_to_cached_method = _get_cached_method_decorator(_cached_method)
_to_cached_method_with_check = _get_cached_method_with_check_decorator(
    _cached_method_with_check, 'ppc'
)


class PosteriorPlotData(PlotData):
    result: PosteriorResult
    _cached_method = _cached_method
    _cached_method_with_check = _cached_method_with_check

    @property
    def ppc(self) -> PPCResult | None:
        return self.result._ppc

    @_to_cached_method
    def get_model_median(self, name: str) -> Array:
        posterior = self.result._idata['posterior'][name]
        return posterior.median(dim=('chain', 'draw')).values

    @_to_cached_method
    def get_model_posterior(self, name: str) -> Array:
        posterior = self.result._idata['posterior'][name].values
        return np.concatenate(posterior)

    def get_model_ppc(self, name: str) -> Array | None:
        if self.ppc is None:
            return None
        else:
            return self.ppc.models_fit[name]

    def get_model_mle(self, name: str) -> Array | None:
        mle = self.result._mle
        if mle is None:
            return None
        else:
            return mle['models'][name]

    @property
    def ce_model(self) -> Array:
        return self.get_model_median(self.name)

    @_to_cached_method
    def ce_model_ci(self, cl: float = 0.683) -> Array:
        assert 0.0 < cl < 1.0
        return np.quantile(
            self.get_model_posterior(self.name),
            q=0.5 + cl * np.array([-0.5, 0.5]),
            axis=0,
        )

    @property
    def sign(self) -> dict[str, Array | None]:
        """Sign of the difference between the data and the fitted models."""
        return {
            'posterior': self._sign_posterior(),
            'median': self._sign_median(),
            'mle': self._sign_mle(),
            'ppc': self._sign_ppc(),
        }

    @_to_cached_method
    def _sign_posterior(self) -> Array:
        ce_posterior = self.get_model_posterior(self.name)
        return np.where(self.ce_data >= ce_posterior, 1.0, -1.0)

    @_to_cached_method
    def _sign_median(self) -> Array:
        ce_median = self.get_model_median(self.name)
        return np.where(self.ce_data >= ce_median, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_mle(self) -> Array | None:
        if self.ppc is None:
            return None

        ce_mle = self.get_model_mle(self.name)
        return np.where(self.ce_data >= ce_mle, 1.0, -1.0)

    @_to_cached_method_with_check
    def _sign_ppc(self) -> Array | None:
        if self.ppc is None:
            return None

        ce_ppc = self.get_model_ppc(self.name)
        return np.where(self.ppc.data[self.name] >= ce_ppc, 1.0, -1.0)

    @property
    def on_models(self) -> dict[str, Array | None]:
        on_name = f'{self.name}_Non_model'
        return {
            'posterior': self.get_model_posterior(on_name),
            'median': self.get_model_median(on_name),
            'mle': self.get_model_mle(on_name),
            'ppc': self.get_model_ppc(on_name),
        }

    @property
    def off_models(self) -> dict[str, Array | None]:
        if self.statistic not in _STATISTIC_WITH_BACK:
            return {
                'posterior': None,
                'median': None,
                'mle': None,
                'ppc': None,
            }

        off_name = f'{self.name}_Noff_model'
        return {
            'posterior': self.get_model_posterior(off_name),
            'median': self.get_model_median(off_name),
            'mle': self.get_model_mle(off_name),
            'ppc': self.get_model_ppc(off_name),
        }

    @property
    def deviance(self) -> dict[str, Array | None]:
        """Median, MLE, and ppc deviance."""
        loglike = self.result._idata['log_likelihood'][self.name].values
        posterior = -2.0 * np.concatenate(loglike)

        mle = self.result._mle
        if mle is not None:
            mle = mle['deviance']['point'][self.name]

        ppc = self.result._ppc
        if ppc is not None:
            ppc = ppc.deviance['point'][self.name]

        return {'posterior': posterior, 'mle': mle, 'ppc': ppc}

    def pit(self) -> tuple:
        return self.result._loo_pit[self.name]

    def residuals(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
        mle: bool = False,
    ) -> Array | tuple[Array, bool | Array, bool | Array]:
        assert rtype in {'deviance', 'pearson', 'quantile'}

        if rtype == 'quantile':
            seed = self.seed if seed is None else int(seed)
            return self.quantile_residuals(seed, random_quantile)
        else:
            point_type = 'mle' if mle else 'median'
            return getattr(self, f'{rtype}_residuals_{point_type}')()

    def residuals_sim(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        seed: int | None = None,
        random_quantile: bool = True,
    ) -> Array | None:
        if self.ppc is None or rtype == 'quantile':
            return None

        if rtype == 'deviance':
            r = self.deviance_residuals_ppc()
        elif rtype == 'pearson':
            r = self.pearson_residuals_ppc()
        else:
            raise NotImplementedError(f'{rtype} residual')
        return r

    def residuals_ci(
        self,
        rtype: Literal['deviance', 'pearson', 'quantile'],
        cl: float = 0.683,
        seed: int | None = None,
        random_quantile: bool = True,
        with_sign: bool = False,
    ) -> Array | None:
        if self.ppc is None or rtype == 'quantile':
            return None

        assert 0 < cl < 1

        r = self.residuals_sim(rtype, seed, random_quantile)

        if with_sign:
            return np.quantile(r, q=0.5 + cl * np.array([-0.5, 0.5]), axis=0)
        else:
            q = np.quantile(np.abs(r), q=cl, axis=0)
            return np.row_stack([-q, q])

    @_to_cached_method
    def deviance_residuals_median(self) -> Array:
        return np.median(self._deviance_residuals('posterior'), axis=0)

    @_to_cached_method_with_check
    def deviance_residuals_mle(self) -> Array:
        return self._deviance_residuals('mle')

    @_to_cached_method_with_check
    def deviance_residuals_ppc(self) -> Array | None:
        if self.ppc is None:
            return None
        return self._deviance_residuals('ppc')

    def _deviance_residuals(
        self, rtype: Literal['posterior', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        # NB: if background is present, then this assumes the background is
        #     being profiled out, so that each src & bkg data pair has ~1 dof
        return self.sign[rtype] * np.sqrt(self.deviance[rtype])

    @_to_cached_method
    def pearson_residuals_median(self) -> Array:
        return np.median(self._pearson_residuals('posterior'), axis=0)

    @_to_cached_method_with_check
    def pearson_residuals_mle(self) -> Array:
        return self._pearson_residuals('mle')

    @_to_cached_method_with_check
    def pearson_residuals_ppc(self) -> Array | None:
        if self.ppc is None:
            return None
        return self._pearson_residuals('ppc')

    def _pearson_residuals(
        self, rtype: Literal['posterior', 'mle', 'ppc']
    ) -> Array | None:
        if rtype in ['mle', 'ppc'] and self.ppc is None:
            return None

        stat = self.statistic

        if rtype in {'posterior', 'mle'}:
            if stat in _STATISTIC_SPEC_NORMAL:
                on_data = self.net_counts
            else:
                on_data = self.spec_counts
        else:
            on_data = self.ppc.data[f'{self.name}_Non']
        on_model = self.on_models[rtype]

        if stat in _STATISTIC_SPEC_NORMAL:
            std = self.net_error
        else:
            std = None

        r = pearson_residuals(on_data, on_model, std)

        if stat in _STATISTIC_WITH_BACK:
            if rtype in {'posterior', 'mle'}:
                off_data = self.back_counts
            else:
                off_data = self.ppc.data[f'{self.name}_Noff']
            off_model = self.off_models[rtype]

            if self.statistic in _STATISTIC_BACK_NORMAL:
                std = self.back_error
            else:
                std = None

            r_b = pearson_residuals(off_data, off_model, std)

            # NB: this assumes the background is being profiled out,
            #     so that each src & bkg data pair has ~1 dof
            r = self.sign[rtype] * np.sqrt(r * r + r_b * r_b)

        return r

    def quantile_residuals(
        self, seed: int, random: bool
    ) -> tuple[Array, Array | bool, Array | bool]:
        pit_minus, pit = self.pit()
        if random:
            pit = np.random.default_rng(seed).uniform(pit_minus, pit)
        r = stats.norm.ppf(pit)

        # Assume the posterior prediction is nchan * ndraw times
        nchain = len(self.result._idata['posterior']['chain'])
        ndraw = len(self.result._idata['posterior']['draw'])
        nsim = nchain * ndraw

        lower = upper = False

        upper_mask = pit == 0.0
        if np.any(upper_mask):
            r[upper_mask] = stats.norm.ppf(1.0 / nsim)
            upper = np.full(r.shape, False)
            upper[upper_mask] = True

        lower_mask = pit == 1.0
        if np.any(lower_mask):
            r[lower_mask] = stats.norm.ppf(1.0 - 1.0 / nsim)
            lower = np.full(r.shape, False)
            lower[lower_mask] = True

        return r, lower, upper


# clean up helpers
del (
    _cached_method,
    _cached_method_with_check,
    _to_cached_method,
    _to_cached_method_with_check,
)