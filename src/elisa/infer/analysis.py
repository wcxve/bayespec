"""Subsequent analysis of likelihood or Bayesian fit."""
from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Sequence

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from iminuit import Minuit
from iminuit.util import FMin
from scipy.stats import norm

from . import fit as _fit  # avoid circular import
from .util import make_pretty_table


class ConfidenceInterval(NamedTuple):
    """Confidence interval result."""

    mle: dict[str, float]
    interval: dict[str, tuple[float, float]]
    error: dict[str, tuple[float, float]]
    cl: float
    method: str
    status: dict


class CredibleInterval(NamedTuple):
    """Credible interval result."""
    mle: dict[str, float]
    median: dict[str, float]
    interval: dict[str, tuple[float, float]]
    prob: float
    hdi: bool
    sample_method: Literal['nuts', 'ns']


class BootstrapResult(NamedTuple):
    """Parametric bootstrap result."""

    mle: dict[str, float]
    simulation: dict[str, jax.Array]
    params: dict[str, jax.Array]
    deviance: dict
    p_value: dict
    valid: jax.Array
    n: int
    n_valid: int
    seed: int


class PPCResult(NamedTuple):
    """Posterior predictive check result."""
    ...


class MLEResult:
    """MLE result obtained from likelihood fit."""

    def __init__(
        self,
        minuit: Minuit,
        fit: _fit.LikelihoodFit
    ):
        self._minuit = minuit
        self._optfn = optfn = fit._helper
        self._free_names = free_names = fit._free_names
        self._params_names = params = fit._params_names
        self._interest_params = fit._interest_names
        self._composite_params = fit._composite
        self._ndata = ndata = fit._ndata
        self._dof = fit._dof
        self._stat_type = fit._stat
        self._seed = fit._seed
        self._boot: BootstrapResult | None = None

        mle_unconstr = np.array(minuit.values)
        mle = np.array(optfn.to_params_array(mle_unconstr))

        cov_unconstr = np.array(optfn.unconstr_covar(mle_unconstr))
        if np.isnan(cov_unconstr).any() and minuit.covariance is not None:
            cov_unconstr = np.array(minuit.covariance)
        cov = optfn.params_covar(mle_unconstr, cov_unconstr)
        err = np.sqrt(np.diagonal(cov))

        stat_info = optfn.deviance_unconstr_info(mle_unconstr)
        stat_group = stat_info['group']
        stat = {k: float(stat_group[k]) for k in ndata if k != 'total'}
        stat_total = minuit.fval
        stat |= {'total': stat_total}

        unconstr_dict = {k: v for k, v in zip(free_names, mle_unconstr)}
        param_dict = optfn.to_params_dict(unconstr_dict)
        constr_dict = {k: float(param_dict[k]) for k in free_names}
        self._result = {
            'unconstr': unconstr_dict,
            'constr': constr_dict,
            'params': param_dict,
            'deviance': {
                'total': stat_total,
                'group': stat_group,
                'point': stat_info['point']
            }
        }

        k = len(free_names)
        n = ndata['total']
        self._mle = {p: (v, e) for p, v, e in zip(params, mle, err)}
        self._stat = stat
        self._aic = stat_total + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        self._bic = stat_total + k * np.log(n)
        self._status = minuit.fmin

    def __repr__(self):
        tab = make_pretty_table(
            ['Parameter', 'Value', 'Error'],
            [(k, f'{v[0]:.4g}', f'{v[1]:.4g}') for k, v in self._mle.items()]
        )
        s = 'MLE:\n' + tab.get_string() + '\n'

        stat_type = self._stat_type
        stat_value = self._stat
        ndata = self._ndata
        stat = [
            f'{i}: {stat_type[i]}={stat_value[i]:.2f}, ndata={ndata[i]}'
            for i in self._ndata.keys()
            if i != 'total'
        ]
        total_stat = stat_value['total']
        dof = self._dof
        stat += [
            f'Total: stat/dof={total_stat/dof:.2f} ({total_stat:.2f}/{dof})'
        ]
        s += '\nStatistic:\n' + '\n'.join(stat) + '\n'
        s += f'AIC: {self.aic:.2f}\n'
        s += f'BIC: {self.bic:.2f}\n'

        s += f'\nFit Status:\n{self.status}'

        return s

    def _repr_html_(self) -> str:
        # TODO
        return self.__repr__()

    @property
    def mle(self) -> dict[str, tuple[float, float]]:
        """MLE and error of parameters."""
        return self._mle

    @property
    def statistic(self) -> dict[str, float]:
        """Fit statistic."""
        return self._stat

    @property
    def ndata(self) -> dict[str, int]:
        """Number of data points."""
        return self._ndata

    @property
    def dof(self) -> int:
        """Degree of freedom."""
        return self._dof

    @property
    def aic(self) -> float:
        """Akaike information criterion with sample size correction."""
        return self._aic

    @property
    def bic(self) -> float:
        """Bayesian information criterion."""
        return self._bic

    @property
    def status(self) -> FMin:
        """Fit status of Minuit."""
        return self._status

    def ci(
        self,
        params: Optional[str | Sequence[str]] = None,
        cl: float | int = 1,
        method: Literal['profile', 'boot'] = 'profile',
        n: Optional[int] = None,
    ) -> ConfidenceInterval:
        """Calculate confidence intervals for given parameters.

        If method is 'profile', Minos algorithm of Minuit is used here to find
        the profile likelihood based CIs, as it is considered as a robust error
        estimator.

        If method is 'boot', parametric bootstrap method is used to calculate
        the bootstrap CIs.

        Parameters
        ----------
        params : str or list of str, optional
            Parameters to calculate confidence intervals. If not provided,
            confidence intervals are calculated for all parameters.
        cl : float or int, optional
            Confidence level for the confidence interval. If 0 < `cl` < 1, the
            value is interpreted as the confidence level. If `cl` >= 1, it is
            interpreted as number of standard deviations. For example, ``cl=1``
            produces a 1-sigma or 68.3% confidence interval. The default is 1.
        method : {'profile', 'boot'}, optional
            Method used to calculate confidence. Either profile likelihood or
            parametric bootstrap method. The default is profile likelihood
            method.
        n : int, optional
            Number of bootstrap to calculate confidence intervals. Takes effect
            only if `method` is 'boot'. If None, set to the default value as in
            :meth:`LikelihoodFit.boot`.

        Returns
        -------
        ConfidenceInterval
            The confidence interval given the parameters and confidence level.

        """
        if not self._minuit.valid:
            msg = 'fit must be valid to calculate confidence interval'
            raise RuntimeError(msg)

        if params is None:
            params = self._interest_params

        elif isinstance(params, str):
            # check if params exist
            if params not in self._params_names:
                raise ValueError(f'parameter: {params} is not exist')

            params = [params]

        elif isinstance(params, Sequence):
            # check if params exist
            params = [str(i) for i in params]
            flag = [i in self._params_names for i in params]
            if not all(flag):
                params_err = ', '.join(
                    [i for i, j in zip(params, flag) if not j]
                )
                raise ValueError(f'parameters: {params_err} are not exist')

            params = list(str(i) for i in params)

        else:
            raise ValueError('params must be str, or sequence of str')

        free_params = [i for i in params if i in self._free_names]
        composite_params = [i for i in params if i in self._composite_params]

        cl_ = 1.0 - 2.0 * norm.sf(cl) if cl >= 1.0 else cl

        mle = {
            k: v for k, v in self._result['params'].items()
            if k in params
        }

        optfn = self._optfn

        if method == 'profile':
            self._minuit.minos(*free_params, cl=cl)

            mle0 = self._minuit.values.to_dict()

            others = {  # set other unconstrained free parameter to mle
                i: mle0[i]
                for i in (set(mle0.keys()) - set(free_params))
            }

            ci = self._minuit.merrors
            lo = optfn.to_params_dict(
                {k: mle0[k] + ci[k].lower for k in free_params} | others
            )
            up = optfn.to_params_dict(
                {k: mle0[k] + ci[k].upper for k in free_params} | others
            )

            interval = {k: (lo[k], up[k]) for k in free_params}
            error = {k: (lo[k] - mle[k], up[k] - mle[k]) for k in free_params}
            status = {
                k: {
                    'valid': (v.lower_valid, v.upper_valid),
                    'at_limit': (v.at_lower_limit, v.at_upper_limit),
                    'at_max_fcn': (v.at_lower_max_fcn, v.at_upper_max_fcn),
                    'new_min': (v.lower_new_min, v.upper_new_min),
                }
                for k, v in ci.items()
            }

            # confidence interval of function of parameters,
            # see, e.g. https://doi.org/10.1007/s11222-021-10012-y
            for p in composite_params:
                def loss(x):
                    """The loss when calculating CI of composite parameter."""
                    unconstr = {k: v for k, v in zip(self._free_names, x[1:])}
                    p0 = optfn.to_params_dict(unconstr)[p]
                    diff = (p0 / x[0] - 1) / 1e-3
                    return optfn.deviance_unconstr(x[1:]) + diff*diff

                mle_p = mle[p]

                m = Minuit(
                    jax.jit(loss),
                    [mle_p, *self._minuit.values],
                    grad=jax.jit(jax.grad(loss)),
                )
                m.strategy = 2
                m.migrad()
                m.minos(0, cl=cl)
                ci = m.merrors['x0']
                interval[p] = (mle_p + ci.lower, mle_p + ci.upper)
                error[p] = (ci.lower, ci.upper)
                status[p] = {
                    'valid': (ci.lower_valid, ci.upper_valid),
                    'at_limit': (ci.at_lower_limit, ci.at_upper_limit),
                    'at_max_fcn': (ci.at_lower_max_fcn, ci.at_upper_max_fcn),
                    'new_min': (ci.lower_new_min, ci.upper_new_min),
                }

        elif method == 'boot':
            if n is None:
                boot_result = self.boot()
            else:
                boot_result = self.boot(n=n)
            interval = jax.tree_map(
                lambda x: tuple(np.quantile(x, q=(0.5 - cl_/2, 0.5 + cl_/2))),
                {k: v for k, v in self._boot.params.items() if k in params}
            )
            error = {
                k: (interval[k][0] - mle[k], interval[k][1] - mle[k])
                for k in params
            }
            status = {'n': boot_result.n, 'n_valid': boot_result.n_valid}

        else:
            raise ValueError('method must be either "profile" or "boot"')

        def format_result(v):
            """Order the result dict and use float as result type."""
            formatted = jax.tree_map(float, v)
            return {k: formatted[k] for k in params}

        return ConfidenceInterval(
            mle=format_result(mle),
            interval=format_result(interval),
            error=format_result(error),
            cl=cl_,
            method=method,
            status=status
        )

    def boot(self, n: int = 10000) -> BootstrapResult:
        """Parametric bootstrap.

        Parameters
        ----------
        n : int, optional
            The number of bootstrap.

        Returns
        -------
        BootstrapResult
            The boostrap result.

        """
        if not self._minuit.valid:
            msg = 'fit must be valid to perform bootstrap'
            raise RuntimeError(msg)

        # self._simulator = simulator
        # self._batch_fit = batch_fit_simulation

        # directly return previous result if the configuration is the same
        if self._boot and self._boot.n == n and self._boot.seed == self._seed:
            return self._boot

        simulation = self._simulator(self._mle['constr'], n)

        result_container = {
            'params': {k: jnp.empty(n) for k in self._params_names},
            'stat': jnp.empty(n),
            'stat_group': {k: jnp.empty(n) for k in self._ndata.keys()},
            'stat_point': {
                k: jnp.empty((n, v)) for k, v in self._ndata.items()
                if k != 'total'
            },
            'grad': jnp.empty(n),
            'valid': jnp.full(n, True, bool)
        }
        mle = jnp.array(list(self._result['unconstr'].values()))

        result = self._batch_fit(simulation, result_container, mle)

        boot_result = BootstrapResult(
            mle=self._result['unconstr'],
            simulation=simulation,
            params=result['params'],
            deviance={
                'total': result['stat'],
                'group': result['stat_group'],
                'point': result['stat_point']
            },
            p_value=...,
            valid=result['valid'],
            n=n,
            n_valid=result['valid'].sum(),
            seed=self._seed
        )

        self._boot = boot_result

        return boot_result


class PosteriorResult:
    """Posterior sampling result obtained from Bayesian fit."""

    def __init__(self, idata: az.InferenceData):
        self._idata = idata

    def ci(
        self,
        prob: float | int = 1,
        hdi: bool = False,
        idata: Optional[Literal['nuts', 'ns']] = None
    ) -> CredibleInterval:
        """Calculate credible intervals for given parameters.

        Parameters
        ----------
        prob : float or int, optional
            The probability mass of samples within the credible interval.
            If 0 < `cl` < 1, the value is interpreted as the probability mass.
            If `cl` >= 1, it is interpreted as number of standard deviations.
            For example, ``cl=1`` produces a 1-sigma or 68.3% credible
            interval. The default is 1.
        hdi : bool, optional
            Whether to return the highest density interval. The default is
            False, which means an equal tailed interval is returned.
        idata : {'nuts', 'ns'}, optional
            Which samples to calculate the credible interval. The default is
            using the latest samples.

        Returns
        -------
        CredibleInterval
            The credible interval given the parameters and probability mass.

        """
        idata = self._idata

    def ppc(self, n: int = 10000) -> PPCResult:
        """

        Parameters
        ----------
        n :

        Returns
        -------

        """
        ...
        # pred_func = numpyro.infer.util.Predictive(
        #     model=self._numpyro_model,
        #     posterior_samples=...,
        #     parallel=True
        # )
        # pred = pred_func(self._PRNGKey)

        # LOO PIT need posterior predictive net counts
