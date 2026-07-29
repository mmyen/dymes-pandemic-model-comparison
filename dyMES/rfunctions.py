import numpy as np
from typing import Callable

def R_mean(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list, mean_func = lambda n: 1) -> float:
    n_array = np.arange(0,m_eff + 1)
    f_array = mean_func(n_array)
    return np.sum(R(transition_function, N, m_eff, params, lambdas) * f_array)

def Rn_mean(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list, mean_func = lambda n: 1) -> float:
    n_array = np.arange(0,m_eff + 1)
    return np.sum(R(transition_function, N, m_eff, params, lambdas) * n_array)

def Rn2_mean(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list, mean_func = lambda n: 1) -> float:
    n_array = np.arange(0,m_eff + 1)**2
    return np.sum(R(transition_function, N, m_eff, params, lambdas) * n_array)

def Rnf_mean(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list, mean_func = lambda n: 1) -> float:
    n_array = np.arange(0,m_eff + 1)
    f_array = mean_func(n_array)
    return np.sum(R(transition_function, N, m_eff, params, lambdas) * f_array * n_array)

def Rf2_mean(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list) -> float:
    n_array = np.arange(0,m_eff + 1)
    def helper(transition_function, n, N, params):
        if n >= N:
            return 0
        else:
            return transition_function(n, N, params)
    f_array = np.array([helper(transition_function, n, N, params) ** 2 for n in n_array])

    numerator = (f_array * R(transition_function, N, m_eff, params, lambdas))
    denominator = R(transition_function, N, m_eff, params, lambdas)
    return np.sum(np.array([numerator[i] / denominator[i] for i in range(len(numerator))]))

def R(transition_function : Callable, N : float, m_eff: float, params : dict, lambdas : list, mean_func = lambda n: 1) -> np.array:
    n_array = np.arange(0,m_eff + 1)
    return np.exp(-1 * lambdas[0] * n_array - lambdas[1] * transition_function(n_array, N, params))
    