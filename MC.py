from time import time

import numpy as np
import pandas as pd
from scipy import stats, integrate
from scipy.special._ufuncs import gammainc

from DataGeneration import generate_data

start = time()


# S0 - initial price of the underlying
# sigma0 - initial volatility
# r - risk-free rate
# alpha - volatility of future price volatility
# beta - exponent in SDE
# rho - correlation coefficient
# Time_horizon - time horizon of the simulation
# N_steps - number of steps in a single simulation
# N_simulations - number of simulations
# SABR model
# dS_t = sigma_t S_t^beta dW_t^1
# dsigma_t = alpha sigma_t dW_t^2
# S(0) = S_0 exp(rT)
# S(t) = bar_S(t) exp(r(T-t))
# bar_S(t) = S_t exp(-r(T-t))
# d<W^1, W^2> = rho dt


def AbsorptionConditionalProb(a, b):
    """ probability that F_n is absorbed by the 0 barrier conditional on inital value F_n-1  """
    cprob = 1. - gammainc(b / 2, a / 2)  # formula (2.10), scipy gammainc is already normalized by gamma(b)
    return cprob


class VanillaOption:
    """Class for European options."""

    def __init__(self, T, K, call_or_put):
        self.call_or_put = call_or_put
        self.T = T
        self.K = K
        self.N_steps_T = None
        self.price = []
        self.benchmark_price = None
        self.tau_0 = None

    def timesteps_to_maturity(self, N_steps, Time_horizon):
        """Set the number of timesteps to maturity."""
        self.N_steps_T = int(self.T / Time_horizon * N_steps)

    def payoff(self, S):
        """Payoff function for the option."""
        if self.call_or_put == 'Call':
            payoff = max(S - self.K, 0)
        else:
            payoff = max(self.K - S, 0)
        return payoff


class SABR_MC():
    """Class for the SABR model using the Euler-Maruyama scheme for forward price and Milstein scheme for volatility."""

    def __init__(self, F_0, sigma_0, r, alpha, beta, rho, Time_horizon, N_steps, N_simulations, write_to_csv=False):
        self.F_0 = F_0
        self.sigma_0 = sigma_0
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Time_horizon = Time_horizon
        self.N_steps = int(N_steps)
        self.N_simulations = int(N_simulations)
        self.dt = Time_horizon / N_steps
        self.write_to_csv = write_to_csv

    def vol_F(self, sigma, F):
        """Volatility function for the forward stock price."""
        return sigma * F ** self.beta

    def vol_sigma(self, sigma):
        """Volatility function for the variance process."""
        return self.alpha * sigma

    def vol_sigma_prime(self, sigma):
        """Derivative of the volatility function with respect to sigma."""
        return self.alpha

    def EulerMaruyamaStep(self, F, sigma, dW):
        """Euler-Maruyama step for the stock price."""
        return F + self.vol_F(sigma, F) * dW

    def MilsteinStep(self, sigma, dW):
        """Milstein step for the variance process."""
        return sigma + self.vol_sigma(sigma) * dW + 1 / 2 * self.vol_sigma_prime(sigma) * self.vol_sigma(sigma) * (dW ** 2 - self.dt)

    def simulate(self, options):
        """Simulate the SABR model."""
        b = 2. - ((1. - 2. * self.beta - (1. - self.beta) * (self.rho ** 2)) / ((1. - self.beta) * (1. - self.rho ** 2)))
        for m in range(self.N_simulations + 1):
            F = np.zeros(self.N_steps + 1)
            sigma = np.zeros(self.N_steps + 1)
            F[0] = self.F_0
            sigma[0] = self.sigma_0
            NN = np.random.randn(2, self.N_steps)
            dW = np.sqrt(self.dt) * NN[0, :]
            dB = np.sqrt(self.dt) * NN[1, :]
            dW_sigma = dW
            dW_F = np.sqrt(1 - self.rho ** 2) * dB + self.rho * dW
            for n in range(1, self.N_steps + 1):
                if F[n - 1] == 0:
                    F[n] = 0
                    continue

                sigma[n] = np.abs(self.MilsteinStep(sigma[n - 1], dW_sigma[n - 1]))
                a = (1. / sigma[n - 1]) * (((F[n - 1] ** (1. - self.beta)) / (1. - self.beta) + (self.rho / self.alpha) * (sigma[n] - sigma[n - 1])) ** 2)
                unif = np.random.uniform(0, 1)
                pr_zero = AbsorptionConditionalProb(a, b)
                if pr_zero > unif:
                    F[n] = 0
                    continue

                F[n] = self.EulerMaruyamaStep(F[n - 1], sigma[n - 1], dW_F[n - 1])
            for option in options:
                option.N_steps_T = int(option.T / self.Time_horizon * self.N_steps)
                df = np.exp(-self.r * option.T)
                # we calculate option payoff based on the stock price at time of maturity
                S = F[option.N_steps_T] * np.exp(-self.r * (self.Time_horizon - option.T))
                payoff = option.payoff(S)
                price = payoff * df
                option.price.append(price)
        for option in options:
            option.price = np.mean(option.price)
        return options


class Heston_MC():
    """Class for the Heston model using the Euler-Maruyama and Milstein methods."""

    def __init__(self, S_0, sigma_0, r, kappa, mu, eta, rho, Time_horizon, N_steps, N_simulations, write_to_csv=False):
        self.S_0 = S_0
        self.sigma_0 = sigma_0
        self.r = r
        self.kappa = kappa
        self.mu = mu
        self.eta = eta
        self.rho = rho
        self.Time_horizon = Time_horizon
        self.N_steps = int(N_steps)
        self.N_simulations = int(N_simulations)
        self.dt = Time_horizon / N_steps
        self.write_to_csv = write_to_csv

    def drift_S(self, S):
        """Drift function for the stock price."""
        return self.r * S

    def vol_S(self, sigma, S):
        """Volatility function for the stock price."""
        return S * np.sqrt(sigma)

    def drift_sigma(self, sigma):
        """Drift function for the variance process."""
        return self.kappa * (self.mu - sigma)

    def vol_sigma(self, sigma):
        """Volatility function for the variance process."""
        return self.eta * np.sqrt(sigma)

    def vol_sigma_prime(self, sigma):
        """Derivative of the volatility function with respect to sigma."""
        return self.eta / (2 * np.sqrt(sigma))

    def EulerMaruyamaStep(self, S, sigma, dW):
        """Euler-Maruyama step for the stock price."""
        return S + self.drift_S(S) * self.dt + self.vol_S(sigma, S) * dW

    def MilsteinStep(self, sigma, dW):
        """Milstein step for the variance process."""
        return sigma + self.drift_sigma(sigma) * self.dt + self.vol_sigma(sigma) * dW + 1 / 2 * self.vol_sigma(sigma) * self.vol_sigma_prime(sigma) * (dW ** 2 - self.dt)

    def simulate(self, options):
        """Simulate the Heston model."""
        for m in range(self.N_simulations + 1):
            S = np.zeros((self.N_steps + 1))
            sigma = np.zeros((self.N_steps + 1))
            S[0] = self.S_0
            sigma[0] = self.sigma_0
            NN = np.random.randn(2, self.N_steps)
            dW_sigma = np.sqrt(self.dt) * NN[1, :]
            dW_F = np.sqrt(self.dt) * (np.sqrt(1 - self.rho ** 2) * NN[0, :] + self.rho * NN[1, :])
            for n in range(1, self.N_steps + 1):
                S[n] = self.EulerMaruyamaStep(S[n - 1], sigma[n - 1], dW_F[n - 1])
                sigma[n] = np.abs(self.MilsteinStep(sigma[n - 1], dW_sigma[n - 1]))
                sigma[n] = np.abs(self.MilsteinStep(sigma[n - 1], -dW_sigma[n - 1]))
            for option in options:
                option.N_steps_T = int(option.T / self.Time_horizon * self.N_steps)
                payoff = option.payoff(S[option.N_steps_T] * np.exp(-self.r * (self.Time_horizon - option.T)))
                df = np.exp(-self.r * option.T)
                price = payoff * df
                option.price.append(price)
        for option in options:
            option.price = np.mean(option.price)
        return options


def benchmark_price_Heston(options, S_0, r, sigma_0, tau, kappa, mu, eta, rho):
    """Benchmark price for European options in the Heston model."""""

    def phi(u, tau):
        """Characteristic function of the log-stock price."""
        alpha_hat = -0.5 * u * (u + 1j)
        beta = kappa - 1j * u * eta * rho
        gamma = 0.5 * eta ** 2
        d = np.sqrt(beta ** 2 - 4 * alpha_hat * gamma)
        g = (beta - d) / (beta + d)
        h = np.exp(-d * tau)
        A_ = (beta - d) * tau - 2 * np.log((g * h - 1) / (g - 1))
        A = kappa * mu / (eta ** 2) * A_
        B = (beta - d) / (eta ** 2) * (1 - h) / (1 - g * h)
        return np.exp(A + B * sigma_0)

    def integral(k, tau):
        """Integral in the benchmark price formula."""
        integrand = (lambda u: np.real(np.exp((1j * u + 0.5) * k) * phi(u - 0.5j, tau)) / (u ** 2 + 0.25))
        i, _, _ = integrate.quad_vec(integrand, 0, np.inf)
        return i

    def call(S_0, option):
        """Benchmark price for a call option."""
        K = option.K
        tau = option.T
        a = np.log(S_0 / K) + r * tau
        i = integral(a, tau)
        return S_0 - option.K * np.exp(-r * tau) / np.pi * i

    for option in options:
        option.benchmark_price = call(S_0, option)
    return options


def benchmark_price_SABR(options, F_0, sigma_0, beta, r, Time_horizon):
    """Benchmark price for European options in SABR model."""
    S_0 = F_0 * np.exp(-r * Time_horizon)
    for option in options:
        F = S_0 * np.exp(r * option.T)
        denom1 = (1 - beta) ** 2 * sigma_0 ** 2 * option.T
        denom2 = 1 - beta
        nom1 = option.K ** (2 - 2 * beta)
        nom2 = F ** (2 - 2 * beta)
        nom3 = 3 - 2 * beta
        Q1 = stats.ncx2(df=nom3 / denom2, nc=nom2 / denom1)
        Q2 = stats.ncx2(df=1 / denom2, nc=nom1 / denom1)
        if option.call_or_put == 'Call':
            benchmark_price = F * (1 - Q1.cdf(nom1 / denom1)) - option.K * Q2.cdf(nom2 / denom1)
        else:
            benchmark_price = option.K * (1 - Q2.cdf(nom2 / denom1)) - F * Q1.cdf(nom1 / denom1)
        option.benchmark_price = benchmark_price
    return options


if __name__ == '__main__':
    # Initial parameters
    F_0 = 100  # initial forward price of the underlying
    sigma_0 = 0.3  # initial volatility of the underlying
    r = 0.05  # risk-free rate
    alpha = 0.2  # volatility of future price volatility
    beta = 0.6  # exponent in SDE
    rho = 0.2  # correlation coefficient
    Time_horizon = 2  # time horizon of the simulation
    N_steps = 12 * 1e3  # number of steps in a single simulation
    N_simulations = 1e5  # number of simulations

    generate_data(F_0, Time_horizon, 'quarterly', 'Data/Options.csv')
    options_csv = pd.read_csv('Data/Options.csv', sep=',')
    options = [VanillaOption(row[0], row[1], row[2]) for row in zip(options_csv['Expiration_date'], options_csv['Strike'], options_csv['Call_or_Put'])]
    mc = SABR_MC(F_0=F_0, sigma_0=sigma_0, r=r, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon, N_steps=N_steps, N_simulations=N_simulations)
    options = mc.simulate(options)
    mc_time = time()
    options = benchmark_price_SABR(options, F_0, sigma_0, beta, r, Time_horizon)
    benchmark_time = time()
    options_csv['Benchmark_price'] = [option.benchmark_price for option in options]
    options_csv['Price'] = [option.price for option in options]
    options_csv.to_csv('Data/Options_results.csv', index=False)

    print('MC time: ', mc_time - start)
    print('Benchmark time: ', benchmark_time - mc_time)
