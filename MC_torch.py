from time import time

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.special._ufuncs import gammainc
from Utilities.DataGeneration import generate_data


start = time()

# SABR model parameters
#S0 - initial price of the underlying
#sigma0 - initial volatility
#r - risk-free rate
#alpha - volatility of future price volatility
#beta - exponent in SDE
#rho - correlation coefficient
#Time_horizon - time horizon of the simulation
#N_steps - number of steps in a single simulation
#N_simulations - number of simulations
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


class VanillaOption():
    """Class for European options."""
    def __init__(self, T, K, call_or_put):
        """T - time to maturity, K - strike, call_or_put - type of the option (Call or Put)"""
        self.call_or_put = call_or_put
        self.T = T
        self.K = K
        self.N_steps_T = None
        self.price = []
        self.benchmark_price = None
        self.tau_0 = None

    def timesteps_to_maturity(self, N_steps, Time_horizon):
        """Set the number of steps to maturity."""
        self.N_steps_T = int(self.T / Time_horizon * N_steps)

    def payoff(self, S):
        """Calculate the payoff of the option."""
        if self.call_or_put == 'Call':
            payoff = torch.clamp(S-self.K, 0)
        else:
            payoff = torch.clamp(self.K-S, 0)
        return payoff


class SABR_MC():
    def __init__(self, device, F_0, sigma_0, r, alpha, beta, rho, Time_horizon, N_steps, N_simulations, write_to_csv=False):
        self.device=device
        self.F_0 = torch.tensor(F_0, requires_grad=False, device=self.device)
        self.sigma_0 = torch.tensor(sigma_0, requires_grad=False, device=self.device)
        self.r = torch.tensor(r, requires_grad=False, device=self.device)
        self.alpha = torch.tensor(alpha, requires_grad=False, device=self.device)
        self.beta = torch.tensor(beta, requires_grad=False, device=self.device)
        self.rho = torch.tensor(rho, requires_grad=False, device=self.device)
        self.Time_horizon = torch.tensor(Time_horizon, requires_grad=False, device=self.device)
        self.N_steps = N_steps
        self.N_simulations = N_simulations
        self.dt = torch.tensor(Time_horizon / N_steps, requires_grad=False, device=self.device)
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
        """Simulate the SABR model and calculate the price of the options."""
        with torch.no_grad():
            b = 2. - ((1. - 2. * self.beta - (1. - self.beta) * (self.rho ** 2)) / ((1. - self.beta) * (1. - self.rho ** 2)))
            F = torch.zeros(self.N_steps+1, self.N_simulations, device=self.device)
            sigma = torch.zeros(self.N_steps+1, self.N_simulations, device=self.device)
            F[0, :] = self.F_0
            sigma[0, :] = self.sigma_0
            NN = torch.randn(2, self.N_steps, self.N_simulations, device=self.device)
            dW = torch.sqrt(self.dt) * NN[0]
            dB = torch.sqrt(self.dt) * NN[1]
            dW_sigma = dW
            dW_F = torch.sqrt(1 - self.rho ** 2) * dB + self.rho * dW
            for n in range(1, self.N_steps.int() + 1):
                if F[n - 1] == 0:
                    F[n] = 0
                    continue

                sigma[n] = np.abs(self.MilsteinStep(sigma[n - 1], dW_sigma[n - 1]))
                a = (1. / sigma[n - 1]) * (((F[n - 1] ** (1. - self.beta)) / (1. - self.beta) + (self.rho / self.alpha) * (sigma[n] - sigma[n - 1])) ** 2)
                unif = torch.rand(N_simulations, device=self.device)
                pr_zero = AbsorptionConditionalProb(a, b)
                torch.where(pr_zero > unif, F[n], 0)

                F = self.EulerMaruyamaStep(F[n-1], sigma[n-1], dW_F[n-1])
                F = torch.clamp(F, 0)
            for option in options:
                option.N_steps_T = int(option.T / self.Time_horizon * self.N_steps)
                df = np.exp(-self.r * option.T)
                # we calculate option payoff based on the stock price at time of maturity
                S = F[option.N_steps_T] * np.exp(-self.r * (self.Time_horizon - option.T))
                if option.call_or_put == 'Call':
                    payoff = torch.clamp(S - option.K, 0)
                else:
                    payoff = torch.clamp(option.K - S, 0)
                price = payoff * df
                option.price = price.mean()
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
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    #     torch.cuda.empty_cache()
    # else:
    #     device = 'cpu'
    device='cpu'
    print(f'Using {device}')

    # Initial parameters
    F_0 = 100  # initial forward price of the underlying
    sigma_0 = 0.3  # initial volatility of the underlying
    r = 0.05  # risk-free rate
    alpha = 0.2  # volatility of future price volatility
    beta = 0.6  # exponent in SDE
    rho = 0.2  # correlation coefficient
    Time_horizon = 2  # time horizon of the simulation
    N_steps = int(12 * 1e3)  # number of steps in a single simulation
    N_simulations = int(1e5)  # number of simulations

    generate_data(F_0, Time_horizon, 'quarterly', 'StandardApproach/Data/Options.csv')
    options_csv = pd.read_csv('StandardApproach/Data/Options.csv', sep=',')
    options = [VanillaOption(row[0], row[1], row[2]) for row in zip(options_csv['Expiration_date'], options_csv['Strike'], options_csv['Call_or_Put'])]
    mc = SABR_MC(device=device, F_0=F_0, sigma_0=sigma_0, r=r, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon, N_steps=N_steps, N_simulations=N_simulations)
    options = mc.simulate(options)
    mc_time = time()
    options = benchmark_price_SABR(options, F_0, sigma_0, beta, r, Time_horizon)
    benchmark_time = time()
    options_csv['Benchmark_price'] = [option.benchmark_price for option in options]
    options_csv['Price'] = [option.price for option in options]
    options_csv.to_csv('Data/Options_results.csv', index=False)

    print('MC time: ', mc_time - start)
    print('Benchmark time: ', benchmark_time - mc_time)
