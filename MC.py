import numpy as np
import pandas as pd
from scipy import stats
from time import time

start = time()
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


class VanillaOption():
    def __init__(self, T, K, call_or_put):
        self.call_or_put = call_or_put
        self.T = T
        self.K = K
        self.N_steps_T = None
        self.price = []
        self.benchmark_price = None
        self.tau_0 = None

    def timesteps_to_maturity(self, N_steps, Time_horizon):
        self.N_steps_T =  int(self.T / Time_horizon * N_steps)

    def payoff(self, S):
        if self.tau_0 <= self.N_steps_T:
            payoff = 0
        else:
            if self.call_or_put == 'Call':
                payoff = max(S-self.K, 0)
            else:
                payoff = max(self.K-S, 0)
        return payoff


class SABR_MC():
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

    def drift_log_F(self, sigma, F):
        return -1/2 * sigma**2 * F**(2*self.beta-2)

    def vol_log_F(self, sigma, F):
        return sigma * F**(self.beta-1)

    def vol_sigma(self, sigma):
        return self.alpha**2 * sigma

    def EulerMaruyamaStep(self, F, sigma, dW):
        return F * np.exp(self.drift_log_F(sigma, F) * self.dt + self.vol_log_F(sigma, F) * dW)

    def MilsteinStep(self, sigma, dW):
        return sigma + self.vol_sigma(sigma) * dW + self.alpha**2 * self.vol_sigma(sigma) * (dW**2 - self.dt)

    def simulate(self, options):
        for m in range(self.N_simulations + 1):
            F = np.zeros((2, self.N_steps+1))
            sigma = np.zeros((2, self.N_steps+1))
            F[0, 0] = F[1, 0] = self.F_0
            sigma[0, 0] = sigma[1, 0] = self.sigma_0
            tau_0 = self.N_steps+1
            NN = np.random.randn(2, self.N_steps)
            dW_sigma = np.sqrt(self.dt) * NN[1, :]
            dW_F = np.sqrt(self.dt) * (np.sqrt(1 - self.rho ** 2) * NN[0, :] + self.rho * NN[1, :])
            for n in range(1, self.N_steps + 1):
                F1 = self.EulerMaruyamaStep(F[0, n-1], sigma[0, n-1], dW_F[n-1])
                F2 = self.EulerMaruyamaStep(F[1, n-1], sigma[1, n-1], -dW_F[n-1])
                sigma[0, n] = np.abs(self.MilsteinStep(sigma[0, n-1], dW_sigma[n-1]))
                sigma[1, n] = np.abs(self.MilsteinStep(sigma[1, n-1], -dW_sigma[n-1]))
                if F1 <= 0 or F2 <= 0:
                    tau_0 = n
                else:
                    F[0, n] = F1
                    F[1, n] = F2
            for option in options:
                option.N_steps_T = int(option.T / self.Time_horizon * self.N_steps)
                option.tau_0 = tau_0
                payoff_plus = option.payoff(F[0, option.N_steps_T]*np.exp(-self.r * (self.Time_horizon - option.T)))
                payoff_minus = option.payoff(F[1, option.N_steps_T]*np.exp(-self.r * (self.Time_horizon - option.T)))
                df = np.exp(-self.r * option.T)
                price = (payoff_plus + payoff_minus) / 2 * df
                option.price.append(price)
        for option in options:
            option.price = np.mean(option.price)
        return options


def benchmark_price(options, F_0, sigma_0, beta, Time_horizon):
    for option in options:
        denom1 = (1-beta)**2 * sigma_0**2 * Time_horizon
        denom2 = 1-beta
        nom1 = option.K**(2-2*beta)
        nom2 = F_0**(2-2*beta)
        nom3 = 3-2*beta
        Q1 = stats.ncx2(df=nom3/denom2, nc=nom2/denom1)
        Q2 = stats.ncx2(df=1/denom2, nc=nom1/denom1)
        if option.call_or_put == 'Call':
            benchmark_price = F_0*(1-Q1.cdf(nom1/denom1))-option.K*Q2.cdf(nom2/denom1)
        else:
            benchmark_price = option.K*(1-Q2.cdf(nom2/denom1))-F_0*Q1.cdf(nom1/denom1)
        option.benchmark_price = benchmark_price
    return options


F_0 = 100
sigma_0 = 0.3
r = 0.05
alpha = 0.2
beta = 0.6
rho = 0.2
Time_horizon = 2
N_steps = 1e4
N_simulations = 1e5

options_csv = pd.read_csv('Data/Options.csv', sep=',')
options = [VanillaOption(row[0], row[1], row[2]) for row in zip(options_csv['Expiration_date'], options_csv['Strike'], options_csv['Call_or_Put'])]
mc = SABR_MC(F_0=F_0, sigma_0=sigma_0, r=r, alpha=alpha, beta=beta, rho=rho, Time_horizon=Time_horizon, N_steps=N_steps, N_simulations=N_simulations)
options = mc.simulate(options)
options = benchmark_price(options, F_0, sigma_0, beta, Time_horizon)
options_csv['Benchmark_price'] = [option.benchmark_price for option in options]
options_csv['Price'] = [option.price for option in options]
options_csv = options_csv.sample(frac=1).reset_index(drop=True)
split = len(options)//10
options_train = options_csv[split:]
options_train.to_csv('Data/Options_train.csv', index=False)
options_test = options_csv[:split]
options_test.to_csv('Data/Options_test.csv', index=False)
