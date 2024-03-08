# neuralSDE
## Target data
In both approaches target data is produced by Monte Carlo simulations under SABR model 

$\text{d}F_t = \sigma_t F_t^{\beta}\text{d}W_t, F_0 = f$

$\text{d}\sigma_t = \alpha \sigma_t \text{d}B_t, \sigma_0 = \sigma $

$\text{d}W_t\text{d}B_t = \rho \text{d}t$

with following parameters:
- $f = 1$ (initial forward price) 
- $\sigma = 0.3$ (initial volatility of the underlying)
- $\alpha = 0.2$ (volatility of forward price volatility)
- $\beta = 0.6$
- $\rho = 0.2$

Simulation parameters:
- $N_{simulations} = 10^5 $
- $N_{steps} = 1.2 * 10^3$

  
## Parameters
'market' parameters are:
- $T = 2$ (Time horizon)
- $r = 0.05$ (Risk free rate)
- $s = e^{- r T}f $
- $v$ (if applicable, initial volatility is initiated randomly and optimised during training)
- $\rho$ (if applicable, correlation between Wiener processes is initiated randomly and optimised during training)

Simulation parameters for neuralSDE:
- $N_{simulations} = 4*10^5$
- $N_{steps} = 96 $
- $N_{options} = 4 $
- $N_{maturities} = 21 $
  
Our model takes one of three possible parametrizations:
- NeuralLV (neural Local Volatility)

  $\text{d}S_t = r S_t \text{d}t + \sigma(t, S_t) S_t \text{d}W_t, S_0 = s$
- NeuralLSV (neural Local Stochastic Volatility)

  $\text{d}S_t = r S_t \text{d}t + \sigma_S(t, S_t, V_t) S_t \text{d}W_t, S_0 = s$
  
  $\text{d}V_t = b_V(V_t) \text{d}t + \sigma_V(V_t) \text{d}B_t, V_0 = v$
  
  $\text{d}W_t\text{d}B_t=\rho\text{d}t$
- NeuralSDE (neural Stochastic Volatility)

  $\text{d}S_t = r S_t \text{d}t + \sigma_S(t, S_t, V_t) S_t \text{d}W_t, S_0 = s$
  
  $\text{d}V_t = b_V(t, S_t, V_t) \text{d}t + \sigma_V(t, S_t, V_t) \text{d}B_t, V_0 = v$
  
  $\text{d}W_t\text{d}B_t=\rho\text{d}t$,

where all functions are given by feedforward neural networks with optional batch normalization layers or residual connections. Then our model is used in tamed Euler algorithm.

## Standard approach
In the standard approach we train models based on Mean Square Error between option prices from the model and from Monte Carlo simulations. File 'StandardApproach/Data/Options_results.csv' contains option prices for chosen maturities.
## Wasserstein approach
In Wasserstein approach we train models based on similarity of distributions between paths from SABR model and paths from our model at various timesteps. File 'Wasserstein/Data/Wasserstein_target.pth.tar' contains simulated paths from SABR model. Our model simulates paths iteratively during training procedure.
## Wasserstein GAN approach
In Wasserstein GAN approach we train two models: generator (based on above models) and discriminator (standard MLP). Our goal is to train generator to succesfully trick the trained discriminator. We make use of Wasserstein GAN with gradient penalty.
