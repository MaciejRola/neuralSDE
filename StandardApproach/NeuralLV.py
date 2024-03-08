import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from Models.NeuralLV import NeuralLV


def train(model, maturities, strikes, target, batch_size, epochs, threshold=2e-5):
    loss_fn = nn.MSELoss()

    model = model.to(model.device)
    networks_SDE = [model.leverage]
    parameters_SDE = list(model.leverage.parameters())
    optimizer_SDE = optim.Adam(parameters_SDE, lr=1e-3)
    scheduler_SDE = optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500, 800], gamma=0.2, verbose=True)
    if model.use_hedging:
        optimizer_Hedging = optim.Adam(list(model.Hedging_Vanilla.parameters()), lr=1e-3)

    loss_val_best = 10
    best_model = None
    itercount = 0
    LOSSES = []
    if model.use_hedging:
        VARIANCES = []
    for epoch in range(epochs):
        if model.use_hedging:
            optim_SDE = (epoch % 2 == 0)
            print(f'SDE parameters optimization: {optim_SDE}')
            if optim_SDE:
                for net in networks_SDE:
                    net.unfreeze()
                for param in parameters_SDE:
                    param.requires_grad_(True)
                model.Hedging_Vanilla.freeze()
            else:
                for net in networks_SDE:
                    net.freeze()
                for param in parameters_SDE:
                    param.requires_grad_(False)
                model.Hedging_Vanilla.unfreeze()
            for batch in range(0, 20 * batch_size, batch_size):
                optimizer_SDE.zero_grad()
                optimizer_Hedging.zero_grad()
                init_time = time.time()
                prices, prices_var = model(maturities, strikes, test=False)
                time_forward = time.time() - init_time
                itercount += 1

                if optim_SDE:
                    loss = loss_fn(prices, target)
                    init_time = time.time()
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters_SDE, 5)
                    time_backward = time.time() - init_time
                    optimizer_SDE.step()
                else:
                    loss_sample_var = prices_var.sum()
                    init_time = time.time()
                    loss = loss_sample_var
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.Hedging_Vanilla.parameters(), 1)
                    time_backward = time.time() - init_time
                    optimizer_Hedging.step()
                print(f'iteration: {itercount}, loss: {loss.item()}, time_forward={time_forward}, time_backward={time_backward}')
        else:
            for batch in range(0, 20 * batch_size, batch_size):
                optimizer_SDE.zero_grad()
                init_time = time.time()
                prices, _ = model(maturities, strikes, test=False)
                time_forward = time.time() - init_time
                itercount += 1
                loss = loss_fn(prices, target)
                init_time = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters_SDE, 5)
                time_backward = time.time() - init_time
                optimizer_SDE.step()
                print(f'iteration: {itercount}, loss: {loss.item()}, time_forward={time_forward}, time_backward={time_backward}')
        scheduler_SDE.step()

        with torch.no_grad():
            prices_mean, prices_var = model(maturities, strikes, test=True)
            print(f'Prices: \n{prices_mean}, \n Target: \n{target}, \n Variance: \n{prices_var}')

        MSE = loss_fn(prices_mean, target)
        loss_val = torch.sqrt(MSE)
        print(f'epoch={epoch}, loss={loss_val.item()}')
        with open("Results/log_eval_NeuralLV.txt", "a") as f:
            f.write(f'{epoch},{loss_val.item()}\n')

        LOSSES.append(loss_val.item())
        if model.use_hedging:
            VARIANCES.append(prices_var.mean().item())
            if len(LOSSES) > 1 and len(VARIANCES) > 1:
                fig, axs = plt.subplots(1, 2)
                axs[0].plot(LOSSES, label='RMSE Error')
                axs[0].legend()
                axs[1].plot(VARIANCES, label='Mean price Variance')
                axs[1].legend()
                plt.savefig('Results/loss_and_var_NeuralLV.png')
                plt.close()
        else:
            if len(LOSSES) > 1:
                plt.plot(LOSSES, label='RMSE Error')
                plt.legend()
                plt.savefig('Results/loss_neuralLV.png')
                plt.close()

        if loss_val < loss_val_best:
            model_best = model
            loss_val_best = loss_val
            print(f'loss_val_best: {loss_val_best.item()}')
            filename = 'Results/NeuralLV.pth.tar'
            checkpoint = {'state_dict': model.state_dict(), 'pred': prices_mean, 'target': target}
            torch.save(checkpoint, filename)

        if loss_val.item() < threshold:
            break

    return model_best


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = 'cpu'
    print(f'Using {device}')

    options = pd.read_csv('Data/Options_results.csv')
    options = options[options['Expiration_date'].isin([0.5, 1.0, 1.5, 2.0])]
    maturities = options['Expiration_date'].unique()
    strikes = options['Strike'].unique() / 100
    target = pd.pivot_table(options, values='Price', index='Strike', columns='Expiration_date').to_numpy() / 100
    target = torch.tensor(target, requires_grad=True, dtype=torch.float, device=device)

    # market parameter
    F0 = 100 / 100
    n_maturities = maturities.shape[0]
    n_strikes = strikes.shape[0]
    Time_horizon = 2
    rfr = 0.05
    S0 = F0 * np.exp(-rfr * Time_horizon)

    # model parameters
    num_layers = 4
    num_layers_hedging = 3
    layer_size = 50
    layer_size_hedging = 30
    dropout = 0.0
    use_hedging = True
    use_batchnorm = False

    # simulation parameters
    batch_size = 40000
    epochs = 1000
    N_simulations = 20 * batch_size
    N_steps = 96
    period_length = N_steps // n_maturities

    # Monte Carlo test data
    MC_samples_test = 200000
    test_normal_variables = torch.randn(MC_samples_test, N_steps, requires_grad=False)
    # We will use antithetic Brownian paths for testing
    test_normal_variables = torch.cat([test_normal_variables, -test_normal_variables], 0)

    with open("Results/log_eval_NeuralLV.txt", "w") as f:
        f.write('epoch,loss\n')

    model = NeuralLV(device=device, batch_size=batch_size, dropout=dropout, use_batchnorm=use_batchnorm, use_hedging=use_hedging,
                     N_simulations=N_simulations, N_steps=N_steps, Time_horizon=Time_horizon, period_length=period_length,
                     S0=S0, n_maturities=n_maturities, n_strikes=n_strikes, rfr=rfr,
                     num_layers=num_layers, layer_size=layer_size,
                     num_layers_hedging=num_layers_hedging, layer_size_hedging=layer_size_hedging,
                     test_normal_variables=test_normal_variables)
    summary(model)
    print('Model initiated')
    train(model, maturities=maturities, strikes=strikes, target=target, batch_size=batch_size, epochs=epochs, threshold=2e-5)
    print('Model trained')
