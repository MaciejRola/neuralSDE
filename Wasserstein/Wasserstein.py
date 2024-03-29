import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def Wasserstein_metric(pred, target, p=1):
    # compute the Wasserstein metric between empirical cdfs
    pred, _ = torch.sort(pred, dim=1)
    target, _ = torch.sort(target)
    W_p = torch.mean(torch.pow(torch.abs(pred - target), p)) ** (1 / p)
    return W_p


def train_Wasserstein(model, target, epochs, batch_size, threshold=1e-5):
    loss_fn = Wasserstein_metric
    model = model.to(model.device)
    modelling_vol = hasattr(model, 'V0')
    N_simulations = model.N_simulations
    N_steps = model.N_steps

    parameters_SDE = list(model.parameters())
    optimizer_SDE = optim.Adam(parameters_SDE, lr=1e-2)
    scheduler_SDE = optim.lr_scheduler.MultiStepLR(optimizer_SDE, milestones=[500, 800], gamma=0.2, verbose=True)

    best_model = None
    loss_val_best = 1e10
    LOSSES = []
    for epoch in range(epochs):
        batch_losses = torch.zeros(20, requires_grad=False)
        for batch in range(0, 20 * batch_size, batch_size):
            model.train()
            S = model.S0.repeat(1, batch_size).to(model.device)
            if modelling_vol:
                V = model.V0.repeat(1, batch_size).to(model.device)
            total_loss = torch.tensor(0, dtype=torch.float32, device=model.device)
            for step in range(1, N_steps + 1):
                curr_target = target[batch:batch + batch_size, step].to(model.device)
                if modelling_vol:
                    S, V = model.simulate_values(S, V, step)
                else:
                    S = model.simulate_values(S, step)
                if step % model.period_length == 0:
                    loss = loss_fn(S, curr_target)
                    total_loss += loss

            optimizer_SDE.zero_grad()
            batch_losses[batch // batch_size] = total_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(parameters_SDE, 5)
            optimizer_SDE.step()

        with torch.no_grad():
            model.eval()
            S = model.S0.repeat(1, N_simulations).to(model.device)
            if modelling_vol:
                V = model.V0.repeat(1, N_simulations).to(model.device)
            test_loss = torch.tensor(0, device=model.device, dtype=torch.float32)
            for step in range(1, N_steps + 1):
                if modelling_vol:
                    S, V = model.simulate_values(S, V, step)
                else:
                    S = model.simulate_values(S, step)
                pred = S
                if step % model.period_length == 0:
                    loss = loss_fn(pred, target[:, step].to(model.device))
                    test_loss += loss
        print(f'Epoch: {epoch}, Loss: {test_loss.item()}')
        scheduler_SDE.step()

        with open(f"./neuralSDE/Wasserstein/Results/log_eval_Wasserstein_{model.__class__.__name__}.txt", "a") as f:
            f.write(f'{epoch},{test_loss.item()}\n')

        LOSSES.append(test_loss.item())
        if len(LOSSES) > 1:
            plt.plot(LOSSES, label='Wasserstein loss')
            plt.legend()
            plt.savefig(f'./neuralSDE/Wasserstein/Results/loss_Wasserstein_{model.__class__.__name__}.png')
            plt.close()

        if test_loss.item() < loss_val_best:
            model_best = model
            loss_val_best = test_loss.item()
            print(f'loss_val_best: {loss_val_best}')
            filename = f'./neuralSDE/Wasserstein/Results/Wasserstein_{model.__class__.__name__}.pth.tar'
            checkpoint = {'state_dict': model.state_dict(), 'loss': loss_val_best}
            torch.save(checkpoint, filename)

        if total_loss.item() < threshold:
            break

    return model_best