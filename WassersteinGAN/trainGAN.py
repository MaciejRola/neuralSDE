import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def train_GAN(generator, discriminator, target, epochs, batch_size, device, n_critic=2, gp=1, lr=1e-4):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizerG = optim.Adam(generator.parameters(), 10 * lr)
    optimizerD = optim.Adam(discriminator.parameters(), lr / 10)
    loss_val_best = 1e10
    losses_G = []
    losses_D = []

    G_best = None
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for _ in range(n_critic):
            loss_D = torch.zeros(1, requires_grad=True, device=device)
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            for batch in range(0, 20 * batch_size, batch_size):
                # print(batch)
                # print(torch.cuda.memory_stats(device=device))
                real = target[batch:batch + batch_size].to(device)
                # print(torch.cuda.memory_stats(device=device))
                fake = generator.simulate_paths(batch_size)
                # print(torch.cuda.memory_stats(device=device))
                u = torch.rand(1, device=device)
                combination = real * u + fake * (1 - u)
                regularizer = torch.autograd.grad(discriminator(combination).sum(), combination, create_graph=True)[0]
                loss_D = loss_D + discriminator(fake) - discriminator(real) + gp * (torch.norm(regularizer, 2) - 1) ** 2
            loss_D = (loss_D / 20).mean()
            print(f'Epoch: {epoch}, loss_D: {loss_D.item()}')
            losses_D.append(loss_D.item())
            loss_D.backward()
            optimizerD.step()
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        loss_G = torch.zeros(1, requires_grad=True, device=device)
        for _ in range(20):
            fake = generator.simulate_paths(batch_size)
            loss_G = loss_G - discriminator(fake)
        loss_G = (loss_G / 20).mean()
        losses_G.append(loss_G.item())
        loss_G.backward()
        optimizerG.step()
        print(f'Epoch: {epoch}, loss_G: {loss_G.item()}, loss_D: {loss_D.item()}')
        if len(losses_G) > 1 and len(losses_D) > 1:
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(losses_G, label='Generator loss')
            axs[0].legend()
            axs[1].plot(losses_D, label='Discriminator loss')
            axs[1].legend()
            plt.savefig(f'./neuralSDE/WassersteinGAN/Results/loss_WassersteinGAN_{generator.__class__.__name__}.png')
            plt.close()

        if loss_G.item() < loss_val_best:
            G_best = generator
            loss_val_best = loss_G.item()
            print(f'loss_val_best: {loss_val_best}')
            filename = f'./neuralSDE/WassersteinGAN/Results/WassersteinGAN_{generator.__class__.__name__}.pth.tar'
            checkpoint = {'state_dict': generator.state_dict(), 'loss': loss_val_best}
            torch.save(checkpoint, filename)

    return G_best
