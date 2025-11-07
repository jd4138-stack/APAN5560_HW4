import torch
from tqdm import tqdm

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10, z_dim=100):
    if isinstance(model, (tuple, list)):
        G, D = model
    else:
        G, D = model['G'], model['D']
    optim_G, optim_D = optimizer['G'], optimizer['D']
    G.to(device).train(); D.to(device).train()

    for epoch in range(1, epochs + 1):
        g_loss_epoch = d_loss_epoch = 0.0; n_seen = 0
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for real, _ in pbar:
            real = real.to(device); bsz = real.size(0); n_seen += bsz
            # D
            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()
            real_lbl = torch.ones(bsz, 1, device=device)
            fake_lbl = torch.zeros(bsz, 1, device=device)
            D_real = D(real); D_fake = D(fake)
            loss_D = 0.5 * (criterion(D_real, real_lbl) + criterion(D_fake, fake_lbl))
            optim_D.zero_grad(set_to_none=True); loss_D.backward(); optim_D.step()
            # G
            z = torch.randn(bsz, z_dim, device=device)
            gen = G(z); D_gen = D(gen)
            loss_G = criterion(D_gen, real_lbl)
            optim_G.zero_grad(set_to_none=True); loss_G.backward(); optim_G.step()
            d_loss_epoch += loss_D.item() * bsz; g_loss_epoch += loss_G.item() * bsz
            pbar.set_postfix(D_loss=f"{d_loss_epoch/n_seen:.4f}", G_loss=f"{g_loss_epoch/n_seen:.4f}")
        print(f"[Epoch {epoch:03d}] D={d_loss_epoch/n_seen:.4f} | G={g_loss_epoch/n_seen:.4f}")
    return {'G': G, 'D': D}
