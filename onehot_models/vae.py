import torch

class VAE(torch.nn.Module):
    def __init__(self, input_size, latent_size = 32, hidden_size = 128):

        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.encode_fc1 = torch.nn.Linear(input_size, hidden_size)
        self.encode_mu = torch.nn.Linear(hidden_size, latent_size)
        self.encode_var = torch.nn.Linear(hidden_size, latent_size)

        self.decode_fc1 = torch.nn.Linear(latent_size, hidden_size)
        self.decode_fc2 = torch.nn.Linear(hidden_size, input_size)

    def encode(self, x):
        x = self.encode_fc1(x).relu()
        mu = self.encode_mu(x)
        logvar = self.encode_var(x)
        return mu, logvar

    def decode(self, z):
        z = self.decode_fc1(z).relu()
        z = self.decode_fc2(z)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar 

def loss_fn(recon_x, x, mu, logvar):
    pass

