import torch


class GaussianLogLikelihoodLoss(torch.nn.Module):
    def __init__(
            self, 
            naive_mean: float, 
            naive_std: float, 
            spectrum_len: int = 283,
            fsg_sigma_true: float = 1e-6,
            airs_sigma_true: float = 1e-5,
            fgs_weight = 2
            ) -> None:
        super().__init__()
        self.naive_mean = torch.tensor(naive_mean)
        self.naive_sigma = torch.tensor(naive_std)
        self.spectrum_len = spectrum_len

        self.fsg_sigma_true = torch.tensor(fsg_sigma_true)
        self.airs_sigma_true = torch.tensor(airs_sigma_true)
        self.fgs_weight = torch.tensor(fgs_weight)

    def forward(self, y_true: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        y_pred = output[:, :self.spectrum_len]
        sigma_pred = torch.clamp(output[:, self.spectrum_len:], min=1e-15)

        sigma_true = torch.cat((torch.tensor([self.fsg_sigma_true]), torch.ones(283 - 1) * self.airs_sigma_true)).to(y_true.device)

        GLL_pred = -0.5 * ((y_true - y_pred) ** 2 / (sigma_pred ** 2) + torch.log(2 * torch.pi * sigma_pred ** 2))
        GLL_true = -0.5 * ((y_true - y_true) ** 2 / (sigma_true ** 2) + torch.log(2 * torch.pi * sigma_true ** 2))
        GLL_mean = -0.5 * ((y_true - self.naive_mean) ** 2 / (self.naive_sigma ** 2) + torch.log(2 * torch.pi * self.naive_sigma ** 2))

        ind_scores = (GLL_pred - GLL_mean) / (GLL_true - GLL_mean)

        # Create weights
        weights = torch.cat((torch.tensor([self.fgs_weight]), torch.ones(283 - 1))).to(y_true.device)
        weights = weights.expand_as(ind_scores)

        weighted_sum = (ind_scores * weights).sum()
        total_weight = weights.sum()
        loss = weighted_sum / total_weight
        
        return torch.log(-loss)