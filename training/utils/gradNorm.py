import torch


class GradNorm:
    def __init__(self, num_task, layer, lr0=1e-4, alpha=0.1):
        self.lr0 = lr0
        self.layer = layer
        self.layer_count = sum(1 for _ in layer.parameters())

        device = next(layer.parameters()).device
        self.weights = torch.ones(num_task, device=device)
        self.weights = torch.nn.Parameter(self.weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=lr0)
        self.init_loss = None
        self.T = self.weights.sum().detach()
        self.alpha = alpha

    def __call__(self, losses):
        if self.init_loss is None:
            self.init_loss = losses.detach()

        # calculate weighted loss for entire model
        weighted_loss = self.weights @ losses
        weighted_loss.backward(retain_graph=True)

        gw = []
        for i in range(len(losses)):
            grad_weight = torch.autograd.grad(
                self.weights[i] * losses[i],
                self.layer.parameters(),
                retain_graph=True,
                create_graph=True,
            )
            norm = tuple(map(torch.norm, grad_weight))
            gw.extend(norm)
        gw = torch.stack(gw)
        gw_avg = gw.mean().detach()

        loss_ratio = losses.detach() / self.init_loss
        rt = loss_ratio / loss_ratio.mean()
        constant = (gw_avg * rt**self.alpha).detach()
        lgrad = torch.abs(gw - constant.repeat_interleave(self.layer_count)).sum()
        self.optimizer.zero_grad()
        lgrad.backward()
        self.optimizer.step()

        # renormalize weights
        self.weights = (self.weights / self.weights.sum() * self.T).detach()
        self.weights = torch.nn.Parameter(self.weights)
        self.optimizer = torch.optim.Adam([self.weights], lr=self.lr0)

        return weighted_loss
