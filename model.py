"""
Physics-Informed Neural Network (PINN) ansatz for the 2D Poisson equation

        -Delta u(x, y) = f(x, y)   on   Omega = (0, 1)^2
             u(x, y)   = g(x, y)   on   partial Omega.

The network takes the two spatial coordinates (x, y) and returns a single
real-valued scalar u. All partial derivatives that enter the PDE loss
(u_xx, u_yy) are obtained by PyTorch automatic differentiation in the
notebook; this module only defines the trainable ansatz.
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """Fully-connected feed-forward network used as the PINN ansatz.

    Default architecture (a common Poisson-2D baseline):

        Input   : 2 neurons     (x, y)
        Hidden  : 4 layers x 50 neurons, tanh activation
        Output  : 1 neuron      (u)

    Design notes
    ------------
    * tanh is used throughout because the physics loss requires the
      network to be at least C^2 in its inputs: u_xx and u_yy are
      obtained through second-order automatic differentiation. ReLU-type
      activations have vanishing second derivatives almost everywhere and
      would make the residual identically zero on the interior.
    * Xavier/Glorot normal initialization keeps the variance of the pre-
      activations roughly constant across the tanh stack at init and
      empirically speeds up PINN convergence.
    * The network is purposely small: the target solution for this
      benchmark, u(x, y) = sin(pi x) sin(pi y), is smooth and
      low-frequency, so a 4 x 50 MLP has more than enough capacity.
      If you plan to attack sharper or multiscale sources, widen
      `layers` via the constructor argument rather than editing this
      module.
    """

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [2] + [50] * 4 + [1]

        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.activation = nn.Tanh()

        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, y):
        """Evaluate the network at collocation points.

        Parameters
        ----------
        x : torch.Tensor of shape (N, 1)
            First spatial coordinate.
        y : torch.Tensor of shape (N, 1)
            Second spatial coordinate.

        Returns
        -------
        u : torch.Tensor of shape (N, 1)
            Predicted scalar field u(x, y).
        """
        z = torch.cat([x, y], dim=1)
        for lin in self.linears[:-1]:
            z = self.activation(lin(z))
        return self.linears[-1](z)
