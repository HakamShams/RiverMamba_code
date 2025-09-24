import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class MLP(nn.Module):

    def __init__(self, embed_dim=32, hidden_dim=32, out_dim=1):

        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_dim, bias=True)
        self.act = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):

            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.1)  # 0.04
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        return self.fc2(self.act(self.fc1(x)))


if __name__ == '__main__':

    test_x = torch.randn((2, 200000, 32), device='cuda')

    model = MLP(embed_dim=32, hidden_dim=32, out_dim=1).to('cuda')
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_parameters}")

    test = model(test_x)
    print(test.shape)

