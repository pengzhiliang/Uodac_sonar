import torch 
import torch.nn.functional as F 
from torch import nn

class DropBlock2D(nn.Module):

    def __init__(self, drop_prob, block_size, steps=100):
        super(DropBlock2D, self).__init__()
        self.drop_prob = 0.
        self.block_size = block_size
        self.i = 0
        self.drop_values = torch.linspace(0., drop_prob, steps=steps)

    def forward(self, x):
        # shape: (bsize, channels, height, width)
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place mask on input device
            mask = (torch.rand(*x.shape) < gamma).float().to(x.device)
            
            mask[:, :, :self.block_size // 2, :] = 0
            mask[:, :, mask.shape[2] - self.block_size // 2:, :] = 0
            mask[:, :, :, :self.block_size // 2] = 0
            mask[:, :, :, mask.shape[3] - self.block_size // 2:] = 0

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            self.step()
            return out

    def _compute_block_mask(self, mask):
        
        block_mask = F.max_pool2d(input=mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask

        return block_mask

    def _compute_gamma(self, x):
        height, width = x.shape[2:]
        return self.drop_prob * height * width / ((self.block_size ** 2) * (height - self.block_size + 1) 
                                                                 * (width - self.block_size + 1))
    
    def step(self):
        if self.i < len(self.drop_values):
            self.drop_prob = self.drop_values[self.i]

        self.i += 1