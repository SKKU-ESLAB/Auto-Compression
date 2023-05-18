import torch
import einops
import torch.nn.functional as F

__all__ = ['ZeropatchPad2d', 'ReplicationPad2d', 'ReflectionPad2d', 'Patch2Feature', 'Mean_3px_Pad2d', 'Mean_2px_Pad2d']
# definition for patch with custom padding
# input : patches
# output : patches

class PaddingModule(torch.nn.Module):
    def __init__(self, padding, num_patches):
        super(PaddingModule, self).__init__()
        self.padding = padding
        self.torch_pad2d = (padding, padding, padding, padding)
        self.num_patches = num_patches
        self.loss = 0.0 
    def forward(self, x):
        pass

    def get_error(self, padded_x):
        self.loss = 0.
        P = self.num_patches
        b, _, patch_h, patch_w = padded_x.size()
        B = b// (P** 2)
        patch_h -= 2; patch_w -= 2
        vertical_patch_idx = [torch.arange(i * P, (i + 1) * P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P) for i in range(P)]
        horizontal_patch_idx = [torch.arange(i, b, P) for i in range(P)]
        
        prediction_top_border = []
        target_top_border = []
        
        for i, line_idx in enumerate(vertical_patch_idx[1:]):
            prediction_top_border.append(padded_x[line_idx, :, 0, 1:patch_w + 1])
        for i, line_idx in enumerate(vertical_patch_idx[:P-1]):
            target_top_border.append(padded_x[line_idx, :, patch_h, 1:patch_w + 1])
        
        prediction_bot_border = []
        target_bot_border = []
        
        for i, line_idx in enumerate(vertical_patch_idx[:P-1]):
            prediction_bot_border.append(padded_x[line_idx, :, patch_h + 1, 1:patch_w + 1])
        for i, line_idx in enumerate(vertical_patch_idx[1:]):
            target_bot_border.append(padded_x[line_idx, :, 1, 1:patch_w + 1])
            
        prediction_left_border = []
        target_left_border = []
        
        for i, line_idx in enumerate(horizontal_patch_idx[1:]):
            prediction_left_border.append(padded_x[line_idx, :, 1:patch_h + 1, 0])
        for i, line_idx in enumerate(horizontal_patch_idx[:P-1]):
            target_left_border.append(padded_x[line_idx, :, 1:patch_h + 1, patch_w])
            
        prediction_right_border = []
        target_right_border = []
        
        for i, line_idx in enumerate(horizontal_patch_idx[:P-1]):
            prediction_right_border.append(padded_x[line_idx, :, 1:patch_h + 1, patch_w + 1])
        for i, line_idx in enumerate(horizontal_patch_idx[1:]):
            target_right_border.append(padded_x[line_idx, :, 1:patch_h + 1, 1])
            
        prediction_top_border = torch.cat(prediction_top_border, dim=0)
        prediction_bot_border = torch.cat(prediction_bot_border, dim=0)
        prediction_left_border = torch.cat(prediction_left_border, dim=0)
        prediction_right_border = torch.cat(prediction_right_border, dim=0)
            
        target_top_border = torch.cat(target_top_border, dim=0)
        target_bot_border = torch.cat(target_bot_border, dim=0)
        target_left_border = torch.cat(target_left_border, dim=0)
        target_right_border = torch.cat(target_right_border, dim=0)
        
        pad_top_err = F.mse_loss(prediction_top_border, target_top_border)
        pad_bot_err = F.mse_loss(prediction_bot_border, target_bot_border)
        pad_left_err = F.mse_loss(prediction_left_border, target_left_border)
        pad_right_err = F.mse_loss(prediction_right_border, target_right_border)
        
        self.loss = (pad_top_err + pad_bot_err + pad_left_err + pad_right_err)/4
        
# post processing
class Patch2Feature(torch.nn.Module):
    def __init__(self, num_patches):
        super(Patch2Feature, self).__init__()
        self.num_patches = num_patches

    def forward(self, x):
        return einops.rearrange(x, '(B p1 p2) C H W -> B C (p1 H) (p2 W)', p1=self.num_patches, p2=self.num_patches)

# for patch per patch layer
# otherwise use pytorch ZeropPad2d
class ZeropatchPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='constant', value=0.0)
        
        P = self.num_patches

        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0

        return x

class ReplicationPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='replicate')

        P = self.num_patches

        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0
        self.get_error(x)

        return x

class Mean_3px_Pad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)
        
        mean_kernel = torch.ones(size=(C, 1, 1, 3), device=x.device)
        mean_kernel /= 3
        
        top_pad = F.conv2d(F.pad(x[:, :, :self.padding, :], (0, 2, 0, 0), mode='constant', value=0.0), mean_kernel, stride=1, groups=C, padding=0)
        bot_pad = F.conv2d(F.pad(x[:, :, H-self.padding:, :], (0, 2, 0, 0), mode='constant', value=0.0), mean_kernel, stride=1, groups=C, padding=0)
        left_pad = F.conv2d(x[:, :, :, :3], mean_kernel, stride=1, groups=C, padding=0)
        right_pad = F.conv2d(x[:, :, :, W-3:], mean_kernel, stride=1, groups=C, padding=0)
        
        x = F.pad(x, self.torch_pad2d, mode='replicate')
        
        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0

        return x

class Mean_2px_Pad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        b, C, H, W = x.size()
        P = self.num_patches
        B = b // (P ** 2)
        
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        top_bot_kernel = torch.ones(size=(C, 1, 2, 1), device=x.device)
        left_right_kernel = torch.ones(size=(C, 1, 1, 2), device=x.device)

        top_bot_kernel /= 2
        left_right_kernel /= 2

        top_pad = F.conv2d(x[:, :, :2, :], top_bot_kernel, stride=1, groups=C)
        bot_pad = F.conv2d(x[:, :, H-2:, :], top_bot_kernel, stride=1, groups=C)
        left_pad = F.conv2d(x[:, :, :, :2], left_right_kernel, stride=1, groups=C)
        right_pad = F.conv2d(x[:, :, :, W-2:], left_right_kernel, stride=1, groups=C)

        x = F.pad(x, self.torch_pad2d, mode='replicate')
        
        x[:, :, :self.padding, self.padding:W+self.padding] = top_pad
        x[:, :, H+self.padding:, self.padding:W + self.padding] = bot_pad
        x[:, :, self.padding:H + self.padding, :self.padding] = left_pad
        x[:, :, self.padding:H + self.padding, W + self.padding:] = right_pad

        x[top, :, :self.padding, :] = 0
        x[bot, :, H+self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, W+self.padding:] = 0

        return x


class ReflectionPad2d(PaddingModule):
    def __init__(self, padding, num_patches):
        super().__init__(padding, num_patches)
    
    def forward(self, x):
        # pad patch with custom padding
        x = F.pad(x, self.torch_pad2d, mode='reflect')
        
        P = self.num_patches
        # remove for border of feature map
        b, C, pad_patch_h, pad_patch_w = x.size()
        B = b // (P ** 2)
        top = torch.arange(0, P, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        bot = torch.arange(P ** 2 - P, P ** 2, 1).repeat(B) + torch.arange(0, b, P ** 2).repeat_interleave(P)
        left = torch.arange(0, b, P)
        right = torch.arange(P-1, b, P)

        x[top, :, :self.padding, :] = 0
        x[bot, :, pad_patch_h-self.padding, :] = 0
        x[left, :, :, :self.padding] = 0
        x[right, :, :, pad_patch_w-self.padding:] = 0
        self.get_error(x)
        return x










