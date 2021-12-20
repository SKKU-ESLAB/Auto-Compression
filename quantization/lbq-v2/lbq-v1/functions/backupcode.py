
'''
            # original:  c, d, gamma

            #print(self.weight.shape)
            #print(self.dw.cuda() - self.mean.cuda())
            w_mask1 = (w_abs < self.cw -_dw).type(torch.float).detach()
            w_mask2 = (w_abs > self.cw + _dw).type(torch.float).detach()
            w_mask3 = (torch.ones_like(w_abs) - w_mask1 - w_mask2).detach()

            w_cal = (0.5/(_dw)) * (w_abs-self.cw+_dw).abs()
            nan_detect(w_cal)
            torch.set_printoptions(profile="full")
            #print(w_cal)
            print(input.min().item(), input.max().item())
            #print(self.cx+self.dx)
            #print(_dw[:10])
            #print(self.gamma)
            torch.set_printoptions(profile="default")
            nan_detect((w_cal).pow(self.gamma))
            w_hat = ( w_mask2 * w_sign ) + ( w_mask3 * ( w_cal ).pow(self.gamma) * w_sign )
            nan_detect(w_hat)
            w_bar = Round.apply(w_hat * self.qw) / self.qw
            nan_detect(w_bar)
            w_bar = w_bar.view(self.w_shape)
            nan_detect(w_bar)

            x_mask1 = (input < self.cx - _dx).type(torch.float).detach()
            x_mask2 = (input > self.cx + _dx).type(torch.float).detach()
            x_mask3 = (torch.ones_like(input) - x_mask1 - x_mask2).detach()
            x_cal = (0.5/_dx) * (input-self.cx+_dx)
            nan_detect(x_cal)
            x_hat = x_mask2 + x_mask3 * x_cal
            nan_detect(x_hat)
            x_bar = Round.apply(x_hat * self.qx) / self.qx
            nan_detect(x_bar)
'''


class lq_conv2d_v1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 is_qt=False, tr_gamma=True, lq=False, block_num=-1, layer_num=-1, index=[]):
        super(lq_conv2d_v1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)
        self.block_num = block_num
        self.layer_num = layer_num
        self.index = index
        self.w_shape = self.weight.shape

        self.is_qt = is_qt
        if lq:
            self.p_w_x1_r2 = Parameter(torch.Tensor([-6.91]))
            self.p_w_x2_r2 = Parameter(torch.Tensor([-2.31]))
            #self.p_w_x1_r2 = Parameter(torch.ones([out_channels, 1])*-6.91)
            #self.p_w_x2_r2 = Parameter(torch.ones([out_channels, 1])*-2.31)

            self.p_a_x1_r2 = Parameter(torch.Tensor([-2.302]))
            self.p_a_x2_r2 = Parameter(torch.Tensor([-0.105]))

            self.gamma_r3 = Parameter(torch.Tensor([1]))
            #self.gamma_r3 = Parameter(torch.ones([out_channels, 1]))
            self.tr_gamma = tr_gamma

    def set_bit_width(self, w_bit, x_bit):
        self.qx = 2**(x_bit) - 1
        self.qw = torch.ones((self.w_shape[0], 1))
        bit_max = 8
        for i in range(bit_max):
            if len(self.index[self.block_num][self.layer_num][i]) == 0:
                continue
            else:
                idx = self.index[self.block_num][self.layer_num][i]
                self.qw[idx] = 2**(i+1)-1
        #print(self.qw)

    def forward(self, input):
        if self.is_qt:
            w_abs = self.weight.abs()
            w_sign = self.weight.sign()

            w_abs = w_abs.view(self.w_shape[0], -1)
            w_sign = w_sign.view(self.w_shape[0], -1)
            ''''
            w_x1 = torch.zeros((w_shape[0], 1)).cuda()
            w_x2 = torch.zeros((w_shape[0], 1)).cuda()
            w_r = torch.zeros((w_shape[0], 1)).cuda()
            w_mask1 = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_mask2 = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_mask3 = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_cal = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_hat = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_bar = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            w_ste = torch.zeros((w_shape[0], w_abs.shape[1])).cuda()
            '''
            '''
            w_x1[idx] = torch.exp(self.p_w_x1_r2[idx])
            w_x2[idx] = torch.exp(self.p_w_x2_r2[idx])
            a_x1 = torch.exp(self.p_a_x1_r2)
            a_x2 = torch.exp(self.p_a_x2_r2)

            w_r[idx] = (self.gamma_r3[idx]) if self.tr_gamma else (self.gamma_r3.detach()[idx])

            w_mask1[idx] = (w_abs[idx] <= w_x1[idx]).type(torch.float).detach()
            w_mask3[idx] = (w_abs[idx] >  w_x1[idx] + 2*w_x2[idx]).type(torch.float).detach()
            w_mask2[idx] = (torch.ones_like(w_abs[idx]) - w_mask1[idx] - w_mask3[idx]).detach()

            w_cal[idx] = w_mask2[idx] * ((0.5/w_x2[idx]) * (w_abs[idx] - w_x1[idx])) + 1e-10
            w_hat[idx] = (w_mask2[idx] * w_cal[idx].pow(w_r[idx]) + w_mask3[idx])
            w_bar[idx] = (torch.round(w_hat[idx] * (self.qw))) / (self.qw)
            w_ste[idx] = ((w_bar[idx].pow(1/w_r[idx]) - w_hat[idx]).detach() + w_hat[idx]) * w_sign[idx]
            '''
            w_x1 = torch.exp(self.p_w_x1_r2)
            w_x2 = torch.exp(self.p_w_x2_r2)
            a_x1 = torch.exp(self.p_a_x1_r2)
            a_x2 = torch.exp(self.p_a_x2_r2)

            w_r = (self.gamma_r3) if self.tr_gamma else (self.gamma_r3.detach())

            w_mask1 = (w_abs <= w_x1).type(torch.float).detach()
            w_mask3 = (w_abs >  w_x1 + 2*w_x2).type(torch.float).detach()
            w_mask2 = (torch.ones_like(w_abs) - w_mask1 - w_mask3).detach()

            w_cal = w_mask2 * ((0.5/w_x2) * (w_abs - w_x1)) + 1e-10
            w_hat = (w_mask2 * w_cal.pow(w_r) + w_mask3)
            w_bar = (torch.round(w_hat * (self.qw.cuda()))) / self.qw.cuda()
            w_ste = ((w_bar.pow(1/w_r) - w_hat).detach() + w_hat) * w_sign
            w_ste = w_ste.view(self.w_shape)
            #exit()

            x_mask1 = (input <= a_x1).type(torch.float).detach()
            x_mask3 = (input >  a_x1 + 2*a_x2).type(torch.float).detach()
            x_mask2 = (torch.ones_like(input) - x_mask1 - x_mask3).detach()

            x_hat = x_mask2 * (0.5/a_x2) * (input - a_x1) + x_mask1
            x_bar = torch.round(x_hat * self.qx)/ self.qx
            x_ste = (x_bar - x_hat).detach() + x_hat

            y = F.conv2d(x_ste, w_ste, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return y

class lq_conv2d_v2(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 is_qt=False, tr_gamma=True, lq=False):
        super(lq_conv2d_v1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)

        self.is_qt = is_qt
        if lq:
            self.p_w_x_r2 = Parameter(torch.Tensor([-1.609]))

            self.p_a_x_r2 = Parameter(torch.Tensor([0.693]))
            self.gamma_r3 = Parameter(torch.Tensor([1]))
            self.tr_gamma = tr_gamma

    def set_bit_width(self, w_bit, x_bit):
        if self.is_qt:
            self.qw = 2**(w_bit-1) - 1
            self.qx = 2**(x_bit) - 1

    def forward(self, input):
        if self.is_qt:
            w_abs = self.weight.abs()
            w_sign = self.weight.sign()
            w_x = torch.exp(self.p_w_x_r2)
            a_x = torch.exp(self.p_a_x_r2)

            w_r = (self.gamma_r3) if self.tr_gamma else (self.gamma_r3.detach())

            w_mask1 = (w_abs >  w_x).type(torch.float).detach()
            w_mask2 = (torch.ones_like(w_abs) - w_mask1).detach()

            w_cal = w_mask2 * ((1/w_x) * w_abs) + 1e-10
            w_hat = (w_mask2 * w_cal.pow(w_r) + w_mask1)
            w_bar = (torch.round(w_hat * self.qw) / self.qw)
            w_ste = ((w_bar.pow(1/w_r) - w_hat).detach() + w_hat) * w_sign

            x_mask1 = (input >  a_x).type(torch.float).detach()
            x_mask2 = (torch.ones_like(input) - x_mask1).detach()

            x_hat = x_mask2 * (1/a_x) * input + x_mask1
            x_bar = torch.round(x_hat * self.qx) / self.qx
            x_ste = (x_bar - x_hat).detach() + x_hat

            y = F.conv2d(x_ste, w_ste, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return y

class lq_conv2d_v(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 is_qt=False, tr_gamma=True, lq=False):
        super(lq_conv2d_v1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)

        self.is_qt = is_qt
        if lq:
            self.p_w_x1_r2 = Parameter(torch.Tensor([-10]))
            self.p_w_x2_r2 = Parameter(torch.Tensor([-1.609]))

            self.p_a_x1_r2 = Parameter(torch.Tensor([-2.302]))
            self.p_a_x2_r2 = Parameter(torch.Tensor([0.693]))
            self.gamma_r3 = Parameter(torch.Tensor([1]))
            self.tr_gamma = tr_gamma

    def set_bit_width(self, w_bit, x_bit):
        if self.is_qt:
            self.qw = 2**(w_bit-1) - 1
            self.qx = 2**(x_bit) - 1

    def forward(self, input):
        if self.is_qt:
            w_abs = self.weight.abs()
            w_sign = self.weight.sign()
            cw = torch.exp(self.p_w_x1_r2).detach()
            dw = torch.exp(self.p_w_x2_r2)
            cx = torch.exp(self.p_a_x1_r2)
            dx = torch.exp(self.p_a_x2_r2)

            w_r = (self.gamma_r3) if self.tr_gamma else (self.gamma_r3.detach())

            w_mask1 = (w_abs <= cw).type(torch.float).detach()
            w_mask3 = (w_abs >  cw + dw).type(torch.float).detach()
            w_mask2 = (torch.ones_like(w_abs) - w_mask1 - w_mask3).detach()
            w_th1 = (w_abs <= cw + dw*((0.5/self.qw)**(1/w_r))).type(torch.float).detach()
            w_th2 = (w_abs > cw + dw*((1-0.5/self.qw)**(1/w_r))).type(torch.float).detach()
            w_els = (torch.ones_like(w_abs) - w_th1 - w_th2).detach()

            w_cal = w_mask2 * ((1/dw) * (w_abs - cw)) + 1e-10
            w_hat = (w_mask2 * w_cal.pow(w_r) + w_mask3)
            w_bar = (torch.round(w_hat * self.qw) / self.qw) + 1e-10
            w_bar_ = (w_els * (dw*w_bar.pow(1/w_r)+cw) + w_th2*( cw+dw*((1-0.5/self.qw)**(1/w_r)) ) )
            w_ste = ( (w_bar_ - w_hat).detach() + w_hat ) * w_sign

            x_mask1 = (input <= cx).type(torch.float).detach()
            x_mask3 = (input >  cx + dx).type(torch.float).detach()
            x_mask2 = (torch.ones_like(input) - x_mask1 - x_mask3).detach()
            x_th1 = (input <= cx + dx*0.5/self.qx).type(torch.float).detach()
            x_th2 = (input > cx + dx - dx*0.5/self.qx).type(torch.float).detach()
            x_els = (torch.ones_like(input) - x_th1 - x_th2).detach()

            x_hat = x_mask2 * (1/dx) * (input - cx) + x_mask3
            x_bar = torch.round(x_hat * self.qx) / self.qx
            x_bar_ = (x_els*(dx*x_bar+cx) + x_th2*(cx+dx-dx*0.5/self.qx))
            x_ste = (x_bar_ - x_hat).detach() + x_hat

            y = F.conv2d(x_ste, w_ste, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return y

class lq_conv2d_v3(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros',
                 is_qt=False, tr_gamma=True, lq=False):
        super(lq_conv2d_v1, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias, padding_mode)

        self.is_qt = is_qt
        if lq:
            self.p_w_x1_r2 = Parameter(torch.Tensor([-6.91]))
            self.p_w_x2_r2 = Parameter(torch.Tensor([-2.31]))
            self.p_w_x3_r2 = Parameter(torch.Tensor([-2.31]))
            self.p_w_y_r2 = Parameter(torch.Tensor([-0.693]))

            self.p_a_x1_r2 = Parameter(torch.Tensor([-2.302]))
            self.p_a_x2_r2 = Parameter(torch.Tensor([-0.105]))
            self.gamma1_r3 = Parameter(torch.Tensor([1]))
            self.gamma2_r3 = Parameter(torch.Tensor([1]))
            self.tr_gamma = tr_gamma

    def set_bit_width(self, w_bit, x_bit):
        if self.is_qt:
            self.qw = 2**(w_bit-1) - 1
            self.qx = 2**(x_bit) - 1

    def forward(self, input):
        if self.is_qt:
            w_abs = self.weight.abs()
            w_sign = self.weight.sign()
            w_x1 = torch.exp(self.p_w_x1_r2)
            w_x2 = torch.exp(self.p_w_x2_r2)
            w_x3 = torch.exp(self.p_w_x3_r2)
            w_y = torch.exp(self.p_w_y_r2)
            a_x1 = torch.exp(self.p_a_x1_r2)
            a_x2 = torch.exp(self.p_a_x2_r2)
            w_r1 = self.gamma1_r3 if self.tr_gamma else self.gamma1_r3.detach()
            w_r2 = self.gamma2_r3 if self.tr_gamma else self.gamma2_r3.detach()

            w_mask1 = (w_abs <= w_x1).type(torch.float).detach()
            w_mask2 = ((w_x1 < w_abs) & (w_abs <= w_x1 + w_x2)).type(torch.float).detach()
            w_mask3 = ((w_x1 + w_x2 < w_abs) & (w_abs <= w_x1 + w_x2 + w_x3)).type(torch.float).detach()
            w_mask4 = (torch.ones_like(w_abs) - w_mask1 - w_mask2 - w_mask3)
            w_x = w_x1+w_x2+((torch.round(w_y*self.qw)+0.5)/self.qw - w_y).pow(1/w_r2) if (torch.round(w_y*self.qw)/self.qw) <= w_y else w_x1+((torch.round(w_y * self.qw)-0.5)/self.qw).pow(1/w_r1)
            w_invm1 = (w_abs <= w_x).type(torch.float).detach()
            w_invm2 = (w_abs > w_x).type(torch.float).detach()

            w_hat = (w_mask1 * 0
                   + w_mask2 * (w_y)*(w_mask2*(1/w_x2)*(w_abs-w_x1) + 1e-10).pow(w_r1)
                   + w_mask3 * ((1-w_y)*(w_mask3*(1/w_x3)*(w_abs-w_x1-w_x2) + 1e-10).pow(w_r2) + w_y)
                   + w_mask4 * 1)
            w_bar = (torch.round(w_hat * self.qw) / self.qw)
            w_ste =((w_invm1 * w_y * (w_bar/w_y).pow(1/w_r1)
                   + w_invm2 * ((1-w_y) * (w_invm2*(w_bar-w_y)/(1-w_y)).pow(1/w_r2) + w_y)
                   - w_hat).detach() + w_hat) * w_sign

            x_mask1 = (input <= a_x1).type(torch.float).detach()
            x_mask2 = (input >  a_x1+a_x2).type(torch.float).detach()
            x_mask3 = (torch.ones_like(input) - x_mask1 - x_mask2).detach()

            x_hat = x_mask2 + x_mask3 * (0.5/a_x2) * (input - a_x1)
            x_bar = torch.round(x_hat * self.qx) / self.qx
            x_ste = (x_bar - x_hat).detach() + x_hat

            y = F.conv2d(x_ste, w_ste, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            y = F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        return y

