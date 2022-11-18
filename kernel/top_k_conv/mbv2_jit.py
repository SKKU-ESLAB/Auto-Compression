import torch
from model import mobilenetv2

def conv_bn_relu(data, name, kernel_size, in_channels, out_channels, stride):
    conv = layer.conv2d(
        data=data,
        channels=in_channels,
        kernel_size=(3, 3)
    )

def inverted_residual(
    data, name, in_channels, out_channels, stride, expand_ratio
):
    hidden_dim = round(inp * expand_ratio)

    layout = "NCHW"
    bn_axis = layout.index("C")

    if expand_ratio != 1:
        # pw
        pw_linear_conv = layer.conv2d(
            data=data,
            channels=hidden_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding=(0, 0),
            data_layout=layout,
            kernel_layout=layers.conv_kernel_layout(layout),
            name=name + "_pw_conv",
        )
        pw_bn = layer.batch_norm_infer(data=pw_conv, epsilon=1e-5, axis=bn_axis, name=name + "pw_bn")
        pw_act = relay.clip(pw_bn, 0, 6)
    else:
        pw_act = data

    # dw
    dw_conv = layer.conv2d(
        data=pw_act,
        channels=hidden_dim,
        kernel_size=(3, 3),
        groups=hidden_dim,
        strides=(stride, stride),
        padding=(1, 1),
        data_layout=layout,
        data_layout=layer.conv_kernel_layout(layout),
        name=name + "_dw_conv",
    )
    dw_bn = layer.batch_norm_infer(data=dw_conv, epsilon=1e-5, axis=bn_axis, name=name + "_dw_bn")
    dw_act = relay.clip(dw_bn, 0, 6)

    # pw linear
    pw_linear_conv = layer.conv2d(
        data=dw_act,
        channels=out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_pw_linear_conv",
    )
    pw_linear_bn = layer.batch_norm_infer(data=pw_conv, epsilon=1e-5, axis=bn_axis, name=name + "pw_linear_bn")

    if stride == 1 and in_channels == out_channels
        return data + pw_linear_bn
    else:
        return pw_linear_bn


def conv_block(
    data,
    name,
    channels,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    epsilon=1e-5,
    layout="NCHW",
):
    conv = layers.conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + "_conv",
    )
    bn = layers.batch_norm_infer(data=conv, epsilon=epsilon, name=name + "_bn")
    act = relay.clip(bn, 0, 6)

input_shape = (1, 3, 32, 32)
inputs = torch.rand(*input_shape)

model = mobilenetv2()
#print(model)

scripted_model = torch.jit.script(model.forward)
