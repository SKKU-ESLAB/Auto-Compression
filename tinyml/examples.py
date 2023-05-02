import numpy as np
import math
from keras.applications import imagenet_utils
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *

# define functional model
layer_params = []
residual_block = []

# pre-trained setting
K.set_image_data_format('channels_last')
img_width, img_height = 224, 224
alpha=1.0
central_frac = 0.875

log_file = open('./log_patch_result.txt', 'w')

# toy class
class Toy(tf.keras.Model):
    def __init__(self):
        super(Toy, self).__init__()
        self.conv1 = Conv2D(3, (3, 3), padding='same', strides=(1,1), use_bias=False)
        self.relu1 = ReLU()
        self.zero2 = ZeroPadding2D(padding=((0, 1), (0, 1)))
        self.conv2 = Conv2D(3, (3, 3), padding='valid',strides=(2,2), use_bias=False)
        self.relu2 = ReLU()
        self.conv3 = Conv2D(3, (3, 3), padding='same', strides=(1, 1), use_bias=False)
        self.relu3 = ReLU()
        self.conv4 = Conv2D(3, (3, 3), padding='same', strides=(1,1), use_bias=False)
        self.relu4 = ReLU()
        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(10,activation='softmax', use_bias=True)

    def call(self, x):
        global log_file, skip
        if not skip:
            log_file.write(f"normal_inference input\n{x}\n")
        x = self.conv1(x)
        if not skip:
            log_file.write(f"normal_inference conv1\n{x}\n")
        x = self.relu1(x)
        if not skip:
            log_file.write(f"normal_inference relu1\n{x}\n")
        x = self.zero2(x)
        if not skip:
            log_file.write(f"normal_inference zero2\n{x}\n")
        x = self.conv2(x)
        if not skip:
            log_file.write(f"normal_inference conv2\n{x}\n")
        x = self.relu2(x)
        if not skip:
            log_file.write(f"normal_inference relu2\n{x}\n")
        x = self.conv3(x)
        if not skip:
            log_file.write(f"normal_inference conv3\n{x}\n")
        x = self.relu3(x)
        if not skip:
            log_file.write(f"normal_inference relu3\n{x}\n")
        x = self.conv4(x)
        if not skip:
            log_file.write(f"normal_inference conv4\n{x}\n")
        x = self.relu4(x)
        if not skip:
            log_file.write(f"normal_inference relu4\n{x}\n")
        x = self.gap(x)
        if not skip:
            log_file.write(f"normal_inference gap\n{x}\n")
        x = self.fc(x)
        if not skip:
            log_file.write(f"normal_inference fc\n{x}\n")
        return x

# patch per patch setting

def preprocess(img, label):
    img = preprocess_input(img, data_format='channels_last')
    img = crop_layer(img)
    label = tf.one_hot(label, depth=1000)
    return img, label

crop_layer = CenterCrop(img_height, img_width)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        '/data/imagenet/val',
        image_size=(math.ceil(img_height/central_frac), math.ceil(img_width/central_frac)),
        batch_size=500)

normalized_val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

input_shape=(img_width, img_height, 3)

model = MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=True,
        )
#model = Toy()
#skip = True
#model(np.random.randn(1, 8, 8, 3).astype(np.float32))
model.trainable = False
layers = model.layers
for layer in layers:
    p = layer.get_config()
    t = type(layer)
    #print(f"{t} --> {layer.get_config()}")
    if t == Conv2D:
        name = p['name']
        weight, bias = None, None
        if len(layer.get_weights()) == 2:
            weight, bias = layer.get_weights()
        else:
            weight = layer.get_weights()[0]
        strides, padding = p['strides'], p['padding']
        use_bias = p['use_bias']

        temp = {
                'op' : t,
                'name': name,
                'weight': weight,
                'bias' : bias,
                'use_bias': use_bias,
                'strides' : strides,
                'padding' : padding,
                }
        layer_params.append(temp)
    elif t == DepthwiseConv2D:
        name = p['name']
        weight, bias = None, None
        if len(layer.get_weights()) == 2:
            weight, bias = layer.get_weights()
        else:
            weight = layer.get_weights()[0]
        strides, padding = p['strides'], p['padding']
        use_bias = p['use_bias']
        temp = {
                'op' : t,
                'name': name,
                'weight': weight,
                'bias' : bias,
                'use_bias': use_bias,
                'strides' : strides,
                'padding' : padding,
                }
        layer_params.append(temp)
    elif t == ReLU:
        name = p['name']
        max_value = p['max_value']
        temp = {
                'op' : t,
                'name': name,
                'max_value': max_value
               }
        layer_params.append(temp)
    elif t == BatchNormalization:
        name = p['name']
        if len(layer.weights) == 4:
            scale, offset, mean, variance= layer.get_weights()
        else:
            raise NotImplementedError(f"BN len = {len(layer.weights)}")
        variance_epsilon = p['epsilon']
        temp = {
                'op' : t,
                'name' : name,
                'offset' : offset,
                'scale' : scale,
                'mean': mean,
                'variance':  variance,
                'variance_epsilon' : variance_epsilon
               }
        layer_params.append(temp)
    elif t == ZeroPadding2D:
        continue
    elif t == GlobalAveragePooling2D:
        name = p['name']
        keepdims = p['keepdims']
        temp = {
                'op' : t,
                'name' : name,
                'keepdims': keepdims
               }
        layer_params.append(temp)
    elif t == Dense:
        name = p['name']
        weight, bias = None, None
        if len(layer.get_weights()) == 2:
            weight, bias = layer.get_weights()
        else:
            weight = layer.get_weights()[0]
        use_bias = p['use_bias']
        activation = p['activation']
        temp = {
                'op' : t,
                'name': name,
                'activation': activation,
                'weight' : weight,
                'bias' : bias,
                'use_bias': use_bias,
               }
        layer_params.append(temp)
    elif t == Add:
        name = p['name']
        block_id = name.split('_')[:2]
        block_id = '_'.join(block_id)
        residual_block.append(block_id)
        temp = {
                'op' : t,
                'name' : name
               }
        layer_params.append(temp)
    elif t == InputLayer:
        continue
    else:
        raise NotImplementedError(f"Invalid {layer.name}, {t}")
    #print(f"{t} \n{layer.get_config()}")


#print(len(layers), len(layer_params))

layers = [layer for layer in layers if not isinstance(layer, (ZeroPadding2D, InputLayer))]
assert len(layers) == len(layer_params), 'Copy Error!'
#print(len(layers), len(layer_params))

# check neccesery DAG
#for orig, to in zip(layers, layer_params):
#    assert type(orig) == to['op'], 'error'

for layer in layer_params:
    name = layer['name']
    for item in residual_block:
        if item in name:
            #print(name)
            layer['use_residual'] = True
            residual_block.pop(0)
            break

#for layer in layer_params:
#    if 'use_residual' in layer:
#        print(layer['name'])


def get_pad(input_dim, stride, kernel_size, pad_str):
    pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
    if pad_str == 'same':
        if input_dim % stride == 0:
            pad_along_dim = max(kernel_size - stride, 0)
        else:
            pad_along_dim = max(kernel_size - (input_dim % stride), 0)

        pad_t = pad_along_dim // 2
        pad_b = pad_along_dim - pad_t
        pad_l = pad_along_dim // 2
        pad_r = pad_along_dim - pad_l
    elif pad_str == 'valid':
        if stride == 2:
            pad_t = kernel_size // 2 - 1 + input_dim % 2
            pad_b = kernel_size // 2
            pad_l = kernel_size // 2 - 1 + input_dim % 2
            pad_r = kernel_size // 2
    else:
        raise NotImplementedError(f"not implement for {pad_str}")
    return ((pad_t, pad_b), (pad_l, pad_r))

def conv2d(input, p, pad_t=0, pad_b=0, pad_l=0, pad_r=0):
    B, H, W, C = input.shape
    kernel, strides, pad_str = p['weight'], p['strides'], p['padding']
    k, s = kernel.shape[0], strides[0]
    pad_tuple_2d = get_pad(H, s, k, pad_str)
    #print('PW --> ',pad_tuple_2d)
    bias, use_bias = p['bias'], p['use_bias']

    if 'is_patch_layer' not in p:
        padded_input = K.spatial_2d_padding(input, padding=pad_tuple_2d)
        out = K.conv2d(padded_input, kernel, strides=strides, padding='valid')
        if use_bias:
            out = tf.nn.bias_add(out, bias)
        return out
    else:
        padded_input = np.zeros((B, H, W, C)).astype(np.float32)
        #print(input[0, pad_t:H-pad_b, pad_l:W-pad_r, 0])
        padded_input[:, pad_t:H-pad_b, pad_l:W-pad_r, :] = input[:, pad_t:H-pad_b, pad_l:W-pad_r, :]
        #print(f'check point: pad_t={pad_t} / pad_b={pad_b} /pad_l={pad_l}/pad_r={pad_r}')
        #print('padded_input\n',padded_input, '\ninput\n',input)
        out = K.conv2d(padded_input, kernel, strides=strides, padding='valid')
        #print(f"CONV_2D {input.shape} --> {out.shape}")

        if s != 1:
            pad_t = pad_t // s
            pad_b = pad_b // s
            pad_l = pad_l // s
            pad_r = pad_r // s
        else:
            if k == 3:
                pad_t = max(0, pad_t - 1)
                pad_b = max(0, pad_b - 1)
                pad_l = max(0, pad_l - 1)
                pad_r = max(0, pad_r - 1)

        B, H, W, C = out.shape
        padded_input = np.zeros((B, H, W, C)).astype(np.float32)
        padded_input[:,pad_t:H-pad_b, pad_l:W-pad_r, :] = out[:, :H-pad_b-pad_t, :W-pad_l-pad_r, :]
        out = padded_input
        if use_bias:
            out = tf.nn.bias_add(out, bias)
        return out, pad_t, pad_b, pad_l, pad_r


def depthwise_conv2d(input, p, pad_t=0, pad_b=0, pad_l=0, pad_r=0):
    B, H, W, C = input.shape
    kernel, strides, pad_str = p['weight'], p['strides'], p['padding']
    k, s = kernel.shape[0], strides[0]
    pad_tuple_2d = get_pad(H, s, k, pad_str)
    #print(f'stride = {s} DW -->',pad_tuple_2d)
    bias, use_bias = p['bias'], p['use_bias']

    if 'is_patch_layer' not in p:
        padded_input = K.spatial_2d_padding(input, padding=pad_tuple_2d)
        out = K.depthwise_conv2d(padded_input, kernel, strides=strides, padding='valid')
        if use_bias:
            assert 1 == 0, 'No bias'
            out = tf.nn.bias_add(out, bias)
        return out
    else:
        padded_input = np.zeros((B, H, W, C)).astype(np.float32)
        #print(input[0, pad_t:H-pad_b, pad_l:W-pad_r, 0])
        padded_input[:, pad_t:H-pad_b, pad_l:W-pad_r, :] = input[:, pad_t:H-pad_b, pad_l:W-pad_r, :]
        out = K.conv2d(padded_input, kernel, strides=strides, padding='valid')
        #print(f"DW_2D {input.shape} --> {out.shape}")
        if s != 1:
            pad_t = pad_t // s
            pad_b = pad_b // s
            pad_l = pad_l // s
            pad_r = pad_r // s
        else:
            if k == 3:
                pad_t = max(0, pad_t - 1)
                pad_b = max(0, pad_b - 1)
                pad_l = max(0, pad_l - 1)
                pad_r = max(0, pad_r - 1)

        B, H, W, C = out.shape
        padded_input = np.zeros((B, H, W, C)).astype(np.float32)
        padded_input[:,pad_t:H-pad_b, pad_l:W-pad_r, :] = out[:, :H-pad_b-pad_t, :W-pad_l-pad_r, :]
        out = padded_input
        if use_bias:
            out = tf.nn.bias_add(out, bias)
        return out, pad_t, pad_b, pad_l, pad_r


def dense(input, p):
    rank = input.shape.rank
    kernel = p['weight']
    bias, use_bias = p['bias'], p['use_bias']
    activation = p['activation']

    #print('adada activation')
    out = tf.tensordot(input, kernel, [[rank - 1], [0]])
    if use_bias:
        out = tf.nn.bias_add(out, bias)
    if activation == 'softmax':
        out = K.softmax(out, axis=-1)
    return out

def relu(input, p):
    max_value = p['max_value']
    return K.relu(input, max_value=max_value)

def batch_norm2d(input, p):
    running_mean, running_var, var_eps = p['mean'], p['variance'], p['variance_epsilon']
    offset, scale = p['offset'], p['scale']
    return tf.nn.batch_normalization(input, running_mean, running_var,offset, scale, var_eps)

def gap2d(input, p):
    return K.mean(input, axis=[1, 2], keepdims=False)

def add(input, p, pad_t=0, pad_b=0, pad_l=0, pad_r=0):
    residual = input[0]
    out = input[1]
    if 'is_patch_layer' in p:
        B, H, W, C = out.shape
        #print(residual)
        #print(f'ADD -> out {out.shape} residual {residual.shape}')
        residual = residual[:, 1:H+1, 1:W+1, :]
    return tf.add(residual, out)

class MBV2_MCU(tf.keras.Model):
    def __init__(self, layer_params):
        super(MBV2_MCU, self).__init__()
        self.layer_params = layer_params
    
    def split_input(self, input, y_idx, x_idx):
        B, H, W, C = input.shape
        patch_pad_t, patch_pad_b, patch_pad_l, patch_pad_r = 0, 0, 0, 0
        if y_idx == 0:
            patch_pad_t = self.pad_t
        elif y_idx == self.n_patches - 1:
            patch_pad_b = self.pad_b
        
        if x_idx == 0:
            patch_pad_l = self.pad_l
        elif x_idx == self.n_patches - 1:
            patch_pad_r = self.pad_r

        length = self.pad_t + self.pad_b + H//self.n_patches

        start_y = max(0, H//self.n_patches * y_idx - self.pad_t)
        start_x = max(0, W//self.n_patches * x_idx - self.pad_l)

        y_fill = length - patch_pad_t - patch_pad_b
        x_fill = length - patch_pad_l - patch_pad_r

        padded_patches = np.zeros((B, length, length, C)).astype(np.float32) 
        padded_patches[:, patch_pad_t:patch_pad_t+y_fill, patch_pad_l:patch_pad_l+x_fill, :] \
        =input[:, start_y:start_y+y_fill, start_x:start_x+x_fill, :]
        return padded_patches, patch_pad_t, patch_pad_b, patch_pad_l, patch_pad_r



    def set_patch_split(self, n_patches, up, down):
        self.n_patches = n_patches
        self.pad_t = up
        self.pad_b = down
        self.pad_l = up
        self.pad_r = down


            
    def get_type(self,op_code):
        if op_code == Conv2D:
            return "CONV_2D"
        elif op_code == DepthwiseConv2D:
            return "DEPTHWISE_CONV_2D"
        elif op_code == Add:
            return "ADD"
        elif op_code == BatchNormalization:
            return "BATCH_NORM_2D"
        elif op_code == Dense:
            return "FC"
        elif op_code == ReLU:
            return "RELU"
        elif op_code == GlobalAveragePooling2D:
            return "GAPool2D"
        else:
            raise NotImplementedError(f"{op_code} is not implemented")

    def set_split_patch(self, n_split):
        num_trans_layer = 0
        for p_layer in self.layer_params:
            if num_trans_layer == n_split:
                break
            p_layer['is_patch_layer'] = True
            t = p_layer['op']
            if t in [Conv2D, DepthwiseConv2D, Add]:
                num_trans_layer += 1

    def normal(self, x):
        residual = None
        for layer_idx, p_layer in enumerate(self.layer_params):
            op = p_layer['op']
            if op == Conv2D:
                if 'use_residual' in p_layer:
                    residual = x
                x = conv2d(x, p_layer)
            elif op == DepthwiseConv2D:
                x = depthwise_conv2d(x, p_layer)
            elif op == BatchNormalization:
                x = batch_norm2d(x, p_layer)
            elif op == ReLU:
                x = relu(x, p_layer)
            elif op == GlobalAveragePooling2D:
                x = gap2d(x, p_layer)
            elif op == Add:
                inputs = [residual, x]
                x = add(inputs, p_layer)
                residual = None
            elif op == Dense:
                x = dense(x, p_layer)
            else:
                raise NotImplementedError(f"{layer_idx} layer {op} is Not implemt")
        return x

    def call(self, x):
        global log_file
        start_layer_per_layer = 0
        to_concat_height = []
        residual = None
        add_pad_t, add_pad_b, add_pad_l, add_pad_r = 0, 0, 0, 0

        for y_idx in range(self.n_patches):
            to_concat_width = []
            for x_idx in range(self.n_patches):
                patch, ppad_t, ppad_b, ppad_l, ppad_r = self.split_input(x, y_idx, x_idx)
                log_file.write(f"{y_idx} / {x_idx} input\n{patch}\n")
                print('start padding  --> ', ppad_t, ppad_b, ppad_l, ppad_r)
                for layer_idx, p_layer in enumerate(self.layer_params):
                    if 'is_patch_layer' not in p_layer:
                        start_layer_per_layer = layer_idx
                        to_concat_width.append(patch)
                        break
                    op = p_layer['op']
                    #print(f"{layer_idx}, OP = {op}")
                    #print(f"top = {ppad_t} / bot = {ppad_b} / left = {ppad_l} / right = {ppad_r}")
                    if op == Conv2D:
                        if 'use_residual' in p_layer:
                            residual = patch
                            add_pad_t = ppad_t
                            add_pad_b = ppad_b
                            add_pad_l = ppad_l
                            add_pad_r = ppad_r

                        patch, ppad_t, ppad_b, ppad_l, ppad_r = conv2d(patch, p_layer, ppad_t, ppad_b, ppad_l, ppad_r)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == DepthwiseConv2D:
                        patch, ppad_t, ppad_b, ppad_l, ppad_r = depthwise_conv2d(patch, p_layer, ppad_t, ppad_b, ppad_l, ppad_r)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == BatchNormalization:
                        patch = batch_norm2d(patch, p_layer)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == ReLU:
                        patch = relu(patch, p_layer)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == GlobalAveragePooling2D:
                        patch = gap2d(patch, p_layer)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == Add:
                        inputs = [residual, patch]
                        patch = add(inputs, p_layer, add_pad_t, add_pad_b, add_pad_l, add_pad_r)
                        residual = None
                        add_pad_t, add_pad_b, add_pad_l, add_pad_r = 0, 0, 0, 0
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    elif op == Dense:
                        patch = dense(patch, p_layer)
                        log_file.write(f"{y_idx} / {x_idx} {op} \n{patch}\n")
                    else:
                        raise NotImplementedError(f"{layer_idx} layer {op} is Not implemt")
            to_concat_height.append(tf.concat(to_concat_width, axis=2))
            #print('result -->', to_concat_height[-1].shape) 
        x = tf.concat(to_concat_height, axis=1)
        print(x.shape)
        for p_layer in self.layer_params[start_layer_per_layer:]:
            assert ('is_patch_layer' not in p_layer), "Error"
            op = p_layer['op']
            if op == Conv2D:
                if 'use_residual' in p_layer:
                    residual = x
                x = conv2d(x, p_layer)
                log_file.write(f"after pating {op}\n{x}\n")
            elif op == DepthwiseConv2D:
                x = depthwise_conv2d(x, p_layer)
            elif op == BatchNormalization:
                x = batch_norm2d(x, p_layer)
            elif op == ReLU:
                x = relu(x, p_layer)
                log_file.write(f"after pating {op}\n{x}\n")
            elif op == GlobalAveragePooling2D:
                x = gap2d(x, p_layer)
            elif op == Add:
                inputs = [residual, x]
                x = add(inputs, p_layer)
                log_file.write(f"after pating {op}\n{x}\n")
                residual = None
            elif op == Dense:
                x = dense(x, p_layer)
                log_file.write(f"after pating {op}\n{x}\n")
            else:
                raise NotImplementedError(f"{layer_idx} layer {op} is Not implemt")
        return x



skip = False
f_model = MBV2_MCU(layer_params)
f_model.set_split_patch(13)
f_model.set_patch_split(2, 13, 6)

patch_testing= """
mcunet_idx = 0
for p_layer in f_model.layer_params:
    op = p_layer['op']
    op_string = f_model.get_type(op)
    string = f"\t\t{op_string}"
    if op in [Conv2D, DepthwiseConv2D, Add]:
        string = f"layer {mcunet_idx}:{op_string}"
        mcunet_idx += 1
    if 'is_patch_layer' in p_layer:
        string += " :patched"
    print(string)
exit()
"""
data = np.random.randn(1,224, 224, 3).astype(np.float32)
#print(data[0])
out2 = model(np.copy(data))
out1 = f_model(np.copy(data))
print(np.allclose(out1, out2, atol=1e-4, rtol=1e-4))
#print(out1.shape)
#print(out2.shape)

print('f_model\n',out1[0, :20])
print('model\n',out2[0, :20])
log_file.close()
exit()

f_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0, momentum=0.0),
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
eval_result = f_model.evaluate(normalized_val_ds)
