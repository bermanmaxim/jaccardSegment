# adapted from kaffe to load pytorch functional functions


import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_PADDING = 'SAME'
DEBUG_SIZES = False
DEBUG_NAMES = False

def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
            if DEBUG_SIZES: print(layer_input.size())
        else:
            layer_input = list(self.terminals)
            if DEBUG_SIZES: print([i.size() for i in layer_input])
        if DEBUG_NAMES: print(name)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

def pad_if_needed(input, padding, kind, k_h, k_w, s_h=1, s_w=1, dilation=1):
    if padding == 'VALID':
        return input
    elif padding == 'SAME' and kind in ('conv2d', 'pool2d'):
        in_height, in_width = input.size(2), input.size(3)
        out_height = int(np.ceil(float(in_height) / float(s_h)))
        out_width  = int(np.ceil(float(in_width) / float(s_w)))

        pad_along_height = max((out_height - 1) * s_h + k_h - in_height, 0)
        pad_along_width = max((out_width - 1) * s_w + k_w - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom))
        return input
    elif kind in ('atrous_conv2d',):
        effective_height = k_h + (k_h - 1) * (dilation - 1)
        effective_width = k_w + (k_w - 1) * (dilation - 1)
        return pad_if_needed(input, padding, 'conv2d', effective_height, effective_width, s_h, s_w, dilation=1)
    else:
        raise NotImplementedError



class Network(object):

    def __init__(self, inputs, weights, trainable=True, is_training=False):
        # The input nodes for this network
        self.inputs = inputs
        self.weights = weights
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = 1.0
        self.setup(is_training)

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h, 
             k_w, 
             c_o, 
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        input = pad_if_needed(input, padding, 'conv2d', k_h, k_w, s_h, s_w)

        result = F.conv2d(input, 
                          self.weights[name + '/weights'], 
                          bias=self.weights[name + '/biases'] if biased else None,
                          padding=0, 
                          groups=group,
                          stride=(s_h, s_w))
        if relu:
            result = F.relu(result)
        return result

    @layer
    def atrous_conv(self,
                    input,
                    k_h, 
                    k_w, 
                    c_o, 
                    dilation,
                    name,
                    relu=True,
                    padding=DEFAULT_PADDING,
                    group=1,
                    biased=True):
        if group != 1:
            raise NotImplementedError
        input = pad_if_needed(input, padding, 'atrous_conv2d', k_h, k_w, dilation=dilation)

        result = F.conv2d(input, 
                          self.weights[name + '/weights'], 
                          bias=self.weights[name + '/biases'] if biased else None,
                          padding=0, 
                          dilation=dilation,
                          groups=group,
                          stride=1)
        if relu:
            result = F.relu(result)
        return result
        
    @layer
    def relu(self, input, name):
        return F.relu(input)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        input = pad_if_needed(input, padding, 'pool2d', k_h, k_w, s_h, s_w)

        return F.max_pool2d(input,
                              kernel_size=(k_h, k_w),
                              stride=(s_h, s_w),
                              padding=0)

    @layer
    def add(self, inputs, name):
        return sum(inputs)
        
    @layer
    def batch_normalization(self, input, # other arguments are ignored
                            name, is_training, activation_fn=None, scale=True, eps=0.001):
        output = F.batch_norm(input, self.weights[name + '/moving_mean'], self.weights[name + '/moving_variance'],
                              weight=self.weights[name + '/gamma'], bias=self.weights[name + '/beta'], eps=eps)
        if activation_fn is not None:
            if activation_fn == 'relu':
                output = F.relu(output)
            else:
                raise NotImplementedError
        return output