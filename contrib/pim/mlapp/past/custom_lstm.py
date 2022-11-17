import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import time
from hwcounter import count

gemv_count = 0

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False, count='0'):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.count = count

        self.input_weights = nn.Linear(1, 1, False)
        self.hidden_weights = nn.Linear(hidden_size, 4* hidden_size, bias)

        self.reset_params()

    def _set_input_to_hidden(self):
        self.input_weights = nn.Linear(input_size, 4* hidden_size, False)

    def reset_params(self):
        k = 1 / self.hidden_size

        for weight in self.input_weights.parameters():
            nn.init.uniform_(weight.data, -(k**0.5), k**0.5)

        for weight in self.hidden_weights.parameters():
            nn.init.uniform_(weight.data, -(k**0.5), k**0.5)

        if self.hidden_weights.bias is not None:
            nn.init.zeros_(self.hidden_weights.bias)

    def forward(self, inputs, hidden, pre_cal):
        hx, cx = hidden

        if not pre_cal:
            _set_input_to_hidden()
            in2hid = self.input_weights(inputs)
            hid2hid = self.hidden_weights(hx)
        else:
            in2hid = inputs
            start = time.time() if self.count=='0' else count()
            hid2hid = self.hidden_weights(hx)
            end = time.time() if self.count=='0' else count()
        gates = in2hid + hid2hid
        chunked_gate = gates.chunk(4, 1)

        in_gate = chunked_gate[0].sigmoid()
        forget_gate = chunked_gate[1].sigmoid()
        cell_gate = chunked_gate[2].tanh()
        out_gate = chunked_gate[3].sigmoid()

        cy = (forget_gate *cx) + (in_gate *cell_gate)
        hy = out_gate * cy.tanh()

        return (hy, cy), end-start

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=False, dropout=0.2, count='0'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.count = count
        self.gemv_count = 0

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i+1)
            if i == 0:
                cell = self._set_first_layer_cell()
            else:
                cell = self._set_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def _set_first_layer_cell(self):
        return LSTMCell(self.input_size, self.hidden_size, self.bias, self.count)

    def _set_other_layer_cell(self):
        return LSTMCell(self.hidden_size, self.hidden_size, self.bias, self.count)

    def _set_input2hidden(self):
        self.input_weights = nn.Linear(self.input_size, 4 * self.hidden_size, False)

    def set_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return (h, c)

    def set_drop(self, inputs, p):
        return PackedSequence(self.dropout(inputs.data, p), inputs.batch_sizes)

    def hidden_slice(self, hidden, start, end):
        if isinstance(hidden, torch.Tensor):
            return hidden.narrow(0, start, end -start)
        elif isinstance(hidden, tuple):
            return (hidden[0].narrow(0, start, end - start), hidden[1].narrow(0, start, end - start))
        else:
            raise TypeError

    def hidden_as_output(self, hidden):
        if isinstance(hidden, torch.Tensor):
            return hidden
        elif isinstance(hidden, tuple):
            return hidden[0]
        else:
            raise TypeError
    
    def padded_layer(self, inputs, hidden, num_cell, pre_cal):
        step_inputs = inputs.unbind(0)
        step_outputs = []
        cell = self._all_layers[num_cell]
        
        gemv_count = 0
        for step_input in step_inputs:
            hidden, time = cell(step_input, hidden, pre_cal)
            step_outputs.append(self.hidden_as_output(hidden))
            gemv_count = gemv_count + time

        return torch.stack(step_outputs, 0), hidden, gemv_count
    
    def forward(self, inputs, init_hidden=None):
        start = time.time() if self.count=='0' else count()
        hidden = self.set_hidden(inputs.size(1), inputs.device) if init_hidden is None else init_hidden
        _hiddens = [hidden] *self.num_layers
        step_hidden = _hiddens[0]

        if inputs.device == torch.device('cpu'):
            self._set_input2hidden()
            seq_len = inputs.size(0)
            batch_size = inputs.size(1)
            inputs = inputs.view(seq_len * batch_size, -1)
            end = time.time() if self.count=='0' else count()
            print("Ready2Compute\t%.6f" %(end - start))
 
            start = time.time() if self.count=='0' else count()
            out = self.input_weights(inputs)
            layer_input = out.view(seq_len, batch_size, -1)
            end = time.time() if self.count=='0' else count()
            print("input pre_cal\t%.6f" %(end - start))

            pre_cal = True
        else:
            layer_input = inputs
            pre_cal = False

        final_hy = []
        final_cy = []

        self.gemv_count = 0
        for i in range(self.num_layers):
            layer_output, final_hidden, gemv_count = self.padded_layer(layer_input, step_hidden, i, pre_cal)
            hy, cy = final_hidden
            final_hy.append(hy)
            final_cy.append(cy)
            layer_input = layer_output
            self.gemv_count = gemv_count

            if (layer_input.requires_grad and self.dropout_p != 0 and i < self.num_layers -1):
                layer_input = self.dropout(layer_input, self.dropout_p)

        print("GEMV compute\t%.6f" %self.gemv_count)
        print("---")

        return layer_input, (torch.stack(final_hy, 0), torch.stack(final_cy, 0))
