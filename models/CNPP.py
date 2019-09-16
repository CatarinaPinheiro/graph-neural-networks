#!/usr/bin/python
# -*- coding: utf-8 -*-

from .MessageFunction import MessageFunction
from .UpdateFunction import UpdateFunction
from .ReadoutFunction import ReadoutFunction
from .GatLayer import GAT

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNPP(nn.Module):
    """
        MPNN as proposed by Gilmer et al..

        This class implements the whole Gilmer et al. model following the functions Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        hidden_state_size : int
            Size of the hidden states (the input will be padded with 0's to this size).
        message_size : int
            Message function output vector size.
        n_layers : int
            Number of iterations Message+Update (weight tying).
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(CNPP, self).__init__()

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers
        self.message_size = message_size

    def forward(self, g, h_in, e):

        print("g.size()", g.size())
        print("h_in.size()", h_in.size())
        print("e.size()", e.size())

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(
            torch.zeros(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data))], 2)

        print("h_t.size()", h_t.size())

        gat= GAT(g, self.in_n, self.args['out'], self.n_layers, self.message_size)
        return gat.forward(h_t)