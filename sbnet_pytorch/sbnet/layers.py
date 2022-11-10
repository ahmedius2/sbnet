import torch
import numpy as np
from collections import namedtuple
from sbnet.conv_dims import calc_padding_4d, calc_out_size_4d, calc_out_size_4d_np
import sbnet.ops
from math import floor
BlockParams = namedtuple('BlockParams', ['bsize', 'bsize_out', 'boffset', 'bcount', 'bstrides'])
ReduceMask = namedtuple('ReduceMask', ['active_block_indices', 'bin_counts'])

def _calc_block_strides(bsize, ksize, strides):
    """Calculates strides for blocks.

    :param bsize:     [list]        List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:     [list]        List of 4 int. Sparse convolution kernel size.
    :param strides:   [list]        List of 4 int. Sparse convolution strides.

    :return           [list]        List of 4 int. Block strides.
    """
    return [1, bsize[1] - ksize[0] + strides[1], bsize[2] - ksize[1] + strides[2], 1]

def calc_block_params(in_size, bsize, ksize, strides, padding):
    """
    Calculates block parameters for a single convolution layer.

    :param in_size:  [list]     List of 4 int, or a Tensor of size 4. Size of the convolution input.
    :param bsize:    [list]     List of 4 int. Size of blocks, or downsample ratio.
    :param ksize:    [list]     List of 4 int. Sparse convolution kernel size.
    :param strides:  [list]     List of 4 int. Sparse convolution stride size.
                                Currently only supports when,
                                1) (bsize[1] - ksize[0]) % strides[1] == 0 and,
                                2) (bsize[2] - ksize[1]) % strides[2] == 0
    :param padding:  [string]   `valid` or `same`, padding method for sparse convolution.

    :return          [tuple]
        bsize:
        bsize_out:
        boffset:
        bcount:
        bstrides:
    """
    static = not torch.is_tensor(in_size)

    assert ((bsize[1] - ksize[0]) % strides[1] == 0)
    assert ((bsize[2] - ksize[1]) % strides[2] == 0)

    bstrides = _calc_block_strides(bsize, ksize, strides)
    pad_h0, pad_h1, pad_w0, pad_w1 = calc_padding_4d(in_size, ksize, strides, padding)
    h = in_size[1]
    w = in_size[2]
    # Make padding divides blocks.
    pad_h1 += (-h + bsize[1]) % bstrides[1]
    pad_w1 += (-w + bsize[2]) % bstrides[2]
    boffset = [-pad_h0, -pad_w0]
    x_pad_shape = [
        in_size[0], in_size[1] + pad_h0 + pad_h1, in_size[2] + pad_w0 + pad_w1, in_size[3]
    ]
    if static:
        out_shape = calc_out_size_4d_np(x_pad_shape, [bsize[1], bsize[2], 1, 1], bstrides, 'valid')
    else:
        out_shape = calc_out_size_4d(x_pad_shape, [bsize[1], bsize[2], 1, 1], bstrides, 'valid')
    bcount = [out_shape[1], out_shape[2]]
    bsize_out = calc_out_size_4d_np(bsize, ksize, strides, 'valid')
    bsize = bsize[1:3]
    bstrides = bstrides[1:3]
    bsize_out = bsize_out[1:3]
    if static:
        assert (pad_h0 == -boffset[0])
        assert (pad_w0 == -boffset[1])
        for i, siz in zip([0, 1], [h, w]):
            # make sure last block is inside
            err_msg = 'Making sure last block is inside boffset' \
                ' {} bstrides {} bcount {} size {}'.format(
                boffset[i], bstrides[i], bcount[i], siz)
            assert (boffset[i] + bstrides[i] * (bcount[i] - 1) < siz), err_msg
    return BlockParams(
        bsize=bsize, bsize_out=bsize_out, boffset=boffset, bcount=bcount, bstrides=bstrides)


def convert_mask_to_indices(mask, block_params, tol, avgpool=False):
    """
    Converts a binary mask to sparse index format for custom CUDA kernel and TF ops.

    :param mask:         [Tensor]   [N, H, W]. 1 indicates non-sparse locations. Dtype float32.
    :param block_params  [tuple]    Contains bsize, boffset, bcount, bstrides.
    :param tol:          [float]    Lower bound of occupancy for creating a rectangle.

    :return          [tuple]
        bin_counts:           [Tensor]. Number of active locations for each bin.
        active_block_indices: [Tensor]. [M]. Center locations of M rectangles. Dtype int64.
    """

    def to_tensor(a, dtype):
        if torch.is_tensor(a):
            if a.dtype != dtype:
                return a.to(dtype)
            else:
                return a
        elif isinstance(a, list):
            if torch.is_tensor(a[0]):
                return torch.stack(a, 0)
            else:
                return torch.tensor(a, dtype=dtype)
        else:
            return torch.tensor(a, dtype=dtype)

    counts, indices = sbnet.ops.reduce_mask(
        mask,
        block_params.bcount,
        bsize=block_params.bsize,
        bstride=block_params.bstrides,
        boffset=block_params.boffset,
        avgpool=avgpool,
        tol=tol)
    return ReduceMask(indices, counts)

# Input x will be in NHWC format
# w : (out_channels, in_channels/groups, kH, kW)
def sparse_conv2d(x,
         w,
         indices,
         block_params,
         strides,
         transpose=True,
         atomic=False):
    assert transpose, 'Only available when transpose is True'
    # TODO calc ksize
    p = sbnet.ops.sparse_gather(
        x,
        indices.bin_counts,
        indices.active_block_indices,
        bsize=block_params.bsize,
        bstride=block_params.bstrides,
        boffset=block_params.boffset,
        transpose=transpose)

    # Convolution on patches.
    q = torch.nn.functional.conv2d(p, w, stride=strides[1:3], padding='valid')

    # Allocate output tensor.
    if strides[1] > 1 or strides[2] > 1:
        x = torch.empty((x.size(0), x.size(1)//strides[1], x.size(2)//strides[2], x.size(3)),
                device=x.device)

    y = sbnet.ops.sparse_scatter(
        q,
        indices.bin_counts,
        indices.active_block_indices,
        x,
        bsize=block_params.bsize_out,
        bstride=block_params.bstrides,
        boffset=(0, 0), # why 0 0?
        add=False,
        transpose=transpose,
        atomic=atomic)

    return y

class SparseBlock_Conv2d_BN_ReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', 
            bias=True):
        super(SparseBlock_Conv2d_BN_ReLU, self).__init__()

        assert (padding == 'same' and stride == 1) or padding == 'valid'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_bn_relu = torch.nn.Sequential(
            # The convolution here will be always without padding.
            # If needed, padding will be done by sparse gather.
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=0, bias=bias),
            torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            torch.nn.ReLU()
        )

        # Computing this requires knowing the input size
        self.block_params=None

    def init_block_params(self):
        pass

    def forward(self, x, indices, atomic=False, block_params=None):
        bp = self.block_params if block_params is None else block_params
        assert bp is not None, 'Block parameters should either be initalized or given'

        p = sbnet.ops.sparse_gather(
            x,
            indices.bin_counts,
            indices.active_block_indices,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=True)

        # Convolution on patches.
        q = self.conv_bn_relu(p)

        # Allocate output tensor
        # Assumes dilation is 1
        outp_sz = list(x.size())
        if self.padding == 'valid':
            ksz = self.kernel_size
            outp_sz = [outp_sz[0],
                    floor(float(outp_sz[1] - (ksz-1) - 1) / self.stride + 1),
                    floor(float(outp_sz[2] - (ksz-1) - 1) / self.stride + 1),
                    outp_sz[3]]
        out = torch.empty(outp_sz, device=x.device)

        y = sbnet.ops.sparse_scatter(
            q,
            indices.bin_counts,
            indices.active_block_indices,
            out,
            bsize=bp.bsize_out,
            bstride=bp.bstrides,
            boffset=(0, 0), # why 0 0?, 
            add=False,
            transpose=True,
            atomic=atomic)

        return y

#example bsize: [1, 16, 16, 1], ksize: [3, 3, in_C, out_C],
#strides: [1, 1, 1, 1], padding: 'same'

#example bsize: [1, 17, 17, 1], ksize: [3, 3, in_C, out_C],
#strides: [1, 2, 2, 1], padding: 'same'
