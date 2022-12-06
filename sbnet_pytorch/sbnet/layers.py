import torch
import numpy as np
from collections import namedtuple
import sbnet.ops
from math import floor
BlockParams = namedtuple('BlockParams', ['bsize', 'bsize_out', 'boffset', 'bcount', 'bstrides'])
ReduceMask = namedtuple('ReduceMask', ['active_block_indices', 'bin_counts'])

# if this returns an integer, it is successful
def bsize_1d(bcount, insize, ksize, stride):
    bsize = (insize - ksize + stride) / bcount + ksize - stride
    return bsize

# Don't worry about this too much, we can pad the output
# if the output size is not the desired one.
def calc_out_size_1d(bcount, bsize, ksize, stride, deconv=False):
    out_size = (bsize-1)*stride+ksize if deconv else ((bsize-ksize)/stride+1)
    return int(bcount * out_size)

def calc_block_params_and_padding_1d(bcount, insize, ksize, stride, deconv=False):
    pad = bcount-((insize - ksize + stride) % bcount)
    if pad == bcount:
        pad = 0
    bsize = bsize_1d(bcount, insize + pad, ksize, stride)
    while (bsize - ksize) % stride != 0:
        pad += bcount
        bsize = bsize_1d(bcount, insize + pad, ksize, stride)
    assert bsize.is_integer()
    bsize = int(bsize)
    bsize_out = calc_out_size_1d(1, bsize, ksize, stride, deconv=deconv)
    boffset = 0
    bstride =  bsize - ksize + stride
    return (bsize, bsize_out, boffset, bcount, bstride, pad)

def gen_full_reducemask(bcount):
    max_num_blocks= bcount[0] * bcount[1]
    inds = torch.empty((max_num_blocks, 3), dtype=torch.int16)
    for i in range(bcount[0]):
        for j in range(bcount[1]):
            inds[i * bcount[1] + j] = torch.tensor((0, i, j), dtype=torch.int16)
    #print('inds', inds)
    inds = inds.cuda()
    counts = torch.full((1,), max_num_blocks, dtype=torch.int32)
    return ReduceMask(inds, counts)

#counts, indices = sbnet.ops.reduce_mask(
#    mask,
#    block_params.bcount,
#    bsize=block_params.bsize,
#    bstride=block_params.bstrides,
#    boffset=block_params.boffset,
#    avgpool=avgpool,
#    tol=tol)
#return ReduceMask(indices, counts)

class SparseBlock_Conv2d_BN_ReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
            bias=True, bn_eps=1e-3, bn_momentum=0.01, bcount=None, 
            transpose_at_scatter=True, deconv=False):
        super(SparseBlock_Conv2d_BN_ReLU, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        if deconv:
            assert kernel_size == stride, 'Deconv supported only if the condition holds'
            conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, bias=bias)
        else:
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, bias=bias)
        self.deconv=deconv
        self.conv_bn_relu = torch.nn.Sequential(
            # The convolution here will be always without padding.
            # If needed, padding will be done explicitly
            conv,
            torch.nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
            torch.nn.ReLU()
        )

        # Computing this requires knowing the input size
        # Wait for calibration
        self.padding_params=None
        self.block_params=None
        self.bcount=bcount
        self.transpose_at_scatter=transpose_at_scatter

    # bcount : [H_bcount, W_bcount]
    def calibrate(self, inp_NHWC, bcount=None):
        assert bcount is not None or self.bcount is not None
        bcount = self.bcount if bcount is None else bcount
        N, H, W, C = list(inp_NHWC.size())
        H_bsize, H_bsize_out, H_boffset, H_bcount, H_bstride, H_pad = \
                calc_block_params_and_padding_1d(bcount[0], H, self.kernel_size, \
                self.stride, self.deconv)
        W_bsize, W_bsize_out, W_boffset, W_bcount, W_bstride, W_pad = \
                calc_block_params_and_padding_1d(bcount[1], W, self.kernel_size, \
                self.stride, self.deconv)
        # I am not sure whether calculating H and W params like this and simply
        # merging is ok, but let's try!
        if H_pad > 0 or W_pad > 0:
            pad_bottom, pad_top = H_pad//2+(H_pad%2), H_pad//2
            pad_right, pad_left = W_pad//2+(W_pad%2), W_pad//2
            self.padding_params = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
 
        self.block_params= BlockParams(bsize=[H_bsize, W_bsize], bsize_out=[H_bsize_out, W_bsize_out],
                boffset=[H_boffset, W_boffset], bcount=bcount, bstrides=[H_bstride, W_bstride])
        print('Block params for input ', inp_NHWC.size(),' :', self.block_params)
        # Run the convolution once with full mask 
        redu_mask = gen_full_reducemask(self.block_params.bcount)
        return self((inp_NHWC, redu_mask))

    # args are supposed to be: inp_NHWC, redu_mask, atomic=False, block_params=None):
    def forward(self, args):
        #print(args)
        inp_NHWC=args[0]
        redu_mask=args[1]
        atomic = args[2] if len(args)>2 else False
        block_params= args[3] if len(args)>3 else None
        if self.block_params is None:
            # Calibration on the fly
            return self.calibrate(inp_NHWC, self.bcount)

        bp = self.block_params if block_params is None else block_params
        #assert bp is not None, 'Block parameters should either be initalized or given'

        if self.padding_params is not None:
            inp_NHWC = torch.nn.functional.pad(inp_NHWC, self.padding_params)

        p = sbnet.ops.sparse_gather(
            inp_NHWC,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=True)

        # Convolution on patches.
        q = self.conv_bn_relu(p)

        # Allocate output tensor
        outp_sz = [inp_NHWC.size(0), #N
            calc_out_size_1d(bp.bcount[0], bp.bsize[0], 
                    self.kernel_size, self.stride, self.deconv), #H
            calc_out_size_1d(bp.bcount[1], bp.bsize[1],
                    self.kernel_size, self.stride, self.deconv), #W
            self.out_channels, #C
        ]

        if not self.transpose_at_scatter:
            outp_sz[1:] = [outp_sz[-1], outp_sz[1], outp_sz[2]]

        out = torch.empty(outp_sz, dtype=inp_NHWC.dtype, device=inp_NHWC.device)

        y = sbnet.ops.sparse_scatter(
            q,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            out,
            bsize=bp.bsize_out,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            add=False,
            transpose=self.transpose_at_scatter,
            atomic=atomic)

        return (y, redu_mask)
