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

#invpad slices the input instead of padding
def calc_block_params_and_padding_1d(bcount, insize, ksize, stride, deconv=False, invpad=False):
    pad = (0 if invpad else bcount) -((insize - ksize + stride) % bcount)
    if pad == bcount:
        pad = 0
    bsize = bsize_1d(bcount, insize + pad, ksize, stride)
    while (bsize - ksize) % stride != 0:
        pad += (-bcount if invpad else bcount)
        bsize = bsize_1d(bcount, insize + pad, ksize, stride)
    assert bsize.is_integer()
    bsize = int(bsize)
    bsize_out = calc_out_size_1d(1, bsize, ksize, stride, deconv=deconv)
    boffset = 0
    bstride =  bsize - ksize + stride
    return (bsize, bsize_out, boffset, bcount, bstride, abs(pad))

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

class SparseGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp_NHWC, redu_mask, bp, do_transpose):
        stacked_slices = sbnet.ops.sparse_gather(
            inp_NHWC,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=do_transpose)

        ctx.inp_size = tuple(inp_NHWC.size())
        ctx.redu_mask = redu_mask
        ctx.bp = bp
        ctx.do_transpose = do_transpose

        return stacked_slices

    @staticmethod
    def backward(ctx, grad_stacked_slices):
        redu_mask = ctx.redu_mask
        bp = ctx.bp
        do_transpose = ctx.do_transpose
        inp_size = ctx.inp_size

        scat_plane= sbnet.ops.sparse_scatter(
            grad_stacked_slices,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            inp_size,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            add=True,
            transpose=do_transpose,
            atomic=True)

        return scat_plane, None, None, None

class SparseScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, redu_mask, outp_sz, bp, do_add, do_transpose, do_atomic):
        scat_plane = sbnet.ops.sparse_scatter(
            inp,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            outp_sz,
            bsize=bp.bsize_out,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            add=do_add,
            transpose=do_transpose,
            atomic=do_atomic)

        ctx.redu_mask = redu_mask
        ctx.bp = bp
        ctx.do_transpose = do_transpose

        return scat_plane

    @staticmethod
    def backward(ctx, grad_scat_plane):
        redu_mask = ctx.redu_mask
        bp = ctx.bp
        do_transpose = ctx.do_transpose

        grad_stacked_slices = sbnet.ops.sparse_gather(
            grad_scat_plane,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            bsize=bp.bsize_out,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=do_transpose)

        return grad_stacked_slices, None, None, None, None, None, None

class SparseBlock_Conv2d_BN_ReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
            bias=True, bn_eps=1e-3, bn_momentum=0.01, bcount=None, 
            transpose=False, deconv=False):
        super(SparseBlock_Conv2d_BN_ReLU, self).__init__()

        self.invpad = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.deconv=deconv
        self.padding_params=None
        self.block_params=None
        self.bcount=bcount
        self.transpose=transpose
        if self.deconv:
            assert kernel_size == stride, 'Deconv supported only if the condition holds'
            conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, bias=bias)
        else:
            conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, bias=bias)
        self.conv_bn_relu = torch.nn.Sequential(
            # The convolution here will be always without padding.
            # If needed, padding will be done explicitly
            conv,
            torch.nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum),
            torch.nn.ReLU()
        )
        if not self.transpose:
            self.conv_bn_relu = self.conv_bn_relu.to(memory_format=torch.channels_last)


    # bcount : [H_bcount, W_bcount]
    def calibrate(self, inp_NHWC, bcount=None):
        assert bcount is not None or self.bcount is not None
        bcount = self.bcount if bcount is None else bcount
        N, C, H, W = list(inp_NHWC.size())
        H_bsize, H_bsize_out, H_boffset, H_bcount, H_bstride, H_pad = \
                calc_block_params_and_padding_1d(bcount[0], H, self.kernel_size, \
                self.stride, self.deconv, self.invpad)
        W_bsize, W_bsize_out, W_boffset, W_bcount, W_bstride, W_pad = \
                calc_block_params_and_padding_1d(bcount[1], W, self.kernel_size, \
                self.stride, self.deconv, self.invpad)
        # I am not sure whether calculating H and W params like this and simply
        # merging is ok, but let's try!
        if H_pad > 0 or W_pad > 0:
            pad_bottom, pad_top = H_pad//2+(H_pad%2), H_pad//2
            pad_right, pad_left = W_pad//2+(W_pad%2), W_pad//2
            self.padding_params = (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
 
        self.block_params= BlockParams(bsize=[H_bsize, W_bsize], bsize_out=[H_bsize_out, W_bsize_out],
                boffset=[H_boffset, W_boffset], bcount=bcount, bstrides=[H_bstride, W_bstride])
        print('Block params for input ', inp_NHWC.size(),' :', self.block_params)
        # Run the convolution once with full mask 
        redu_mask = gen_full_reducemask(self.block_params.bcount)

        # Try all input sizes for benchmarking
        while redu_mask.bin_counts.item() > 0:
            self.forward((inp_NHWC, redu_mask))
            redu_mask.bin_counts[0] -= 1

    # args are supposed to be: inp_NHWC, redu_mask, atomic=False, block_params=None):
    def forward(self, args):
        #print(args)
        inp_NHWC=args[0]
        redu_mask=args[1]
        atomic = args[2] if len(args)>2 else False
        block_params= args[3] if len(args)>3 else None

        #assert inp_NHWC.is_contiguous(memory_format=torch.channels_last)

        # If training, do regular convolution
        if self.block_params is None:
            # Calibration on the fly
            self.calibrate(inp_NHWC, self.bcount)

        bp = self.block_params if block_params is None else block_params
        #assert bp is not None, 'Block parameters should either be initalized or given'

        #torch.cuda.nvtx.range_push(f'SBC_inp_{list(inp_NHWC.size())}')
        if self.padding_params is not None:
            if self.invpad:
                wl, wr, hl, hr, cl, cr = self.padding_params
                inp_NHWC = inp_NHWC[..., hl:-hr, wl:-wr] # not sure if this involves copy or not
            else:
                inp_NHWC = torch.nn.functional.pad(inp_NHWC, self.padding_params)

        p = SparseGatherFunction.apply(inp_NHWC, redu_mask, bp, self.transpose)
#        p = sbnet.ops.sparse_gather(
#            inp_NHWC,
#            redu_mask.bin_counts,
#            redu_mask.active_block_indices,
#            bsize=bp.bsize,
#            bstride=bp.bstrides,
#            boffset=bp.boffset,
#            transpose=self.transpose)


        # Convolution on patches.
        q = self.conv_bn_relu(p)

        # Allocate output tensor as NHWC
        outp_sz = [int(inp_NHWC.size(0)), #N
            int(self.out_channels), #C
            int(calc_out_size_1d(bp.bcount[0], bp.bsize[0], 
                    self.kernel_size, self.stride, self.deconv)), #H
            int(calc_out_size_1d(bp.bcount[1], bp.bsize[1],
                    self.kernel_size, self.stride, self.deconv)), #W
        ]

        y = SparseScatterFunction.apply(q, redu_mask, outp_sz, bp, False, \
                self.transpose, atomic)
#        y = sbnet.ops.sparse_scatter(
#            q,
#            redu_mask.bin_counts,
#            redu_mask.active_block_indices,
#            outp_sz,
#            bsize=bp.bsize_out,
#            bstride=bp.bstrides,
#            boffset=bp.boffset,
#            add=False,
#            transpose=self.transpose,
#            atomic=atomic)
        #torch.cuda.nvtx.range_pop()

        #print(bp)
        #print('Sizes:', inp_NHWC.size(), p.size(), q.size(), y.size())

        return (y, redu_mask)
