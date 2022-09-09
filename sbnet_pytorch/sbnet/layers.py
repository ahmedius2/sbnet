import torch
import numpy as np
from collections import namedtuple
import sbnet.ops
from math import floor
import time
import gc
BlockParams = namedtuple('BlockParams', ['bsize', 'bsize_out', 'boffset', 'bcount', 'bstrides', 'padding'])
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

def calc_block_params(inp, bcount, ksize, stride, deconv=False, invpad=False):
    #N, C, H, W = list(inp.size())
    N, H, W, C = list(inp.size())
    H_bsize, H_bsize_out, H_boffset, H_bcount, H_bstride, H_pad = \
            calc_block_params_and_padding_1d(bcount[0], H, ksize, \
            stride, deconv, invpad)
    W_bsize, W_bsize_out, W_boffset, W_bcount, W_bstride, W_pad = \
            calc_block_params_and_padding_1d(bcount[1], W, ksize, \
            stride, deconv, invpad)

    padding_params=None
    if H_pad > 0 or W_pad > 0:
        pad_bottom, pad_top = H_pad//2+(H_pad%2), H_pad//2
        pad_right, pad_left = W_pad//2+(W_pad%2), W_pad//2
        #padding_params = (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
        padding_params = (0, 0, pad_left, pad_right, pad_top, pad_bottom)

    return BlockParams(bsize=[H_bsize, W_bsize], bsize_out=[H_bsize_out, W_bsize_out],
            boffset=[H_boffset, W_boffset], bcount=bcount, bstrides=[H_bstride, W_bstride],
            padding=padding_params)

def gen_reducemask(bcount, sparsity=0.):
    assert sparsity >= 0. and sparsity < 1.
    max_num_blocks= bcount[0] * bcount[1]
    inds = torch.empty((max_num_blocks, 3), dtype=torch.int16)
    for i in range(bcount[0]):
        for j in range(bcount[1]):
            inds[i * bcount[1] + j] = torch.tensor((0, i, j), dtype=torch.int16)
    inds = inds.cuda()
    counts = torch.full((1,), (bcount[0] * bcount[1]) * (1.0 - sparsity), \
            dtype=torch.int32)
    perms = torch.randperm(bcount[0] * bcount[1]) # shuffle
    return ReduceMask(inds[perms], counts)

class SparseGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, redu_mask, bp, do_transpose):
        stacked_slices = sbnet.ops.sparse_gather(
            inp,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=do_transpose)

        ctx.inp_size = tuple(inp.size())
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

        #if not grad_stacked_slices.is_contiguous():
        #    grad_stacked_slices = grad_stacked_slices.to(memory_format=torch.contiguous_format)

        grad_scat_plane= sbnet.ops.sparse_scatter(
            grad_stacked_slices,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            inp_size,
            bsize=bp.bsize,
            bstride=bp.bstrides,
            boffset=bp.boffset,
            transpose=do_transpose,
            add=True,
            atomic=True)

        #assert grad_scat_plane.is_contiguous(memory_format=torch.channels_last)

        return grad_scat_plane, None, None, None

class SparseScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, redu_mask, outp_sz, bp, do_add, do_transpose, do_atomic):
        scat_plane = sbnet.ops.sparse_scatter(
            inp,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            outp_sz,
            bsize=bp.bsize_out,
            bstride=bp.bsize_out,
            boffset=bp.boffset,
            transpose=do_transpose,
            add=do_add,
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

        #if not grad_scat_plane.is_contiguous(memory_format=torch.channels_last):
        #    grad_scat_plane = grad_scat_plane.to(memory_format=torch.channels_last)

        grad_stacked_slices = sbnet.ops.sparse_gather(
            grad_scat_plane,
            redu_mask.bin_counts,
            redu_mask.active_block_indices,
            bsize=bp.bsize_out,
            bstride=bp.bsize_out,
            boffset=bp.boffset,
            transpose=do_transpose)

        #assert grad_stacked_slices.is_contiguous()

        return grad_stacked_slices, None, None, None, None, None, None

# Abstract class
class SparseBlock(torch.nn.Module):
    def __init__(self, bcount=None, transpose=False):
        super().__init__()

        self.conv = None
        self.kernel_size = None
        self.stride = None
        self.out_channels = None
        self.deconv = None
        self.block_params = None
        self.bcount=bcount
        self.transpose=transpose
        self.do_gather = True
        self.do_scatter = True
        self.in_chain = False

        self.invpad = False

    # bcount : [H_bcount, W_bcount]
    def calibrate(self, inp, bcount=None):
        assert (bcount is not None or self.bcount is not None) and self.conv is not None
        bcount = self.bcount if bcount is None else bcount

        self.block_params = calc_block_params(inp, bcount, self.kernel_size, self.stride, \
                self.deconv, self.invpad)

        print('Calibrating block params for input ', inp.size(),' :', self.block_params)

        if not self.in_chain and not self.training:
            redu_mask = gen_reducemask(self.block_params.bcount)
            # Try all input sizes for benchmarking, takes a long time...
            while redu_mask.bin_counts.item() > 0:
                self.forward((inp, redu_mask))
                redu_mask.bin_counts[0] -= 1

    # args are supposed to be: inp, redu_mask, atomic=False, block_params=None):
    def forward(self, args):
        assert self.conv is not None

        inp=args[0]
        redu_mask=args[1]
        atomic = args[2] if len(args)>2 else False
        block_params= args[3] if len(args)>3 else None

        #assert inp.is_contiguous(memory_format=torch.channels_last)
        # If training, do regular convolution
        if block_params is None and self.block_params is None:
            # Calibration on the fly
            self.calibrate(inp, self.bcount)

        bp = self.block_params if block_params is None else block_params
        #assert bp is not None, 'Block parameters should either be initalized or given'
        #torch.cuda.nvtx.range_push(f'SBC_inp_{list(inp.size())}')

        if self.do_gather:
            padding_params = bp.padding
            if padding_params is not None:
                if self.invpad:
                    wl, wr, hl, hr, cl, cr = padding_params
                    inp = inp[..., hl:-hr, wl:-wr] # not sure if this involves copy or not
                else:
                    inp = torch.nn.functional.pad(inp, padding_params)

            p = SparseGatherFunction.apply(inp, redu_mask, bp, self.transpose)
        else:
            p = inp

#        # DEBUG check if gather works as expected
#        inds = redu_mask.active_block_indices.cpu()
#        slices=[]
#        for ind in inds:
#            xpos = ind[1]*bp.bstrides[0]
#            ypos = ind[2]*bp.bstrides[1]
#            slc = inp[ind[0]:(ind[0]+1), ..., xpos:(xpos+bp.bsize[0]), ypos:(ypos+bp.bsize[1])]
#            slices.append(slc)
#        slices = torch.cat(slices, dim=0)
#        if self.transpose:
#            slices = slices.to(memory_format=torch.contiguous_format)

#        if not torch.equal(p, slices):
#            print('SparseGather is not working properly!')
#            print('expected:', slices.size(), 'but got:', p.size())

#        for slc in p:
#            if torch.all(slc == 0).cpu().item():
#                print('A gathered slice is all zeros!')

        # Convolution on patches.
        q = self.conv(p)

        if self.do_scatter:
            # Allocate output tensor as NHWC
            # TODO batch size can be different than 1 in case of no gather
            batch_size = 1
            outp_sz = [batch_size, #N 
                int(calc_out_size_1d(bp.bcount[0], bp.bsize[0], 
                        self.kernel_size, self.stride, self.deconv)), #H
                int(calc_out_size_1d(bp.bcount[1], bp.bsize[1],
                        self.kernel_size, self.stride, self.deconv)), #W
                int(self.out_channels), #C
            ]

            y = SparseScatterFunction.apply(q, redu_mask, outp_sz, bp, False, \
                    self.transpose, atomic)
        else:
            y = q

#        # DEBUG check if scatter works as expected
#        scat_plane = torch.zeros(outp_sz, dtype=q.dtype, device=q.device)
#        for slc, ind in zip(q, inds):
#            xpos = ind[1]*bp.bsize_out[0]
#            ypos = ind[2]*bp.bsize_out[1]
#            scat_plane[ind[0], ..., xpos:(xpos+bp.bsize_out[0]), ypos:(ypos+bp.bsize_out[1])] = slc
#
#        if self.transpose:
#            scat_plane = scat_plane.to(memory_format=torch.channels_last)
#        if not torch.equal(y, scat_plane):
#            print('SparseScatter is not working properly!')
#            print('expected:', y.size(), 'but got:', scat_plane.size())


        #torch.cuda.nvtx.range_pop()

        return (y, redu_mask)

class SparseBlock_Conv2d(SparseBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1,
                bias=True, bcount=None, transpose=False, deconv=False):
        super().__init__(bcount, transpose)
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        self.deconv=deconv
        self.groups=groups

        if self.deconv:
            assert kernel_size == stride, 'Deconv supported only if the condition holds'
            self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, groups=self.groups, bias=bias)
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=0, groups=self.groups, bias=bias)

        #if not self.transpose:
        #    self.conv = self.conv.to(memory_format=torch.channels_last)

    def fill_bias(self, val):
        self.conv.bias.data.fill_(val)

    def forward(self, args):
        return super().forward(args)

class SparseBlock_Conv2d_BN_ReLU(SparseBlock_Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1,
            bias=True, bn_eps=1e-05, bn_momentum=0.1, norm_groups=None, bcount=None, 
            transpose=False, deconv=False):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                groups, bias, bcount, transpose, deconv)

        # maybe this is unnecessary, but anyway
        #if not self.transpose:
        #    self.conv = self.conv.to(memory_format=torch.contiguous_format)

        # if groups are more than one, use group convolution?
        if norm_groups is None:
            norm_groups = groups

        if norm_groups == 1:
            norm = torch.nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum)
        else:
            norm = torch.nn.GroupNorm(norm_groups, out_channels, eps=bn_eps)
        self.conv = torch.nn.Sequential(self.conv, norm, torch.nn.ReLU())

        #if not self.transpose:
        #    self.conv = self.conv.to(memory_format=torch.channels_last)
    
    def fill_bias(self, val):
        self.conv[0].bias.data.fill_(val)

    def forward(self, args):
        return super().forward(args)

class SparseBlock_ConvChain(torch.nn.Module):
    def __init__(self, sg_mask=None):
        super().__init__()

        self.conv_list = torch.nn.ModuleList()
        self.calibrated = False
        self.hardcoded_sg_mask = sg_mask

    def forward(self, args):
        #assert args[0].is_contiguous(memory_format=torch.channels_last)
        if not self.calibrated:
            self.calibrate(args)

        for conv in self.conv_list:
            args = conv.forward(args)
        return args

    def append_conv(self, conv_module):
        assert isinstance(conv_module, SparseBlock_Conv2d_BN_ReLU) or \
            isinstance(conv_module, SparseBlock_Conv2d)

        conv_module.in_chain = True
        self.conv_list.append(conv_module)

    def __update_block_params(self, sg_mask, default_bps, inp_sizes):
        # Based on sg_mask, determine gather/scatter operations
        # First layer always gathers
        num_convs = len(self.conv_list)
        self.conv_list[0].do_gather = True
        for j in range(num_convs-1):
            do_sg = (sg_mask & (1 << (num_convs-j-2)) != 0)
            self.conv_list[j].do_scatter  = do_sg
            self.conv_list[j+1].do_gather = do_sg
        # Last layer always scatters
        self.conv_list[-1].do_scatter = True

        # Reset block params first
        for conv, bp in zip(self.conv_list, default_bps):
            conv.block_params = bp
        # Detect the gs chains and update block params
        conv_idx  = 0
        while conv_idx < num_convs:
            if self.conv_list[conv_idx].do_gather:
                chain_end_idx = conv_idx
                while not self.conv_list[chain_end_idx].do_scatter:
                    chain_end_idx += 1
                bp = self.conv_list[chain_end_idx].block_params
                if conv_idx == chain_end_idx:
                    # Same layer gathers and scatters, proceed to next
                    #print(f'Block params for conv {conv_idx}: {bp}')
                    conv_idx += 1
                else:
                    # Chain detected, update the block params of gathering layer
                    bsize_out_unit, bsize_out_pair = 1, 2
                    bsize_arr = np.array((bsize_out_unit, bsize_out_pair, bp.bsize_out[0]))
                    for k in range(chain_end_idx, conv_idx-1, -1):
                        conv = self.conv_list[k]
                        bsize_arr = (bsize_arr - 1) * \
                                conv.stride + conv.kernel_size

                    # Assume block w and h are same
                    bsize = [int(bsize_arr[2])] * 2
                    boverlap = bsize_arr[1] - 2 * (bsize_arr[1] - bsize_arr[0])
                    bstrides = [bsize[0] - boverlap, bsize[1] - boverlap]
                    bsize_out = [int((bsize[0]-conv.kernel_size)/conv.stride+1),
                            int((bsize[1]-conv.kernel_size)/conv.stride+1)]

                    # Based on these, calculate the padding
                    inp_size = inp_sizes[conv_idx]
                    req_inp_size_H = (bp.bcount[0]  - 1) * bstrides[0] + bsize[0]
                    req_inp_size_W = (bp.bcount[1]  - 1) * bstrides[1] + bsize[1]
                    #H_pad = req_inp_size_H - inp_size[2]
                    #W_pad = req_inp_size_W - inp_size[3]
                    H_pad = req_inp_size_H - inp_size[1]
                    W_pad = req_inp_size_W - inp_size[2]
                    assert H_pad >= 0 and W_pad >= 0
                    if H_pad > 0 or W_pad > 0:
                        pad_bottom, pad_top = H_pad//2+(H_pad%2), H_pad//2
                        pad_right, pad_left = W_pad//2+(W_pad%2), W_pad//2
                        #padding_params = (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
                        padding_params = (0, 0, pad_left, pad_right, pad_top, pad_bottom)

                    # Override the block params
                    bp = BlockParams(bsize, bsize_out, bp.boffset,
                            bp.bcount, bstrides, padding_params)
                    self.conv_list[conv_idx].block_params = bp
                    conv_idx = chain_end_idx + 1 # proceed to next gather


    def calibrate(self, args):
        # First, run through the network to calculate default block params

        #assert args[0].is_contiguous(memory_format=torch.channels_last)
        inp_sizes = []
        default_bps = []
        initial_inp=args[0]
        for conv in self.conv_list:
            inp_sizes.append(args[0].size())
            args = conv.forward(args)
            default_bps.append(conv.block_params)

        self.calibrated=True # To prevent recursive calibrate call

        if self.hardcoded_sg_mask is not None:
            best_sg_mask = self.hardcoded_sg_mask
        else:
            # Now, try out different combinations by enabling/disabling scat/gather
            # First layer has to gather, last layer has to scatter
            pattern_timings=[]
            num_convs = len(self.conv_list)
            repetition, sparsities = 10,  (0.5, 0.7, 0.9)
            for sg_mask in range(2**(num_convs-1)):
                print('Benchmarking scatter/gather pattern:', sg_mask)
                self.__update_block_params(sg_mask, default_bps, inp_sizes)

                for i, conv in enumerate(self.conv_list):
                    print(f'Block params for conv {i}: {conv.block_params}, g:{conv.do_gather}, s:{conv.do_scatter}')
                # Benchmark the current chain config for different sparsities
                time_ms_per_sparsity=[]
                for sparsity in sparsities:
                    gc.collect()
                    # Block count is same for everyone
                    args = (initial_inp, gen_reducemask(default_bps[0].bcount, sparsity))
                    self.forward(args) # this call is to invoke cudnn benchmarking
                    torch.cuda.synchronize()
                    t1 = time.time()
                    for i in range(repetition):
                        self.forward(args)
                    torch.cuda.synchronize()
                    tdiff_ms = round((time.time()-t1)/repetition*1000,2)
                    time_ms_per_sparsity.append(tdiff_ms)
                print('Timings in ms for sparsities from 0.5 to 0.9:', time_ms_per_sparsity)
                pattern_timings.append(time_ms_per_sparsity)

            # What is left is to choose the pattern with best timing, also the pattern
            # can be different depending on sparsity, let's see...
            sparsity_and_pattern=[]
            for i, sparsity in enumerate(sparsities):
                timings = [p[i] for p in pattern_timings]
                min_time = min(timings)
                pattern = timings.index(min_time)
                sparsity_and_pattern.append((sparsity, pattern, min_time))
            print('Sparsities and best patterns:')
            print(sparsity_and_pattern)

            # Choose the pattern that is best overall
            best_sg_mask = np.bincount(np.array([sp[1] for sp in sparsity_and_pattern])).argmax()

        print('Decided to use the mask', best_sg_mask)
        self.__update_block_params(best_sg_mask, default_bps, inp_sizes)

        print('Calibrating the chain w.r.t. chosen mask...')
        redu_mask = gen_reducemask(default_bps[0].bcount)
        # Try all input sizes for benchmarking, takes a long time...
        while redu_mask.bin_counts.item() > 0:
            self.forward((initial_inp, redu_mask))
            redu_mask.bin_counts[0] -= 1


