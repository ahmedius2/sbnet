import time
from sbnet.layers import *
import sys

class Conv2d_BN_ReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', 
            bias=True):
        super(Conv2d_BN_ReLU, self).__init__()

        self.conv_bn_relu = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.conv_bn_relu(x)

def run_benchmark(func, args, name):
    print('Input  size:', args[0].size())
    with torch.no_grad():
        y_ = func(*args)
    print('Output size:', y_.size())

    # Benchmark it
    torch.cuda.synchronize()
    t1 = time.time()
    num_samples = 100
    with torch.no_grad():
        for i in range(num_samples):
            y_ = func(*args)
    torch.cuda.synchronize()
    t2 = time.time()
    tdiff = (t2-t1) * 1000.0 / num_samples
    print(f'Avrg {name} time: {tdiff} ms')
    return tdiff

N, C, H, W = 1, 64, 256, 256
x = torch.rand((N, H, W, C)).cuda()
C_out, C_in, kH, kW = 64, 64, 3, 3
w = torch.rand((C_out, C_in, kH, kW)).cuda()
#bsize, strides, padding = [1, 17, 17, 1], [1, 1, 1, 1], 'valid'
bsize, strides, padding = [1, 15, 15, 1], [1, 2, 2, 1], 'valid'
#bsize, strides, padding = [1, 32, 32, 1], [1, 1, 1, 1], 'same'

##################################
# Regular convolution benchmark
##################################
cbnr = Conv2d_BN_ReLU(C_in, C_out, kH, strides[1], padding, bias=False).cuda()

x_ = x.permute(0, 3, 1, 2).contiguous()
# invoke cudnn benchmark initially
run_benchmark(cbnr, (x_,), 'regular convolution')
##################################


########################################
# Sparse block convolution benchmark
########################################
flipped_wsize = list(w.size())[::-1]
block_params = calc_block_params(list(x.size()), bsize, flipped_wsize, strides, padding)
print('block_params:', block_params)

##################################
# Actual reduce mask generation
#mask = torch.rand((N, block_params.bcount[0], block_params.bcount[1], 1)).cuda()
#tol, avgpool = 0.05, True
#rm = convert_mask_to_indices(mask, block_params, tol, avgpool)
#print('reduce mask:', rm)
##################################

sb_cbnr = SparseBlock_Conv2d_BN_ReLU(C_in, C_out, kH, strides[1], padding, bias=False).cuda()

##################################
# Synthetic reduce mask generation
max_num_blocks= block_params.bcount[0] * block_params.bcount[1]
inds = torch.empty((max_num_blocks, 3), dtype=torch.int16)
for i in range(block_params.bcount[0]):
    for j in range(block_params.bcount[1]):
        inds[i * block_params.bcount[1] + j] = torch.tensor((0, i, j), dtype=torch.int16)
#print('inds', inds)
inds = inds.cuda()
counts = torch.ones(1, dtype=torch.int32)
rm = ReduceMask(inds, counts)
##################################

conv_times = []
for c in range(1,max_num_blocks+1):
    # invoke cudnn benchmark initially
    tdiff = run_benchmark(sb_cbnr, (x, rm, False, block_params), 
            f'sparse conv for {rm.bin_counts[0]} blocks')

    conv_times.append({'num_blocks': c, 'time_ms': round(tdiff,3)})
    rm.bin_counts[0] += 1

import csv
with open('timing_stats.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_blocks', 'time_ms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ct in conv_times:
        writer.writerow(ct)

