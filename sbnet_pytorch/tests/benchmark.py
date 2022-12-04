import time
import sys
import torch
from sbnet.layers import BlockParams, ReduceMask, \
        SparseBlock_Conv2d_BN_ReLU, gen_full_reducemask

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

N, C, H, W = 1, 128, 256, 256
x = torch.rand((N, H, W, C)).cuda()
C_out, C_in, ksize, stride = 128, C, 3, 1

##################################
# Regular convolution benchmark
##################################
cbnr = Conv2d_BN_ReLU(C_in, C_out, ksize, stride, 'same', bias=False).cuda()

x_ = x.permute(0, 3, 1, 2).contiguous()
run_benchmark(cbnr, (x_,), 'regular convolution')
##################################

########################################
# Sparse block convolution benchmark
########################################
sb_cbnr = SparseBlock_Conv2d_BN_ReLU(C_in, C_out, ksize, stride, bias=False).cuda()
bcount = [16, 16]
sb_cbnr.calibration(x, bcount)
rm = gen_full_reducemask(bcount)
conv_times = []
for c in range(1, bcount[0]*bcount[1]+1):
    rm.bin_counts[0] = c
    tdiff = run_benchmark(sb_cbnr, (x, rm, False), 
            f'sparse conv for {rm.bin_counts[0]} blocks')

    conv_times.append({'num_blocks': c, 'time_ms': round(tdiff,3)})

import csv
with open('timing_stats.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_blocks', 'time_ms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ct in conv_times:
        writer.writerow(ct)

