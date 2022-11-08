import time
from sparse_conv_lib_torch import *

N, C, H, W = 1, 64, 256, 256
x = torch.rand((N, H, W, C)).cuda()
C_out, C_in, kH, kW = 64, 64, 3, 3
w = torch.rand((C_out, C_in, kH, kW)).cuda()
#bsize, strides, padding = [1, 17, 17, 1], [1, 2, 2, 1], 'valid'
bsize, strides, padding = [1, 32, 32, 1], [1, 1, 1, 1], 'same'

##################################
# Regular convolution benchmark
##################################
cbnr = Conv2d_BN_ReLU(C_in, C_out, kH, strides[1], padding, bias=False).cuda()
x_ = x.permute(0, 3, 1, 2).contiguous()

# invoke cudnn benchmark initially
with torch.no_grad():
    y_ = cbnr(x_)

# Benchmark it
torch.cuda.synchronize()
t1 = time.time()
with torch.no_grad():
    for i in range(100):
        y_ = cbnr(x_)
        #y_ = torch.nn.functional.conv2d(x_, w, None, strides[1:3], padding)
torch.cuda.synchronize()
t2 = time.time()
print(f'Avrg regular conv time: {(t2-t1)*1000.0/100.0} ms')
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
    with torch.no_grad():
        y = sb_cbnr(x, rm, atomic=False, block_params=block_params)

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(100):
        with torch.no_grad():
            y = sb_cbnr(x, rm, atomic=False, block_params=block_params)
    torch.cuda.synchronize()
    t2 = time.time()
    tdiff= (t2-t1)*1000.0/100.0
    print(f'Avrg conv time for {rm.bin_counts[0]} blocks: {tdiff} ms')
    conv_times.append({'num_blocks': c, 'time_ms': round(tdiff,3)})
    rm.bin_counts[0] += 1

import csv
with open('timing_stats.csv', 'w', newline='') as csvfile:
    fieldnames = ['num_blocks', 'time_ms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for ct in conv_times:
        writer.writerow(ct)

