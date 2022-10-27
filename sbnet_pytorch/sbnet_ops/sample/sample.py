"""

   Sparse Blocks Network
   Copyright (c) 2017, Uber Technologies, Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

#
# A minimal sample implementing a single sparse convolution layer with synthetic data using SBNet primitives.
#

import torch
import numpy as np
import sys

from torch.utils.cpp_extension import load
sbnet_path = '/root/sbnet/sbnet_pytorch/sbnet_ops'
sbnet= load(
    'sbnet', [
		sbnet_path + '/reduce_mask.cpp',
		sbnet_path + '/reduce_mask_cuda.cu',
		sbnet_path + '/sparse_gather.cpp',
		sbnet_path + '/pybind_modules.cpp',
	],
	verbose=True)
#help(sbnet)

def divup(a, b):
    return (a+b-1) // b

# Specify input tensor dimensions and block-sparsity parameters
batch = 4
hw = 256
channels = 64
blockSize = [16, 16]
blockStride = [14, 14]
blockOffset = [0, 0]
blockCount = [divup(hw, blockStride[0]), divup(hw, blockStride[1])]

# build kwargs to simplify op calls
inBlockParams = { "bsize": blockSize, "boffset": blockOffset, "bstride": blockStride }
outBlockParams = { "bsize": [blockSize[0]-2, blockSize[1]-2], "boffset": blockOffset, "bstride": blockStride }
#for i in range(100):
# create a random mask representing attention/a priori sparsity
# threshold the mask to a specified percentile sparsity
mask = np.random.randn(batch, blockCount[0], blockCount[1], channels).astype(np.float32)
#threshold = np.percentile(mask, 90)
#sparseMask = np.greater(mask, threshold).astype(np.float32)

# upsample the mask to full resolution
#upsampledMask = sparseMask.repeat(blockStride[0], axis=1).repeat(blockStride[1], axis=2)

# create a random input tensor
x = torch.from_numpy( np.random.randn(batch, hw, hw, channels).astype(np.float32) )
x = x.contiguous()

# create a random weight tensor
w = torch.from_numpy( np.random.randn(channels, channels, 3, 3).astype(np.float32) )
w = w.contiguous()

# reduce the mask to indices by using a fused pooling+indexing operation
mask = torch.from_numpy(mask)
#counts2, indices2 = sbnet.reduce_mask(mask.cuda(), blockCount, **inBlockParams, avgpool=True, tol=0.05)
counts1, indices1 = sbnet.reduce_mask(mask, blockCount, **inBlockParams, avgpool=True, tol=0.05)
print('counts:', int(counts1[0]))
print('indices size:', indices1.size())
#	indices1=indices1[:counts1[0]]
#	indices2=indices2[:counts2[0]]
#	if not torch.equal(counts1, counts2.cpu()) or \
#			not torch.equal(indices1, indices2.cpu()):
#		print('Possible error, please check:')
#		print('counts1', counts1.size(), counts1)
#		print('indices1', indices1.size(), indices1)
#		print('counts2', counts2.size(), counts2)
#		print('indices2', indices2.size(), indices2)


print('x size:', x.size())
# stack active overlapping tiles to batch dimension
blockStack = sbnet.sparse_gather(
    x, counts1, indices1, transpose=True, **inBlockParams)

print('blockStack size:', blockStack.size())

# perform dense convolution on a sparse stack of tiles
convBlocks = torch.nn.functional.conv2d(blockStack, w, padding='valid')
# write/scatter the tiles back on top of original tensor
# note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
validX = x[:, 1:hw-1, 1:hw-1, :]
y = sbnet.sparse_scatter(
    convBlocks, counts1, indices1,
    validX, **outBlockParams, transpose=True, add=False, atomic=False)
print('y1 size:', y.size())

# perform dense convolution on a sparse stack of tiles
convBlocks = torch.nn.functional.conv2d(blockStack, w, padding='same')
# write/scatter the tiles back on top of original tensor
# note that the output tensor is reduced by 1 on each side due to 'VALID' convolution
y = sbnet.sparse_scatter(
    convBlocks, counts1, indices1,
    x, **inBlockParams, transpose=True, add=False, atomic=False)
print('y2 size:', y.size())



sys.exit()


