/*

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

 */

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <iostream>

void LaunchReduceMaskGPU(
		torch::Tensor mask,               // Mask array.
		int N,                            // Batch dimension of the mask.
		int H,                            // Height of the mask.
		int W,                            // Width of the mask.
		float threshold,                  // Threshold for being active.
		int bOffsH0,                      // Block padding offset height, negative.
		int bOffsW0,                      // Block padding offset width, negative.
		int bSzH,                         // Block size height.
		int bSzW,                         // Block size width.
		int bStrH,                        // Block stride, height.
		int bStrW,                        // Block stride, width.
		int bCntH,                        // Number of blocks, height.
		int bCntW,                        // Number of blocks, width.
		unsigned int numBins,             // number of bins in binCounts
		unsigned int binSize,             // maximum size of each counter bin
		torch::Tensor activeBlockIndices, // triples of [n, ih, iw] indices for active blocks.
		torch::Tensor binCounts,          // Number of indices of active blocks.
		bool avgPool                      // true for avg pooling, false for max pooling
);


//#include <cuda_runtime_api.h>

// CPU implementation of reduce mask op.
// This is a naive CPU implementation, just for reference comparison/testing purposes.
template<typename scalar_t>
void ReduceMaskCPU(
		const scalar_t* mask,           // Mask array.
		int N,                          // Batch dimension of the mask.
		int H,                          // Height of the mask.
		int W,                          // Width of the mask.
		scalar_t threshold,                // Threshold for being active.
		int bOffsH0,                    // Block padding offset height, negative.
		int bOffsW0,                    // Block padding offset width, negative.
		int bSzH,                       // Block size height.
		int bSzW,                       // Block size width.
		int bStrH,                      // Block stride, height.
		int bStrW,                      // Block stride, width.
		int bCntH,                      // Number of blocks, height.
		int bCntW,                      // Number of blocks, width.
		int16_t* activeBlockIndices,   // Indices of active blocks. OUTPUT
		int32_t* binCounts,            // Number of active indices. OUTPUT
		bool avgPool)
{
	int count = 0;
	const int C = 1;
	for (int n = 0; n < N; ++n) {
		for (int bh = 0; bh < bCntH; ++bh) {
			for (int bw = 0; bw < bCntW; ++bw) {
				int h0 = bOffsH0 + bh * bStrH;
				int w0 = bOffsW0 + bw * bStrW;
				bool active = false; // Whether a block is active.
				scalar_t sum = 0.0f;
				for (int hh = std::max(0, h0); hh < h0 + bSzH && hh < H; ++hh) {
					for (int ww = std::max(0, w0); ww < w0 + bSzW && ww < W; ++ww) {
						for (int cc = 0; cc < C; cc++) {
							scalar_t val = mask[n*H*W*C + hh*W*C + ww*C + cc];
							if (avgPool)
								sum += val;
							else
								active |= (val > threshold);
				} } }
				if (avgPool)
					active = ( (sum/(bSzH*bSzW)) > threshold );
				if (active) {
					activeBlockIndices[count*3+0] = n;
					activeBlockIndices[count*3+1] = bh;
					activeBlockIndices[count*3+2] = bw;
					count++;
				}
	} } }
	*binCounts = count;
}

std::pair<torch::Tensor, torch::Tensor> ReduceMask(torch::Tensor mask,
		std::pair<int,int> bcount_dynamic,
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool avgpool_,
		float tol_) 
{

	// Grabs input shape.
	int N = mask.size(0);
	int H = mask.size(1);
	int W = mask.size(2);

	int bCntH = bcount_dynamic.first;
	int bCntW = bcount_dynamic.second;
	int bSzH = bsize_dynamic.first;
	int bSzW = bsize_dynamic.second;
	int bStrH = bstride_dynamic.first;
	int bStrW = bstride_dynamic.second;
	int bOffsH0 = boffset_dynamic.first;
	int bOffsW0 = boffset_dynamic.second;

	int maxIndices = N * bCntH * bCntW;

	auto mask_device = mask.device().type();
	auto tensor_options = torch::TensorOptions()
		.layout(torch::kStrided)
		.dtype(torch::kInt16)
		.device(mask_device)
		.requires_grad(false);

	torch::Tensor activeBlockIndices = torch::empty({maxIndices, 3}, tensor_options);

	unsigned int numBins = 1;
	unsigned int binSize = (maxIndices + numBins - 1) / numBins;

	torch::Tensor binCounts = torch::empty({numBins}, tensor_options.dtype(torch::kInt32));

	if (mask_device == torch::kCPU){
		assert(numBins == 1);
		AT_DISPATCH_FLOATING_TYPES(mask.scalar_type(), "ReduceMaskCPU", ([&] {
			ReduceMaskCPU<scalar_t>(
					mask.data_ptr<scalar_t>(),                // Mask array.
					N,                                        // Batch dimension of the mask.
					H,                                        // Height of the mask.
					W,                                        // Width of the mask.
					tol_,                                     // Threshold for being active.
					bOffsH0,                                  // Block padding offset height.
					bOffsW0,                                  // Block padding offset width.
					bSzH,                                     // Block size height.
					bSzW,                                     // Block size width.
					bStrH,                                    // Block stride, height.
					bStrW,                                    // Block stride, width.
					bCntH,                                    // Number of blocks, height.
					bCntW,                                    // Number of blocks, width.
					activeBlockIndices.data_ptr<int16_t>(),   // Indices of active blocks.
					binCounts.data_ptr<int32_t>(),            // Counts per bin of active blocks.
					avgpool_
			);
		}));
	}
	else{
		// call GPU equivalent
		LaunchReduceMaskGPU(
				mask,                                     // Mask array.
				N,                                        // Batch dimension of the mask.
				H,                                        // Height of the mask.
				W,                                        // Width of the mask.
				tol_,                                     // Threshold for being active.
				bOffsH0,                                  // Block padding offset height.
				bOffsW0,                                  // Block padding offset width.
				bSzH,                                     // Block size height.
				bSzW,                                     // Block size width.
				bStrH,                                    // Block stride, height.
				bStrW,                                    // Block stride, width.
				bCntH,                                    // Number of blocks, height.
				bCntW,                                    // Number of blocks, width.
				numBins,
				binSize,
				activeBlockIndices,                       // Indices of active blocks.
				binCounts,                                // Counts per bin of active blocks.
				avgpool_
		);
	}

	return {binCounts.cpu(), activeBlockIndices};
}

