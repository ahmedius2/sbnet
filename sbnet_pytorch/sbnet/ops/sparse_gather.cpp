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
#include <mutex>
#include <omp.h>
#include <vector>

torch::Tensor LaunchSparseGatherGPU(
        torch::Tensor x, int N, int H, int W, int C,
        torch::Tensor y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, torch::Tensor activeBlockIndices, bool transpose);

torch::Tensor LaunchSparseScatterGPU(
        torch::Tensor x, int N, int H, int W, int C, torch::Tensor y,
        int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
        int numActive, torch::Tensor activeBlockIndices, bool add,
		bool transpose, bool atomic);

// CPU specialization of actual computation.
// This is a naive CPU implementation, just for testing purpose.
template <typename scalar_t>
void SparseGatherCPU(
		const scalar_t* x, int N, int H, int W, int C, scalar_t* y,
		int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
		int numActive, const int16_t* activeBlockIndices, bool transpose)
{
    const int R = bSzH, S = bSzW;
    #pragma omp parallel for
    for (int ib = 0; ib < numActive; ib++) {
        int biN = activeBlockIndices[ib*3+0];
        int biH = activeBlockIndices[ib*3+1];
        int biW = activeBlockIndices[ib*3+2];
        int h0 = bOffsH0 + biH * bStrH;
        int w0 = bOffsW0 + biW * bStrW;
        for (int intraBh = 0; intraBh < R; ++intraBh) {
        for (int intraBw = 0; intraBw < S; ++intraBw) {
        for (int cc = 0; cc < C; cc++) {
            int hh = h0 + intraBh;
            int ww = w0 + intraBw;
            scalar_t readVal = 0.0f;
            if (hh >= 0 && ww >= 0 && hh < H && ww < W)
                readVal = x[biN*H*W*C + hh*W*C + ww*C + cc];
            if (transpose) // output to gathered blocks in NCHW
                y[ib*R*S*C + cc*R*S + intraBh*S + intraBw] = readVal;
            else
                y[ib*R*S*C + intraBh*S*C + intraBw*C + cc] = readVal;
        } } }
    }
}

// CPU specialization of actual computation.
// This is a naive CPU implementation, just for testing purpose.
template <typename scalar_t>
void SparseScatterCPU(
		const scalar_t* x, int N, int H, int W, int C, scalar_t* y,
		int bOffsH0, int bOffsW0, int bSzH, int bSzW, int bStrH, int bStrW,
		int numActive, const int16_t* activeBlockIndices, bool add, bool transpose, bool atomic)
{
    omp_lock_t writeLock;
    omp_init_lock(&writeLock);

    const int R = bSzH, S = bSzW;
    #pragma omp parallel for
    for (int ib = 0; ib < numActive; ib++) {
        int biN = activeBlockIndices[ib*3+0];
        int biH = activeBlockIndices[ib*3+1];
        int biW = activeBlockIndices[ib*3+2];
        for (int intraBh = 0; intraBh < R; ++intraBh) {
        for (int intraBw = 0; intraBw < S; ++intraBw) {
        for (int cc = 0; cc < C; cc++) {
            int h0 = bOffsH0 + biH * bStrH;
            int w0 = bOffsW0 + biW * bStrW;
            int hh = h0 + intraBh;
            int ww = w0 + intraBw;
            scalar_t readVal;
            if (transpose)
                readVal = x[ib*R*S*C + cc*R*S + intraBh*S + intraBw];
            else
                readVal = x[ib*R*S*C + intraBh*S*C + intraBw*C + cc];
            if (hh >= 0 && ww >= 0 && hh < H && ww < W) {
                if (add) {
                    omp_set_lock(&writeLock);
                    y[biN * H * W * C + hh * W * C + ww * C + cc] += readVal;
                    omp_unset_lock(&writeLock);
                } else
                    y[biN*H*W*C + hh*W*C + ww*C + cc] = readVal;
            }
        } } }
    }
}

torch::Tensor SparseGather(torch::Tensor x,
		torch::Tensor bin_counts_tensor,
		torch::Tensor activeBlockIndices,
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool transpose_)
{
	int bSzH = bsize_dynamic.first;
	int bSzW = bsize_dynamic.second;
	int bStrH = bstride_dynamic.first;
	int bStrW = bstride_dynamic.second;
	int bOffsH0 = boffset_dynamic.first;
	int bOffsW0 = boffset_dynamic.second;

    // read the number of active blocks from bin_counts input that is expected to be always in host mem
	int32_t bin0Count = bin_counts_tensor[0].item<int32_t>();

	// Grabs input shape. It's size appears as NCHW but the layout supposed to be NHWC
	int N = x.size(0);
	int C = x.size(1);
	int H = x.size(2);
	int W = x.size(3);

	// Initializes output.
    auto outp_mem_format = torch::MemoryFormat::ChannelsLast;
	if (transpose_){
        outp_mem_format = torch::MemoryFormat::Contiguous;
    }
	std::vector<int64_t> yShapeArr{ bin0Count, C, bSzH, bSzW};
	auto x_device = x.device().type();
	auto tensor_options = torch::TensorOptions()
		.layout(torch::kStrided)
		.dtype(x.scalar_type())
		.device(x_device)
		.requires_grad(x.requires_grad())
        .memory_format(outp_mem_format);
	torch::Tensor y = torch::zeros(yShapeArr, tensor_options);

	if (x_device == torch::kCPU){
		AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "SparseGatherCPU", ([&] {
			SparseGatherCPU<scalar_t>(
				x.data_ptr<scalar_t>(), N, H, W, C,
				y.data_ptr<scalar_t>(),
				bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
				bin0Count, activeBlockIndices.data_ptr<int16_t>(),
				transpose_);
		}));
	}
	else{
		y = LaunchSparseGatherGPU(
			x, N, H, W, C, y, bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
			bin0Count, activeBlockIndices,transpose_);
	}

	return y;	
}

torch::Tensor SparseScatter(torch::Tensor x,
		torch::Tensor bin_counts_tensor,
		torch::Tensor activeBlockIndices,
		std::vector<int64_t> out_size, // output tensor size
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool transpose_,
		bool add_,
		bool atomic_)
{
        // if transpose is true:  x is read as NCHW
        // if transpose is false: x is read as NHWC
        // ybase is treated as NHWC always

        auto x_device = x.device().type();
        auto tensor_options = torch::TensorOptions()
            .layout(torch::kStrided)
            .dtype(x.scalar_type())
            .device(x_device)
            .requires_grad(x.requires_grad());
        //    .memory_format(torch::MemoryFormat::ChannelsLast); // not working
        torch::Tensor outp = torch::zeros(out_size, tensor_options);
	outp = outp.to(torch::MemoryFormat::ChannelsLast);

        int N = outp.size(0);
        int C = outp.size(1);
        int H = outp.size(2);
        int W = outp.size(3);

		int bSzH = bsize_dynamic.first;
		int bSzW = bsize_dynamic.second;
		int bStrH = bstride_dynamic.first;
		int bStrW = bstride_dynamic.second;
		int bOffsH0 = boffset_dynamic.first;
		int bOffsW0 = boffset_dynamic.second;

        if (!atomic_ && add_) {
			// TODO PRINT SOME ERROR
        }
        
		// read the number of active blocks from bin_counts input that is expected to be always in host mem
		int32_t bin0Count = bin_counts_tensor[0].item<int32_t>();

        // Splat/add x on top of y
		// We have outp allocated, use it as output
		if (x_device == torch::kCPU){
			AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "SparseScatterCPU", ([&] {
				SparseScatterCPU(
					x.data_ptr<scalar_t>(), N, H, W, C, outp.data_ptr<scalar_t>(),
					bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
					bin0Count, activeBlockIndices.data_ptr<int16_t>(),
					add_, transpose_, atomic_
				);
			}));
		}
		else{
			outp = LaunchSparseScatterGPU(x, N, H, W, C, outp,
					bOffsH0, bOffsW0, bSzH, bSzW, bStrH, bStrW,
					bin0Count, activeBlockIndices, add_, transpose_, atomic_);
		}

		return outp;
}

